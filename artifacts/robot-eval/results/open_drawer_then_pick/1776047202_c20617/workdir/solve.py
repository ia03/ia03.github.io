"""Heuristic controller for: open drawer then pick block (Franka Panda)."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _quat_to_small_angle(q_err: np.ndarray) -> np.ndarray:
    # For small errors: q_err ~= [1, 0.5*axis*angle].
    # Ensure shortest-path.
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


@dataclass
class PandaKinematics:
    hand_body_id: int
    left_finger_body_id: int
    right_finger_body_id: int
    q_nom: np.ndarray
    hand_quat_nom: np.ndarray


def _make_kin(sim: Sim) -> PandaKinematics:
    m, d = sim.model, sim.data
    hand_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_finger_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    # A safe-ish nominal configuration within actuator limits.
    q_nom = np.array([0.0, -0.55, 0.0, -1.85, 0.0, 1.35, 0.75], dtype=float)

    # Nominal hand orientation from current (after reset).
    hand_quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(hand_quat, d.xmat[hand_body_id])

    return PandaKinematics(
        hand_body_id=hand_body_id,
        left_finger_body_id=left_finger_body_id,
        right_finger_body_id=right_finger_body_id,
        q_nom=q_nom,
        hand_quat_nom=hand_quat.copy(),
    )


def _clip_to_actuator_ranges(sim: Sim, q: np.ndarray) -> np.ndarray:
    m = sim.model
    q = q.copy()
    for i in range(7):
        lo, hi = m.actuator_ctrlrange[i]
        q[i] = float(np.clip(q[i], lo, hi))
    return q


def _solve_dls(J: np.ndarray, e: np.ndarray, damping: float) -> np.ndarray:
    # dq = J^T (J J^T + λ^2 I)^-1 e
    JJ = J @ J.T
    JJ = JJ + (damping**2) * np.eye(JJ.shape[0], dtype=float)
    return J.T @ np.linalg.solve(JJ, e)


def _set_arm_target_pose(
    sim: Sim,
    kin: PandaKinematics,
    target_hand_pos: np.ndarray,
    target_hand_quat: np.ndarray,
    *,
    use_orientation: bool = True,
    pos_gain: float = 5.0,
    rot_gain: float = 2.5,
    damping: float = 0.08,
    null_gain: float = 0.25,
    max_dq: float = 0.045,
):
    m, d = sim.model, sim.data
    jacp = np.zeros((3, m.nv), dtype=float)
    jacr = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jacBody(m, d, jacp, jacr, kin.hand_body_id)
    Jp = jacp[:, :7]
    Jr = jacr[:, :7]

    cur_pos = d.xpos[kin.hand_body_id].copy()

    e_pos = (target_hand_pos - cur_pos) * pos_gain
    e_pos = np.clip(e_pos, -1.2, 1.2)

    if use_orientation:
        cur_quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(cur_quat, d.xmat[kin.hand_body_id])
        q_err = _quat_mul(target_hand_quat, _quat_conj(cur_quat))
        e_rot = _quat_to_small_angle(q_err) * rot_gain
        e_rot = np.clip(e_rot, -1.2, 1.2)
        J = np.vstack([Jp, Jr])
        e = np.hstack([e_pos, e_rot])
    else:
        J = Jp
        e = e_pos

    dq_task = _solve_dls(J, e, damping=damping)

    # Nullspace posture bias to keep IK well-behaved.
    task_dim = J.shape[0]
    JJT = (J @ J.T) + (damping**2) * np.eye(task_dim, dtype=float)
    J_pinv = J.T @ np.linalg.solve(JJT, np.eye(task_dim, dtype=float))
    N = np.eye(7, dtype=float) - J_pinv @ J
    dq_null = N @ (null_gain * (kin.q_nom - d.qpos[:7]))

    dq = dq_task + dq_null
    dq = np.clip(dq, -max_dq, max_dq)

    q_des = _clip_to_actuator_ranges(sim, d.qpos[:7] + dq)
    d.ctrl[:7] = q_des


def _run_policy(sim: Sim, *, save_path: str, render_dir: str | None = None):
    kin = _make_kin(sim)
    m, d = sim.model, sim.data

    # Controls: [7 arm position targets, 1 gripper tendon target, 1 drawer motor force]
    d.ctrl[:] = 0.0
    d.ctrl[:7] = _clip_to_actuator_ranges(sim, kin.q_nom)
    d.ctrl[7] = 0.0  # open gripper
    d.ctrl[8] = 0.0  # drawer motor force
    mujoco.mj_forward(m, d)

    def finger_midpoint() -> np.ndarray:
        return 0.5 * (d.xpos[kin.left_finger_body_id] + d.xpos[kin.right_finger_body_id])

    def hand_from_midpoint(target_mid: np.ndarray) -> np.ndarray:
        # Track the midpoint between fingers (changes slightly as gripper closes).
        cur_hand = d.xpos[kin.hand_body_id].copy()
        cur_mid = finger_midpoint()
        mid_offset = cur_mid - cur_hand
        return target_mid - mid_offset

    # Stage targets
    safe_hand = np.array([0.40, 0.0, 0.78], dtype=float)
    hover_extra_z = 0.20

    # Stage 1: open drawer fully while staying safe.
    early_saved = False
    for t in range(1200):
        d.ctrl[7] = 0.0  # keep open
        d.ctrl[8] = 0.9  # push drawer open

        _set_arm_target_pose(sim, kin, safe_hand, kin.hand_quat_nom, use_orientation=False, max_dq=0.08)

        sim.step(1)

        if (not early_saved) and sim.drawer_open_amount() >= 0.09 and t >= 200:
            # Mandatory early save: at least some progress with stage order intact.
            sim.save_final_state(save_path)
            early_saved = True

        if sim.drawer_open_amount() >= 0.135 and t >= 300:
            break

    # Let drawer + block settle a bit.
    for _ in range(200):
        d.ctrl[8] = 0.35
        _set_arm_target_pose(sim, kin, safe_hand, kin.hand_quat_nom, use_orientation=False, max_dq=0.08)
        sim.step(1)

    # Stage 2: hover above block (track x/y).
    for _ in range(900):
        d.ctrl[8] = 0.25
        block = sim.block_position()
        target_mid = np.array([block[0], block[1], block[2] + hover_extra_z], dtype=float)
        target_hand = hand_from_midpoint(target_mid)
        _set_arm_target_pose(sim, kin, target_hand, kin.hand_quat_nom, use_orientation=False, pos_gain=7.5, damping=0.06, max_dq=0.09, null_gain=0.18)
        sim.step(1)

    # Stage 3: descend to grasp height.
    grasp_xy = None
    for _ in range(650):
        d.ctrl[8] = 0.2
        block = sim.block_position()
        target_mid = np.array([block[0], block[1], block[2] + 0.005], dtype=float)
        target_hand = hand_from_midpoint(target_mid)
        _set_arm_target_pose(sim, kin, target_hand, kin.hand_quat_nom, use_orientation=False, pos_gain=8.0, damping=0.06, max_dq=0.09, null_gain=0.14)
        sim.step(1)
        if grasp_xy is None and np.linalg.norm(finger_midpoint()[:2] - block[:2]) < 0.03:
            grasp_xy = finger_midpoint()[:2].copy()

    if grasp_xy is None:
        grasp_xy = finger_midpoint()[:2].copy()

    # Stage 4: close gripper while holding pose.
    for k in range(350):
        d.ctrl[8] = 0.2
        block = sim.block_position()
        target_mid = np.array([grasp_xy[0], grasp_xy[1], block[2] + 0.003], dtype=float)
        target_hand = hand_from_midpoint(target_mid)
        _set_arm_target_pose(sim, kin, target_hand, kin.hand_quat_nom, use_orientation=False, pos_gain=8.5, damping=0.06, max_dq=0.08, null_gain=0.12)
        d.ctrl[7] = float(np.clip((k / 200.0) * 255.0, 0.0, 255.0))
        sim.step(1)

    # Stage 5: lift high and hold.
    lift_target_mid = np.array([grasp_xy[0], grasp_xy[1], 0.80], dtype=float)
    for _ in range(1200):
        d.ctrl[8] = 0.15
        target_hand = hand_from_midpoint(lift_target_mid)
        _set_arm_target_pose(sim, kin, target_hand, kin.hand_quat_nom, use_orientation=False, pos_gain=7.0, damping=0.06, max_dq=0.08, null_gain=0.10)
        d.ctrl[7] = 255.0
        sim.step(1)

    # Small hold tail so replay final state isn't a transient.
    for _ in range(300):
        d.ctrl[8] = 0.12
        target_hand = hand_from_midpoint(lift_target_mid)
        _set_arm_target_pose(sim, kin, target_hand, kin.hand_quat_nom, use_orientation=False, pos_gain=5.5, damping=0.06, max_dq=0.06, null_gain=0.08)
        d.ctrl[7] = 255.0
        sim.step(1)

    if render_dir:
        os.makedirs(render_dir, exist_ok=True)
        frame = sim.render()
        np.save(os.path.join(render_dir, "final_frame.npy"), frame)

    sim.save_final_state(save_path)


def _replay_metrics(path: str) -> dict:
    payload = dict(np.load(path))
    ctrl_trace = payload["ctrl_trace"]

    sim = Sim()
    sim.reset()
    m, d = sim.model, sim.data

    max_drawer = 0.0
    max_block_z = -1e9
    first_block_z_ge_050 = None
    drawer_before_retrieval_ok = False

    for i in range(ctrl_trace.shape[0]):
        d.ctrl[:] = ctrl_trace[i]
        sim.step(1)
        drawer = sim.drawer_open_amount()
        block_z = sim.block_position()[2]
        max_drawer = max(max_drawer, drawer)
        max_block_z = max(max_block_z, block_z)
        if first_block_z_ge_050 is None and block_z >= 0.50:
            first_block_z_ge_050 = i
        if first_block_z_ge_050 is None and max_drawer >= 0.05:
            drawer_before_retrieval_ok = True

    # Settle: 500 steps at last ctrl.
    contact = 0
    for _ in range(500):
        if sim.has_gripper_block_contact():
            contact += 1
        sim.step(1)
    settle_contact_fraction = contact / 500.0

    final_drawer = sim.drawer_open_amount()
    final_block_z = sim.block_position()[2]

    return {
        "ctrl_steps": int(ctrl_trace.shape[0]),
        "max_replay_drawer_open": float(max_drawer),
        "max_replay_block_z": float(max_block_z),
        "drawer_open_before_block_z_ge_0p50": bool(drawer_before_retrieval_ok),
        "final_drawer_open_after_settle": float(final_drawer),
        "final_block_z_after_settle": float(final_block_z),
        "settle_contact_fraction": float(settle_contact_fraction),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/final_state.npz")
    ap.add_argument("--render-dir", default=None)
    ap.add_argument("--metrics", action="store_true")
    args = ap.parse_args()

    if not args.metrics:
        sim = Sim()
        _run_policy(sim, save_path=args.out, render_dir=args.render_dir)
        return

    if not os.path.exists(args.out):
        raise SystemExit(f"missing {args.out}")
    metrics = _replay_metrics(args.out)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
