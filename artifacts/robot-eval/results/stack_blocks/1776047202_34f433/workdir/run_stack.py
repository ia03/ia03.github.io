#!/usr/bin/env python3
import argparse
import numpy as np
import mujoco

from sim import Sim


def _clip_to_actuator_ranges(sim: Sim, q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=float).copy()
    for i in range(min(7, sim.model.nu)):
        if sim.model.actuator_ctrllimited[i]:
            lo, hi = sim.model.actuator_ctrlrange[i]
            q[i] = float(np.clip(q[i], lo, hi))
    return q


def _hand_id(sim: Sim) -> int:
    return mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")


def _finger_ids(sim: Sim) -> tuple[int, int]:
    lf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    rf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    return lf, rf


def _finger_avg_pos(sim: Sim) -> np.ndarray:
    lf, rf = _finger_ids(sim)
    return 0.5 * (_body_pos(sim, lf) + _body_pos(sim, rf))


def _body_pos(sim: Sim, body_id: int) -> np.ndarray:
    return np.array(sim.data.xpos[body_id], dtype=float)


def _jacp_for_body(sim: Sim, body_id: int) -> np.ndarray:
    jacp = np.zeros((3, sim.model.nv), dtype=float)
    jacr = np.zeros((3, sim.model.nv), dtype=float)
    mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, body_id)
    return jacp


def _jac_for_body(sim: Sim, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    jacp = np.zeros((3, sim.model.nv), dtype=float)
    jacr = np.zeros((3, sim.model.nv), dtype=float)
    mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, body_id)
    return jacp, jacr


def _move_hand(
    sim: Sim,
    target_pos: np.ndarray,
    *,
    max_iters: int,
    target_z: np.ndarray | None = None,
    rot_weight: float = 0.35,
    tol: float = 0.004,
    damping: float = 1e-3,
    step_scale: float = 0.55,
    inner_steps: int = 5,
):
    hid = _hand_id(sim)
    dofs = np.arange(7, dtype=int)
    for _ in range(max_iters):
        cur = _body_pos(sim, hid)
        err = target_pos - cur
        rot_err = np.zeros(3, dtype=float)
        if target_z is not None:
            z_cur = sim.data.xmat[hid].reshape(3, 3)[:, 2].copy()
            z_des = np.array(target_z, dtype=float)
            z_des /= float(np.linalg.norm(z_des) + 1e-12)
            rot_err = np.cross(z_cur, z_des)
        err6 = np.concatenate([err, rot_weight * rot_err])
        if float(np.linalg.norm(err)) < tol and float(np.linalg.norm(rot_err)) < 0.12:
            break
        jacp, jacr = _jac_for_body(sim, hid)
        J = np.vstack([jacp[:, dofs], rot_weight * jacr[:, dofs]])
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping * np.eye(6)), err6)
        dq = np.clip(dq, -0.12, 0.12) * step_scale
        q_cur = sim.data.qpos[:7].copy()
        q_des = _clip_to_actuator_ranges(sim, q_cur + dq)
        sim.data.ctrl[:7] = q_des
        sim.step(inner_steps)


def _hold(sim: Sim, steps: int):
    sim.step(int(steps))


def _open_gripper(sim: Sim, value: float, steps: int = 200):
    sim.data.ctrl[7] = float(np.clip(value, 0.0, 255.0))
    _hold(sim, steps)


def run_policy(sim: Sim, *, settle_steps: int = 1500, render_debug: bool = False):
    # Stabilize at a legal posture (joint4 limit makes ctrl=0 invalid).
    q0 = _clip_to_actuator_ranges(sim, sim.data.qpos[:7])
    sim.data.ctrl[:7] = q0
    _open_gripper(sim, 255.0, steps=300)
    # Prefer a top-down approach: keep hand z-axis pointing downward.
    hid = _hand_id(sim)
    z_down = np.array([0.0, 0.0, -1.0], dtype=float)

    red0 = sim.block_positions()["red"]
    green0 = sim.block_positions()["green"]

    above_red = red0 + np.array([0.0, 0.0, 0.26])
    lift = red0 + np.array([0.0, 0.0, 0.30])
    above_green = green0 + np.array([0.0, 0.0, 0.25])

    _move_hand(sim, above_red, max_iters=320, target_z=z_down)

    # Align grasp using current finger-average offset (more stable than using hand body origin).
    favg = _finger_avg_pos(sim)
    hand = _body_pos(sim, hid)
    hand_minus_favg = hand - favg
    desired_favg = red0 + np.array([0.0, 0.0, 0.010])
    grasp_hand = desired_favg + hand_minus_favg

    _move_hand(sim, grasp_hand + np.array([0.0, 0.0, 0.05]), max_iters=260, tol=0.004, inner_steps=4, target_z=z_down)
    _move_hand(sim, grasp_hand, max_iters=420, tol=0.003, inner_steps=3, step_scale=0.45, target_z=z_down)
    _open_gripper(sim, 0.0, steps=500)  # close firmly
    _move_hand(sim, grasp_hand + np.array([0.0, 0.0, -0.015]), max_iters=260, tol=0.003, inner_steps=3, step_scale=0.35, target_z=z_down)
    _move_hand(sim, lift, max_iters=520, target_z=z_down)

    # Simple retry if grasp failed (red block didn't lift).
    red_after = sim.block_positions()["red"]
    if float(red_after[2] - red0[2]) < 0.035:
        _open_gripper(sim, 255.0, steps=250)
        _move_hand(sim, above_red + np.array([0.04, 0.0, 0.0]), max_iters=300, tol=0.005, target_z=z_down)
        favg = _finger_avg_pos(sim)
        hand = _body_pos(sim, hid)
        hand_minus_favg = hand - favg
        desired_favg = red0 + np.array([0.020, 0.0, 0.008])
        grasp_hand = desired_favg + hand_minus_favg
        _move_hand(sim, grasp_hand + np.array([0.0, 0.0, 0.05]), max_iters=260, tol=0.004, inner_steps=4, target_z=z_down)
        _move_hand(sim, grasp_hand, max_iters=460, tol=0.003, inner_steps=3, step_scale=0.45, target_z=z_down)
        _open_gripper(sim, 0.0, steps=550)
        _move_hand(sim, grasp_hand + np.array([0.0, 0.0, -0.015]), max_iters=260, tol=0.003, inner_steps=3, step_scale=0.35, target_z=z_down)
        _move_hand(sim, lift, max_iters=560, target_z=z_down)

    # Estimate grasp offset for more accurate placement.
    hand_now = _body_pos(sim, hid)
    red_now = sim.block_positions()["red"]
    hand_minus_red = hand_now - red_now

    desired_red = green0 + np.array([0.0, 0.0, 0.055])
    place_hand = desired_red + hand_minus_red

    _move_hand(sim, above_green, max_iters=720, target_z=z_down)
    # Lower slowly near contact.
    _move_hand(sim, place_hand + np.array([0.0, 0.0, 0.06]), max_iters=580, tol=0.004, inner_steps=4, target_z=z_down)
    _move_hand(sim, place_hand, max_iters=800, tol=0.003, inner_steps=3, step_scale=0.45, target_z=z_down)

    # Gentle release.
    for u, st in [(40.0, 80), (90.0, 120), (255.0, 200)]:
        _open_gripper(sim, u, steps=st)

    # Retreat up and back a bit, then let things settle.
    retreat = _body_pos(sim, hid) + np.array([-0.10, 0.0, 0.18])
    _move_hand(sim, retreat, max_iters=820, tol=0.005, target_z=z_down)
    _hold(sim, settle_steps)

    if render_debug:
        frame = sim.render()
        return frame
    return None


def eval_stack(sim: Sim, settle: int = 500):
    # Measure after additional settle, mimicking grader.
    qpos = sim.data.qpos.copy()
    qvel = sim.data.qvel.copy()
    ctrl = sim.data.ctrl.copy()
    trace_len = len(sim._ctrl_trace)

    for _ in range(int(settle)):
        sim.step(1)
    pos = sim.block_positions()
    d = pos["red"] - pos["green"]
    dx, dy, dz = map(float, d)

    # Restore.
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel
    sim.data.ctrl[:] = ctrl
    mujoco.mj_forward(sim.model, sim.data)
    sim._ctrl_trace = sim._ctrl_trace[:trace_len]
    return dx, dy, dz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/final_state.npz")
    ap.add_argument("--settle-steps", type=int, default=1500)
    ap.add_argument("--debug-render", action="store_true")
    args = ap.parse_args()

    sim = Sim()
    run_policy(sim, settle_steps=args.settle_steps, render_debug=args.debug_render)

    dx, dy, dz = eval_stack(sim, settle=500)
    print(f"dx={dx:+.4f} dy={dy:+.4f} dz={dz:+.4f} steps={len(sim._ctrl_trace)} t={sim.data.time:.3f}")

    sim.save_final_state(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
