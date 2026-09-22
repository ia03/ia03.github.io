import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim, SUCCESS_CUP_Z


ARM_DOF = 7
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0
LEFT_PAD_GEOM = 69
RIGHT_PAD_GEOM = 77


@dataclass
class EvalResult:
    final_cup_z: float
    contact_fraction: float
    passed: bool


def clamp_to_ctrlrange(sim: Sim, q: np.ndarray) -> np.ndarray:
    q = q.copy()
    lo = sim.model.actuator_ctrlrange[:ARM_DOF, 0]
    hi = sim.model.actuator_ctrlrange[:ARM_DOF, 1]
    return np.clip(q, lo, hi)


def hand_axes(sim: Sim):
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    rot = sim.data.xmat[hand_id].reshape(3, 3)
    return rot[:, 0].copy(), rot[:, 1].copy(), rot[:, 2].copy()


def pad_positions(sim: Sim):
    return sim.data.geom_xpos[LEFT_PAD_GEOM].copy(), sim.data.geom_xpos[RIGHT_PAD_GEOM].copy()


def pinch_midpoint(sim: Sim):
    left, right = pad_positions(sim)
    return 0.5 * (left + right)


def pinch_axis(sim: Sim):
    left, right = pad_positions(sim)
    axis = left - right
    norm = np.linalg.norm(axis)
    return axis / max(norm, 1e-9)


def set_arm_ctrl(sim: Sim, q: np.ndarray, gripper: float | None = None):
    sim.data.ctrl[:ARM_DOF] = clamp_to_ctrlrange(sim, q)
    if gripper is not None:
        sim.data.ctrl[7] = np.clip(gripper, 0.0, GRIPPER_OPEN)


def ik_features(sim: Sim):
    mid = pinch_midpoint(sim)
    _, hand_y, hand_z = hand_axes(sim)
    return np.concatenate([mid, hand_y, hand_z])


def ik_target(mid_target: np.ndarray, hand_y_target: np.ndarray, hand_z_target: np.ndarray):
    return np.concatenate([mid_target, hand_y_target, hand_z_target])


def solve_ik(
    sim: Sim,
    mid_target: np.ndarray,
    hand_y_target: np.ndarray,
    hand_z_target: np.ndarray,
    q_init: np.ndarray,
    max_iters: int = 60,
):
    q = clamp_to_ctrlrange(sim, q_init)
    target = ik_target(mid_target, hand_y_target, hand_z_target)
    weights = np.diag([12, 12, 12, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5])
    eps = 2e-4
    for _ in range(max_iters):
        sim.data.qpos[:ARM_DOF] = q
        sim.data.qvel[:] = 0
        mujoco.mj_forward(sim.model, sim.data)
        feat = ik_features(sim)
        err = target - feat
        if np.linalg.norm(err[:3]) < 2e-3 and np.linalg.norm(err[3:]) < 4e-2:
            break
        J = np.zeros((9, ARM_DOF))
        for j in range(ARM_DOF):
            q2 = q.copy()
            q2[j] += eps
            sim.data.qpos[:ARM_DOF] = q2
            sim.data.qvel[:] = 0
            mujoco.mj_forward(sim.model, sim.data)
            feat2 = ik_features(sim)
            J[:, j] = (feat2 - feat) / eps
        sim.data.qpos[:ARM_DOF] = q
        sim.data.qvel[:] = 0
        mujoco.mj_forward(sim.model, sim.data)
        A = weights @ J
        b = weights @ err
        dq = np.linalg.solve(A.T @ A + 2e-3 * np.eye(ARM_DOF), A.T @ b)
        q = clamp_to_ctrlrange(sim, q + 0.7 * dq)
    return q


def move_to_pose(sim: Sim, q_goal: np.ndarray, gripper: float, steps: int):
    q_start = sim.data.ctrl[:ARM_DOF].copy()
    g_start = float(sim.data.ctrl[7])
    for i in range(1, steps + 1):
        alpha = i / steps
        q = (1 - alpha) * q_start + alpha * q_goal
        g = (1 - alpha) * g_start + alpha * gripper
        set_arm_ctrl(sim, q, g)
        sim.step(1)


def hold(sim: Sim, q_goal: np.ndarray, gripper: float, steps: int):
    set_arm_ctrl(sim, q_goal, gripper)
    sim.step(steps)


def finger_cup_contact_fraction(sim: Sim, steps: int = 500):
    finger_bodies = {
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
    }
    cup_body = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    hits = 0
    for _ in range(steps):
        touched = False
        for i in range(sim.data.ncon):
            con = sim.data.contact[i]
            b1 = sim.model.geom_bodyid[con.geom1]
            b2 = sim.model.geom_bodyid[con.geom2]
            if (b1 in finger_bodies and b2 == cup_body) or (b2 in finger_bodies and b1 == cup_body):
                touched = True
                break
        hits += int(touched)
        sim.step(1)
    return hits / steps


def evaluate_ctrl_trace(ctrl_trace: np.ndarray) -> EvalResult:
    sim = Sim()
    for ctrl in ctrl_trace:
        sim.data.ctrl[:] = ctrl
        sim.step(1)
    contact_fraction = finger_cup_contact_fraction(sim, 500)
    final_cup_z = float(sim.cup_position()[2])
    return EvalResult(
        final_cup_z=final_cup_z,
        contact_fraction=contact_fraction,
        passed=final_cup_z >= SUCCESS_CUP_Z and contact_fraction >= 0.5,
    )


def run_attempt(
    descend_mid_z: float,
    close_steps: int,
    lift_mid_z: float,
    hold_steps: int,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    overcup_mid_z: float = 0.56,
    preclose_hold: int = 30,
):
    sim = Sim()
    cup = sim.cup_position().copy()
    _, hand_y0, hand_z0 = hand_axes(sim)
    q0 = sim.data.qpos[:ARM_DOF].copy()
    set_arm_ctrl(sim, q0, GRIPPER_OPEN)
    sim.step(60)

    cup_xy = cup[:2] + np.array([x_offset, y_offset])
    mid_above = np.array([cup_xy[0], cup_xy[1], overcup_mid_z])
    mid_descend = np.array([cup_xy[0], cup_xy[1], descend_mid_z])
    mid_lift = np.array([cup_xy[0], cup_xy[1], lift_mid_z])

    q_above = solve_ik(sim, mid_above, hand_y0, hand_z0, sim.data.qpos[:ARM_DOF].copy())
    move_to_pose(sim, q_above, GRIPPER_OPEN, 220)
    hold(sim, q_above, GRIPPER_OPEN, 40)

    q_descend = solve_ik(sim, mid_descend, hand_y0, hand_z0, q_above)
    move_to_pose(sim, q_descend, GRIPPER_OPEN, 180)
    hold(sim, q_descend, GRIPPER_OPEN, preclose_hold)

    move_to_pose(sim, q_descend, GRIPPER_CLOSED, close_steps)
    hold(sim, q_descend, GRIPPER_CLOSED, 40)

    q_lift = solve_ik(sim, mid_lift, hand_y0, hand_z0, q_descend)
    move_to_pose(sim, q_lift, GRIPPER_CLOSED, 220)
    hold(sim, q_lift, GRIPPER_CLOSED, hold_steps)
    return sim


def main():
    attempts = [
        dict(descend_mid_z=0.482, close_steps=180, lift_mid_z=0.63, hold_steps=200),
        dict(descend_mid_z=0.478, close_steps=220, lift_mid_z=0.64, hold_steps=240, x_offset=0.002),
        dict(descend_mid_z=0.476, close_steps=240, lift_mid_z=0.65, hold_steps=260, y_offset=0.002),
        dict(descend_mid_z=0.480, close_steps=200, lift_mid_z=0.62, hold_steps=260, x_offset=-0.002),
    ]

    best_eval = None
    best_sim = None
    for i, params in enumerate(attempts, start=1):
        sim = run_attempt(**params)
        result = evaluate_ctrl_trace(np.array(sim._ctrl_trace))
        cup_z_now = float(sim.cup_position()[2])
        print(f"attempt {i}: now_z={cup_z_now:.4f} replay_final_z={result.final_cup_z:.4f} contact={result.contact_fraction:.3f} passed={result.passed} params={params}")
        if best_eval is None or (result.passed, result.final_cup_z + 0.1 * result.contact_fraction) > (
            best_eval.passed,
            best_eval.final_cup_z + 0.1 * best_eval.contact_fraction,
        ):
            best_eval = result
            best_sim = sim
            best_sim.save_final_state("/work/final_state.npz")
            print("saved best-so-far to /work/final_state.npz")
        if result.passed:
            break

    if best_eval is None:
        raise RuntimeError("no attempts executed")
    print(
        f"best replay: final_z={best_eval.final_cup_z:.4f} "
        f"contact={best_eval.contact_fraction:.3f} passed={best_eval.passed}"
    )


if __name__ == "__main__":
    main()
