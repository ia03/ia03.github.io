import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


HAND_BODY = "hand"
OPEN_GRIPPER = 255.0
CLOSE_GRIPPER = 40.0
CLOSED_GRIPPER = 0.0
HAND_TO_PAD_MID_LOCAL = np.array([0.0, 0.0, 0.1029])
ARM_CTRL_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
ARM_CTRL_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


@dataclass
class TrialResult:
    score: float
    passed_like: bool
    ever_contact: bool
    best_x: float
    best_z: float
    final_x: float
    final_z: float
    settle_contact: float
    params: dict
    sim: Sim


def rot_down(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array(
        [
            [c, s, 0.0],
            [s, -c, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )


def orient_err(r_cur: np.ndarray, r_tgt: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(r_cur[:, 0], r_tgt[:, 0])
        + np.cross(r_cur[:, 1], r_tgt[:, 1])
        + np.cross(r_cur[:, 2], r_tgt[:, 2])
    )


def set_arm_ctrl(sim: Sim, q: np.ndarray, grip: float) -> None:
    sim.data.ctrl[:7] = np.clip(q, ARM_CTRL_MIN, ARM_CTRL_MAX)
    sim.data.ctrl[7] = grip


def move_hand(sim: Sim, target_pos: np.ndarray, target_rot: np.ndarray, grip: float, steps: int = 140) -> float:
    body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    q_nom = np.array([0.0, 0.4, 0.0, -1.8, 0.0, 2.2, 0.8])
    for _ in range(steps):
        pos = sim.data.xpos[body_id].copy()
        rot = sim.data.xmat[body_id].reshape(3, 3).copy()
        pos_err = target_pos - pos
        rot_err = orient_err(rot, target_rot)
        err = np.concatenate([8.0 * pos_err, 1.8 * rot_err])
        if np.linalg.norm(pos_err) < 0.008 and np.linalg.norm(rot_err) < 0.06:
            break
        jacp = np.zeros((3, sim.model.nv))
        jacr = np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, body_id)
        j = np.vstack([jacp[:, :7], jacr[:, :7]])
        jj = j @ j.T + 1e-4 * np.eye(6)
        dq = j.T @ np.linalg.solve(jj, err)
        dq += 0.03 * (q_nom - sim.data.qpos[:7])
        q_target = sim.data.qpos[:7] + np.clip(dq, -0.08, 0.08)
        set_arm_ctrl(sim, q_target, grip)
        sim.step(6)
    final_pos = sim.data.xpos[body_id].copy()
    return float(np.linalg.norm(target_pos - final_pos))


def hold(sim: Sim, n: int, grip: float) -> None:
    set_arm_ctrl(sim, sim.data.qpos[:7].copy(), grip)
    sim.step(n)


def trial(params: dict) -> TrialResult:
    sim = Sim()
    sim.data.ctrl[7] = OPEN_GRIPPER
    sim.step(40)

    yaw = params["yaw"]
    r = rot_down(yaw)
    cup0 = sim.cup_position().copy()
    side_y = params["side_y"]
    pre_height = params["pre_height"]

    waypoints = [
        np.array([0.36, side_y, pre_height]),
        np.array([0.50, side_y, pre_height + 0.02]),
        np.array([cup0[0] + 0.02, side_y, pre_height + 0.01]),
    ]
    for wp in waypoints:
        move_hand(sim, wp, r, OPEN_GRIPPER, steps=120)

    grasp_cup = cup0 + np.array([params["grasp_dx"], params["grasp_dy"], params["grasp_dz"]])
    grasp_hand = grasp_cup - r @ HAND_TO_PAD_MID_LOCAL
    move_hand(sim, grasp_hand + np.array([0.0, 0.0, 0.09]), r, OPEN_GRIPPER, steps=120)
    move_hand(sim, grasp_hand + np.array([0.0, 0.0, 0.03]), r, OPEN_GRIPPER, steps=120)
    move_hand(sim, grasp_hand, r, OPEN_GRIPPER, steps=180)

    set_arm_ctrl(sim, sim.data.qpos[:7].copy(), CLOSE_GRIPPER)
    sim.step(180)
    hold(sim, 100, CLOSED_GRIPPER)

    cup_after_grasp = sim.cup_position().copy()
    lift = [
        np.array([cup_after_grasp[0], cup_after_grasp[1], params["lift_height"]]),
        np.array([params["cross_x"], cup_after_grasp[1], params["lift_height"]]),
        np.array([params["final_x"], cup_after_grasp[1], params["lift_height"]]),
    ]
    for wp in lift:
        move_hand(sim, wp, r, CLOSED_GRIPPER, steps=160)
        hold(sim, 80, CLOSED_GRIPPER)

    hold(sim, 520, CLOSED_GRIPPER)

    contact_trace = np.array([entry["cup_contact"] for entry in sim._trace], dtype=float)
    cup_trace = np.array([entry["cup_pos"] for entry in sim._trace], dtype=float)
    ever_contact = bool(contact_trace.max() > 0.5)
    best_x = float(cup_trace[:, 0].min())
    best_z = float(cup_trace[:, 2].max())
    final = sim.cup_position().copy()

    settle_contacts = []
    sim2 = Sim()
    sim2.data.qpos[:] = sim.data.qpos
    sim2.data.qvel[:] = sim.data.qvel
    sim2.data.ctrl[:] = sim.data.ctrl
    mujoco.mj_forward(sim2.model, sim2.data)
    for _ in range(500):
        settle_contacts.append(float(sim2.has_gripper_cup_contact()))
        sim2.step(1)
    settle_contact = float(np.mean(settle_contacts))

    grasp_progress = 1.0 if ever_contact else 0.0
    return_progress = float(np.clip((0.57 - best_x) / (0.57 - 0.50), 0.0, 1.0))
    lift_progress = float(np.clip((best_z - 0.435) / (0.56 - 0.435), 0.0, 1.0))
    settle_progress = float(np.clip(settle_contact / 0.5, 0.0, 1.0))
    final_return_progress = float(np.clip((0.57 - float(final[0])) / (0.57 - 0.42), 0.0, 1.0))
    final_height_progress = float(np.clip((float(final[2]) - 0.435) / (0.56 - 0.435), 0.0, 1.0))
    replay_steps = int(cup_trace.shape[0] * 10)
    efficiency_progress = float(np.clip(1200 / max(replay_steps, 1), 0.0, 1.0))

    score_components = [
        {"name": "grasp_progress", "value": grasp_progress, "weight": 0.05},
        {"name": "return_progress", "value": return_progress, "weight": 0.15},
        {"name": "lift_progress", "value": lift_progress, "weight": 0.10},
        {"name": "final_return_progress", "value": final_return_progress, "weight": 0.10},
        {"name": "final_height_progress", "value": final_height_progress, "weight": 0.25},
        {"name": "settle_contact_progress", "value": settle_progress, "weight": 0.25},
        {"name": "efficiency_progress", "value": efficiency_progress, "weight": 0.10},
    ]
    score = sum(item["value"] * item["weight"] for item in score_components)
    passed_like = (
        len(sim._ctrl_trace) >= 20
        and ever_contact
        and best_x <= 0.50
        and best_z >= 0.56
        and float(final[0]) <= 0.42
        and float(final[2]) >= 0.56
        and settle_contact >= 0.5
    )
    return TrialResult(
        score=score,
        passed_like=passed_like,
        ever_contact=ever_contact,
        best_x=best_x,
        best_z=best_z,
        final_x=float(final[0]),
        final_z=float(final[2]),
        settle_contact=settle_contact,
        params=params,
        sim=sim,
    )


def param_grid():
    for yaw in [2.62, -2.62, 0.0]:
        for grasp_dx in [0.0, 0.005]:
            for grasp_dy in [-0.01, 0.0, 0.01]:
                for grasp_dz in [-0.04, -0.03]:
                    for lift_height in [0.62, 0.66]:
                        for cross_x in [0.42, 0.40]:
                            yield {
                                "yaw": yaw,
                                "side_y": 0.24,
                                "pre_height": 0.80,
                                "grasp_dx": grasp_dx,
                                "grasp_dy": grasp_dy,
                                "grasp_dz": grasp_dz,
                                "lift_height": lift_height,
                                "cross_x": cross_x,
                                "final_x": 0.38,
                            }


def main() -> None:
    best = None
    for i, params in enumerate(param_grid(), start=1):
        result = trial(params)
        print(
            f"trial {i}: score={result.score:.3f} pass_like={result.passed_like} "
            f"contact={result.ever_contact} best_x={result.best_x:.3f} best_z={result.best_z:.3f} "
            f"final=({result.final_x:.3f}, {result.final_z:.3f}) settle={result.settle_contact:.3f}",
            flush=True,
        )
        if best is None or result.score > best.score or result.passed_like:
            best = result
            best.sim.save_final_state("/work/final_state.npz")
            print("saved best-so-far to /work/final_state.npz", flush=True)
        if result.passed_like:
            break
    print("best params", best.params, flush=True)


if __name__ == "__main__":
    main()
