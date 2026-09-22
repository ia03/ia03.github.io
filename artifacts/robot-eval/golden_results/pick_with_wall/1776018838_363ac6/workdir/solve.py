from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


HAND_BODY = "hand"
LEFT_FINGER_BODY = "left_finger"
RIGHT_FINGER_BODY = "right_finger"
TARGET_DOWN = np.array([0.0, 0.0, -1.0], dtype=float)
OPEN_GRIPPER = 255.0
CLOSE_GRIPPER = 40.0
CLOSED_GRIPPER = 0.0
MIN_TRACE_SAMPLES = 20
SUCCESS_CUP_Z = 0.56
SUCCESS_CUP_X_MAX = 0.42
RETURN_X_CHECKPOINT = 0.50
IDEAL_REPLAY_STEPS = 1200


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


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def weighted_subscore(*components):
    total_weight = sum(weight for _, _, weight in components)
    if total_weight <= 0:
        return 0.0
    return float(sum(clamp01(value) * weight for _, value, weight in components) / total_weight)


def orient_err(r_cur: np.ndarray, r_tgt: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(r_cur[:, 0], r_tgt[:, 0])
        + np.cross(r_cur[:, 1], r_tgt[:, 1])
        + np.cross(r_cur[:, 2], r_tgt[:, 2])
    )


def solve_midpoint_ik(model, hand_id, left_id, right_id, target_midpoint, seed):
    q = seed.copy()
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]
    scratch = mujoco.MjData(model)

    for _ in range(140):
        scratch.qpos[:7] = q
        scratch.qpos[7:9] = 0.04
        mujoco.mj_forward(model, scratch)

        hand_z = scratch.xmat[hand_id].reshape(3, 3)[:, 2]
        midpoint = 0.5 * (scratch.xpos[left_id] + scratch.xpos[right_id])
        pos_err = target_midpoint - midpoint
        ori_err = np.cross(hand_z, TARGET_DOWN)

        jacp_l = np.zeros((3, model.nv))
        jacr_l = np.zeros((3, model.nv))
        jacp_r = np.zeros((3, model.nv))
        jacr_r = np.zeros((3, model.nv))
        jacp_h = np.zeros((3, model.nv))
        jacr_h = np.zeros((3, model.nv))

        mujoco.mj_jacBodyCom(model, scratch, jacp_l, jacr_l, left_id)
        mujoco.mj_jacBodyCom(model, scratch, jacp_r, jacr_r, right_id)
        mujoco.mj_jacBody(model, scratch, jacp_h, jacr_h, hand_id)

        jac = np.vstack([0.5 * (jacp_l[:, :7] + jacp_r[:, :7]), 0.35 * jacr_h[:, :7]])
        err = np.concatenate([pos_err, 0.35 * ori_err])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), err)
        q = np.clip(q + 0.8 * dq, qmin, qmax)

    return q


def best_q(model, data, hand_id, left_id, right_id, target_midpoint, seeds):
    candidates = []
    for seed in seeds:
        q = solve_midpoint_ik(model, hand_id, left_id, right_id, target_midpoint, seed)
        scratch = mujoco.MjData(model)
        scratch.qpos[:7] = q
        scratch.qpos[7:9] = 0.04
        mujoco.mj_forward(model, scratch)
        midpoint = 0.5 * (scratch.xpos[left_id] + scratch.xpos[right_id])
        hand_z = scratch.xmat[hand_id].reshape(3, 3)[:, 2]
        err = np.linalg.norm(midpoint - target_midpoint) + 0.2 * np.linalg.norm(np.cross(hand_z, TARGET_DOWN))
        candidates.append((err, q))
    return min(candidates, key=lambda item: item[0])[1]


def play_segment(sim, q0, q1, g0, g1, steps):
    for i in range(steps):
        u = (i + 1) / steps
        sim.data.ctrl[:7] = (1.0 - u) * q0 + u * q1
        sim.data.ctrl[7] = (1.0 - u) * g0 + u * g1
        sim.step(1)


def evaluate(params: dict) -> TrialResult:
    sim = Sim()
    model = sim.model
    data = sim.data

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_FINGER_BODY)
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_FINGER_BODY)

    seeds = [
        np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0], dtype=float),
        np.array([0.08, -0.19, -0.07, -1.88, -0.01, 1.69, 0.5], dtype=float),
        np.array([-0.08, -0.19, 0.07, -1.88, 0.01, 1.69, -0.5], dtype=float),
        np.array([0.26, 0.60, -0.25, -1.70, 0.07, 3.75, 0.87], dtype=float),
        np.array([0.13, 1.20, -0.35, -0.63, -0.02, 3.35, 1.14], dtype=float),
    ]

    sim.data.ctrl[7] = OPEN_GRIPPER
    sim.step(40)

    cup0 = sim.cup_position().copy()
    side_y = params["side_y"]

    q_start = np.zeros(7, dtype=float)
    q_above = best_q(model, data, hand_id, left_id, right_id, np.array([0.36, side_y, params["above_z"]], dtype=float), seeds)
    grasp_target = cup0 + np.array([params["grasp_dx"], params["grasp_dy"], params["grasp_dz"]], dtype=float)
    q_grasp = best_q(model, data, hand_id, left_id, right_id, grasp_target, seeds + [q_above])

    play_segment(sim, q_start, q_above, OPEN_GRIPPER, OPEN_GRIPPER, 220)
    play_segment(sim, q_above, q_grasp, OPEN_GRIPPER, OPEN_GRIPPER, 120)
    play_segment(sim, q_grasp, q_grasp, OPEN_GRIPPER, CLOSE_GRIPPER, 160)
    play_segment(sim, q_grasp, q_grasp, CLOSE_GRIPPER, CLOSED_GRIPPER, 80)

    cup_after_grasp = sim.cup_position().copy()
    q_lift = best_q(
        model,
        data,
        hand_id,
        left_id,
        right_id,
        np.array([cup_after_grasp[0], cup_after_grasp[1], params["lift_z"]], dtype=float),
        seeds + [q_grasp],
    )
    q_cross = best_q(
        model,
        data,
        hand_id,
        left_id,
        right_id,
        np.array([params["cross_x"], cup_after_grasp[1], params["lift_z"]], dtype=float),
        seeds + [q_lift],
    )
    q_final = best_q(
        model,
        data,
        hand_id,
        left_id,
        right_id,
        np.array([params["final_x"], cup_after_grasp[1], params["lift_z"]], dtype=float),
        seeds + [q_cross],
    )

    play_segment(sim, q_grasp, q_lift, CLOSED_GRIPPER, CLOSED_GRIPPER, 180)
    play_segment(sim, q_lift, q_lift, CLOSED_GRIPPER, CLOSED_GRIPPER, 100)
    play_segment(sim, q_lift, q_cross, CLOSED_GRIPPER, CLOSED_GRIPPER, 220)
    play_segment(sim, q_cross, q_final, CLOSED_GRIPPER, CLOSED_GRIPPER, 220)
    play_segment(sim, q_final, q_final, CLOSED_GRIPPER, CLOSED_GRIPPER, 260)

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
    return_progress = float(np.clip((0.57 - best_x) / (0.57 - RETURN_X_CHECKPOINT), 0.0, 1.0))
    lift_progress = float(np.clip((best_z - 0.435) / (SUCCESS_CUP_Z - 0.435), 0.0, 1.0))
    settle_progress = float(np.clip(settle_contact / 0.5, 0.0, 1.0))
    final_return_progress = float(np.clip((0.57 - float(final[0])) / (0.57 - SUCCESS_CUP_X_MAX), 0.0, 1.0))
    final_height_progress = float(np.clip((float(final[2]) - 0.435) / (SUCCESS_CUP_Z - 0.435), 0.0, 1.0))
    replay_steps = int(cup_trace.shape[0] * 10)
    efficiency_progress = float(np.clip(IDEAL_REPLAY_STEPS / max(replay_steps, 1), 0.0, 1.0))

    score_components = [
        {"name": "grasp_progress", "value": grasp_progress, "weight": 0.05},
        {"name": "return_progress", "value": return_progress, "weight": 0.15},
        {"name": "lift_progress", "value": lift_progress, "weight": 0.10},
        {"name": "final_return_progress", "value": final_return_progress, "weight": 0.10},
        {"name": "final_height_progress", "value": final_height_progress, "weight": 0.25},
        {"name": "settle_contact_progress", "value": settle_progress, "weight": 0.25},
        {"name": "efficiency_progress", "value": efficiency_progress, "weight": 0.10},
    ]
    score = weighted_subscore(*[(item["name"], item["value"], item["weight"]) for item in score_components])
    passed_like = (
        len(sim._ctrl_trace) >= MIN_TRACE_SAMPLES
        and ever_contact
        and best_x <= RETURN_X_CHECKPOINT
        and best_z >= SUCCESS_CUP_Z
        and float(final[0]) <= SUCCESS_CUP_X_MAX
        and float(final[2]) >= SUCCESS_CUP_Z
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
    for grasp_dx in [0.0, 0.005]:
        for grasp_dy in [-0.01, 0.0, 0.01]:
            for grasp_dz in [-0.04, -0.03]:
                for lift_z in [0.62, 0.66]:
                    for cross_x in [0.42, 0.40]:
                        yield {
                            "side_y": 0.24,
                            "above_z": 0.54,
                            "grasp_dx": grasp_dx,
                            "grasp_dy": grasp_dy,
                            "grasp_dz": grasp_dz,
                            "lift_z": lift_z,
                            "cross_x": cross_x,
                            "final_x": 0.38,
                        }


def main() -> None:
    best = None
    for i, params in enumerate(param_grid(), start=1):
        result = evaluate(params)
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
