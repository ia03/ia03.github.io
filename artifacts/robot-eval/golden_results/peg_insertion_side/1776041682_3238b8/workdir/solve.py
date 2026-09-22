import numpy as np
import mujoco

from sim import Sim


HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
SAFE = np.array([0.0, -0.35, 0.0, -1.90, 0.0, 1.95, 0.785], dtype=float)
SEEDS = [
    HOME.copy(),
    SAFE.copy(),
    np.array([0.08, -0.19, -0.07, -1.88, -0.01, 1.69, 0.50], dtype=float),
    np.array([-0.08, -0.19, 0.07, -1.88, 0.01, 1.69, -0.50], dtype=float),
]

PEG_TARGET_Y = -0.10
PEG_TARGET_Z = 0.52
PEG_SUCCESS_X = 0.565
FORWARD_SETTLE_STEPS = 500

DESIRED_X = np.array([1.0, 0.0, 0.0], dtype=float)
DESIRED_Z = np.array([0.0, 0.0, -1.0], dtype=float)


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def quat_to_x_axis(quat):
    quat = quat / np.linalg.norm(quat)
    return np.array(
        [
            1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
            2 * (quat[1] * quat[2] + quat[0] * quat[3]),
            2 * (quat[1] * quat[3] - quat[0] * quat[2]),
        ]
    )


def best_q(model, data, hand_id, left_id, right_id, target_pos, extra_seed=None):
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]
    candidates = []
    seeds = list(SEEDS)
    if extra_seed is not None:
        seeds.append(extra_seed.copy())

    for seed in seeds:
        q = seed.copy()
        for _ in range(240):
            data.qpos[:7] = q
            data.qpos[7:9] = 0.04
            mujoco.mj_forward(model, data)

            hand_R = data.xmat[hand_id].reshape(3, 3)
            midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
            pos_err = np.array(target_pos, dtype=float) - midpoint
            ori_err = np.cross(hand_R[:, 0], DESIRED_X) + np.cross(hand_R[:, 2], DESIRED_Z)

            jacp_l = np.zeros((3, model.nv))
            jacr_l = np.zeros((3, model.nv))
            jacp_r = np.zeros((3, model.nv))
            jacr_r = np.zeros((3, model.nv))
            jacp_h = np.zeros((3, model.nv))
            jacr_h = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp_l, jacr_l, left_id)
            mujoco.mj_jacBody(model, data, jacp_r, jacr_r, right_id)
            mujoco.mj_jacBody(model, data, jacp_h, jacr_h, hand_id)

            jac = np.vstack([0.5 * (jacp_l[:, :7] + jacp_r[:, :7]), 0.2 * jacr_h[:, :7]])
            err = np.concatenate([pos_err, 0.2 * ori_err])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), err)
            q = np.clip(q + 0.7 * dq, qmin, qmax)

        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)
        hand_R = data.xmat[hand_id].reshape(3, 3)
        midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        err = np.linalg.norm(midpoint - np.array(target_pos, dtype=float))
        err += 0.2 * np.linalg.norm(np.cross(hand_R[:, 0], DESIRED_X))
        err += 0.2 * np.linalg.norm(np.cross(hand_R[:, 2], DESIRED_Z))
        candidates.append((err, q.copy()))

    return min(candidates, key=lambda item: item[0])[1]


def move_segment(sim, q_target, grip_target, steps):
    q_start = sim.data.ctrl[:7].copy()
    g_start = float(sim.data.ctrl[7])
    for i in range(steps):
        a = (i + 1) / steps
        sim.data.ctrl[:7] = (1.0 - a) * q_start + a * q_target
        sim.data.ctrl[7] = (1.0 - a) * g_start + a * grip_target
        sim.step(1)


def hold_segment(sim, q_target, grip_target, steps):
    for _ in range(steps):
        sim.data.ctrl[:7] = q_target
        sim.data.ctrl[7] = grip_target
        sim.step(1)


def evaluate_trial(sim):
    model = sim.model
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for ctrl in sim._ctrl_trace:
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
    peg_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    for _ in range(FORWARD_SETTLE_STEPS):
        mujoco.mj_step(model, data)

    peg_pos = data.xpos[peg_bid].copy()
    peg_x_axis = quat_to_x_axis(data.qpos[10:14])
    x = float(peg_pos[0])
    y_error = float(abs(peg_pos[1] - PEG_TARGET_Y))
    z_error = float(abs(peg_pos[2] - PEG_TARGET_Z))
    alignment = float(abs(peg_x_axis[0]))
    success = x >= PEG_SUCCESS_X and y_error <= 0.025 and z_error <= 0.020 and alignment >= 0.45
    score = (
        0.45 * clamp01((x - 0.505) / (PEG_SUCCESS_X - 0.505))
        + 0.15 * clamp01(1.0 - y_error / 0.025)
        + 0.15 * clamp01(1.0 - z_error / 0.020)
        + 0.20 * clamp01(alignment / 0.45)
        + 0.05 * clamp01(4500 / max(len(sim._ctrl_trace), 1))
    )
    metrics = {
        "peg_xyz_after": [float(v) for v in peg_pos],
        "peg_x_axis": [float(v) for v in peg_x_axis],
        "x_axis_alignment": alignment,
        "y_error": y_error,
        "z_error": z_error,
    }
    return success, score, metrics


def run_trial(grasp_z, grip_close, lift_x, lift_z, push_y, push_z, final_x, close_hold, final_hold):
    sim = Sim()
    model = sim.model
    data = sim.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    peg0 = sim.peg_position().copy()

    data.ctrl[:7] = HOME
    data.ctrl[7] = 255.0
    sim.step(300)

    q_above = best_q(model, data, hand_id, left_id, right_id, [peg0[0], peg0[1], peg0[2] + 0.10])
    q_grasp = best_q(model, data, hand_id, left_id, right_id, [peg0[0], peg0[1], grasp_z], extra_seed=q_above)
    q_lift = best_q(model, data, hand_id, left_id, right_id, [lift_x, peg0[1], lift_z], extra_seed=q_grasp)
    q_preinsert = best_q(model, data, hand_id, left_id, right_id, [0.555, push_y, push_z], extra_seed=q_lift)
    q_insert = best_q(model, data, hand_id, left_id, right_id, [final_x, push_y, push_z], extra_seed=q_preinsert)

    move_segment(sim, q_above, 255.0, 320)
    hold_segment(sim, q_above, 255.0, 220)
    move_segment(sim, q_grasp, 255.0, 420)
    hold_segment(sim, q_grasp, 255.0, 420)
    move_segment(sim, q_grasp, grip_close, 260)
    hold_segment(sim, q_grasp, grip_close, close_hold)
    move_segment(sim, q_lift, grip_close, 340)
    hold_segment(sim, q_lift, grip_close, 220)
    move_segment(sim, q_preinsert, grip_close, 260)
    move_segment(sim, q_insert, grip_close, 360)
    hold_segment(sim, q_insert, grip_close, final_hold)

    return sim, evaluate_trial(sim)


def main():
    best_sim = None
    best_score = -1.0
    best_metrics = None
    best_params = None
    best_passed = False

    search = [
        (0.522, 10.0, 0.540, 0.555, -0.102, 0.525, 0.585, 260, 900),
        (0.522, 0.0, 0.540, 0.555, -0.102, 0.525, 0.585, 260, 900),
        (0.520, 10.0, 0.545, 0.555, -0.102, 0.525, 0.590, 320, 1000),
        (0.520, 0.0, 0.545, 0.555, -0.102, 0.525, 0.590, 320, 1000),
        (0.518, 10.0, 0.545, 0.548, -0.100, 0.522, 0.590, 320, 1000),
        (0.518, 0.0, 0.545, 0.548, -0.100, 0.522, 0.590, 320, 1000),
    ]

    for params in search:
        sim, (passed, score, metrics) = run_trial(*params)
        print(f"trial params={params} passed={passed} score={score:.4f} metrics={metrics}", flush=True)
        if passed or score > best_score:
            best_sim = sim
            best_score = score
            best_metrics = metrics
            best_params = params
            best_passed = passed
            best_sim.save_final_state("/work/final_state.npz")
            print(f"saved best params={best_params} passed={best_passed} metrics={best_metrics}", flush=True)
        if passed:
            break

    print(f"best params={best_params} passed={best_passed} score={best_score:.4f} metrics={best_metrics}", flush=True)


if __name__ == "__main__":
    main()
