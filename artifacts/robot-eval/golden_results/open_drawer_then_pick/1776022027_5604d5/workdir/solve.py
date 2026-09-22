import numpy as np
import mujoco

from sim import Sim


HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
SAFE = np.array([0.0, -0.35, 0.0, -1.90, 0.0, 1.95, 0.785], dtype=float)
Q_HANDLE = np.array(
    [0.260114734942896, 0.5978935416914477, -0.2548739609310991, -1.7011568239308756, 0.0731417910040021, 3.7525, 0.8723261334629767],
    dtype=float,
)
Q_PULL = np.array(
    [0.12598874856351472, 1.1973462499176968, -0.34850701301211456, -0.625005200213342, -0.022487162744560703, 3.3513450345879296, 1.1390361931507589],
    dtype=float,
)
DOWN = np.array([0.0, 0.0, -1.0], dtype=float)
TILT_Y_PLUS = np.array([0.0, 0.10, -0.995], dtype=float)
TILT_Y_MINUS = np.array([0.0, -0.10, -0.995], dtype=float)
TILT_X_PLUS = np.array([0.10, 0.0, -0.995], dtype=float)
TILT_X_MINUS = np.array([-0.10, 0.0, -0.995], dtype=float)
DRAWER_OPEN_CTRL = 1.0


def solve_midpoint_ik(model, data, hand_id, left_id, right_id, target_midpoint, seed, target_down):
    q = seed.copy()
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]

    target_down = np.array(target_down, dtype=float)
    target_down = target_down / max(np.linalg.norm(target_down), 1e-9)

    for _ in range(160):
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)

        hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
        midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        pos_err = target_midpoint - midpoint
        ori_err = np.cross(hand_z, target_down)

        jacp_l = np.zeros((3, model.nv))
        jacr_l = np.zeros((3, model.nv))
        jacp_r = np.zeros((3, model.nv))
        jacr_r = np.zeros((3, model.nv))
        jacp_h = np.zeros((3, model.nv))
        jacr_h = np.zeros((3, model.nv))

        mujoco.mj_jacBodyCom(model, data, jacp_l, jacr_l, left_id)
        mujoco.mj_jacBodyCom(model, data, jacp_r, jacr_r, right_id)
        mujoco.mj_jacBody(model, data, jacp_h, jacr_h, hand_id)

        jac = np.vstack([0.5 * (jacp_l[:, :7] + jacp_r[:, :7]), 0.35 * jacr_h[:, :7]])
        err = np.concatenate([pos_err, 0.35 * ori_err])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), err)
        q = np.clip(q + 0.8 * dq, qmin, qmax)

    return q


def move_q(sim, q_target, grip, steps, drawer_ctrl=DRAWER_OPEN_CTRL):
    q_start = sim.data.qpos[:7].copy()
    g_start = float(sim.data.ctrl[7])
    d_start = float(sim.data.ctrl[8])
    for i in range(steps):
        a = (i + 1) / steps
        sim.data.ctrl[:7] = (1 - a) * q_start + a * q_target
        sim.data.ctrl[7] = (1 - a) * g_start + a * grip
        sim.data.ctrl[8] = (1 - a) * d_start + a * drawer_ctrl
        sim.step(1)


def hold(sim, q_target, grip, steps, drawer_ctrl=DRAWER_OPEN_CTRL):
    for _ in range(steps):
        sim.data.ctrl[:7] = q_target
        sim.data.ctrl[7] = grip
        sim.data.ctrl[8] = drawer_ctrl
        sim.step(1)


def best_q(model, data, hand_id, left_id, right_id, target_xyz, target_down, seeds, extra_seed=None):
    target = np.array(target_xyz, dtype=float)
    candidates = []
    all_seeds = list(seeds)
    if extra_seed is not None:
        all_seeds.append(extra_seed.copy())
    for seed in all_seeds:
        q = solve_midpoint_ik(model, data, hand_id, left_id, right_id, target, seed, target_down)
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)
        midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
        err = np.linalg.norm(midpoint - target) + 0.2 * np.linalg.norm(np.cross(hand_z, target_down / max(np.linalg.norm(target_down), 1e-9)))
        candidates.append((err, q))
    return min(candidates, key=lambda item: item[0])[1]


def open_drawer(sim):
    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.data.ctrl[8] = DRAWER_OPEN_CTRL
    sim.step(200)
    move_q(sim, Q_HANDLE, 255, 260)
    move_q(sim, Q_HANDLE, 40, 160)
    move_q(sim, Q_PULL, 40, 320)
    hold(sim, Q_PULL, 40, 120)
    move_q(sim, Q_PULL, 255, 100)
    hold(sim, Q_PULL, 255, 40)
    move_q(sim, SAFE, 255, 180)
    hold(sim, SAFE, 255, 40)


def attempt_floor_pick(sim, seeds, target_down, xy_offset, grasp_z_extra, approach_z_extra, lift_z, retreat_q):
    model = sim.model
    data = sim.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    block = sim.block_position().copy()
    bx = float(block[0] + xy_offset[0])
    by = float(block[1] + xy_offset[1])
    bz = float(block[2])

    above_far = np.array([bx, by, max(bz + approach_z_extra, 0.24)], dtype=float)
    above = np.array([bx, by, max(bz + 0.14, 0.12)], dtype=float)
    grasp = np.array([bx, by, max(bz + grasp_z_extra, 0.065)], dtype=float)
    lift = np.array([bx, by, lift_z], dtype=float)

    q_above_far = best_q(model, data, hand_id, left_id, right_id, above_far, target_down, seeds, extra_seed=retreat_q)
    q_above = best_q(model, data, hand_id, left_id, right_id, above, target_down, seeds, extra_seed=q_above_far)
    q_grasp = best_q(model, data, hand_id, left_id, right_id, grasp, target_down, seeds, extra_seed=q_above)
    q_lift = best_q(model, data, hand_id, left_id, right_id, lift, target_down, seeds, extra_seed=q_grasp)

    move_q(sim, q_above_far, 255, 180)
    move_q(sim, q_above, 255, 120)
    move_q(sim, q_grasp, 255, 100)
    move_q(sim, q_grasp, 80, 120)
    move_q(sim, q_grasp, 0, 120)
    hold(sim, q_grasp, 0, 100)
    move_q(sim, q_lift, 0, 240)
    hold(sim, q_lift, 0, 260)

    final_block_z = float(sim.block_position()[2])
    held = sim.has_gripper_block_contact()
    return final_block_z, held, q_lift


def main():
    sim = Sim()
    model = sim.model
    data = sim.data

    seeds = [
        HOME.copy(),
        np.array([0.08, -0.19, -0.07, -1.88, -0.01, 1.69, 0.5]),
        np.array([-0.08, -0.19, 0.07, -1.88, 0.01, 1.69, -0.5]),
        Q_HANDLE.copy(),
        Q_PULL.copy(),
        SAFE.copy(),
    ]

    open_drawer(sim)

    pick_plan = [
        (DOWN, (0.0, 0.0), 0.060, 0.28, 0.68),
        (TILT_Y_PLUS, (0.0, 0.012), 0.055, 0.28, 0.68),
        (TILT_Y_MINUS, (0.0, -0.012), 0.055, 0.28, 0.68),
        (TILT_X_PLUS, (0.008, 0.0), 0.055, 0.28, 0.68),
        (TILT_X_MINUS, (-0.008, 0.0), 0.055, 0.28, 0.68),
    ]

    best_lift_q = SAFE.copy()
    for idx, (target_down, xy_offset, grasp_z_extra, approach_z_extra, lift_z) in enumerate(pick_plan):
        final_block_z, held, q_lift = attempt_floor_pick(
            sim,
            seeds=seeds,
            target_down=target_down,
            xy_offset=xy_offset,
            grasp_z_extra=grasp_z_extra,
            approach_z_extra=approach_z_extra,
            lift_z=lift_z,
            retreat_q=SAFE,
        )
        best_lift_q = q_lift
        if final_block_z >= 0.60 and held:
            break
        move_q(sim, SAFE, 255, 140)
        hold(sim, SAFE, 255, 60)

    hold(sim, best_lift_q, 0, 220)
    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
