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
FRONT = np.array([1.0, 0.0, -0.06], dtype=float)
FRONT_LEFT = np.array([0.995, 0.08, -0.06], dtype=float)
FRONT_RIGHT = np.array([0.995, -0.08, -0.06], dtype=float)
DRAWER_OPEN_CTRL = 1.0


def solve_midpoint_ik(model, data, hand_id, left_id, right_id, target_midpoint, seed, target_down):
    q = seed.copy()
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]
    target_down = np.array(target_down, dtype=float)
    target_down = target_down / max(np.linalg.norm(target_down), 1e-9)

    for _ in range(180):
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
        q = np.clip(q + 0.75 * dq, qmin, qmax)

    return q


def best_q(model, data, hand_id, left_id, right_id, target_xyz, target_down, seeds, extra_seed=None):
    target = np.array(target_xyz, dtype=float)
    candidates = []
    all_seeds = list(seeds)
    if extra_seed is not None:
        all_seeds.append(extra_seed.copy())
    target_down = np.array(target_down, dtype=float)
    target_down = target_down / max(np.linalg.norm(target_down), 1e-9)

    for seed in all_seeds:
        q = solve_midpoint_ik(model, data, hand_id, left_id, right_id, target, seed, target_down)
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)
        midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
        err = np.linalg.norm(midpoint - target) + 0.2 * np.linalg.norm(np.cross(hand_z, target_down))
        candidates.append((err, q))
    return min(candidates, key=lambda item: item[0])[1]


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


def open_drawer(sim):
    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.data.ctrl[8] = DRAWER_OPEN_CTRL
    sim.step(320)
    hold(sim, HOME, 255, 80)
    hold(sim, SAFE, 255, 80)


def front_extract_and_lift(sim, target_down, xy_offset, z_offset):
    model = sim.model
    data = sim.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    block = sim.block_position().copy()
    bx = float(block[0] + xy_offset[0])
    by = float(block[1] + xy_offset[1])
    bz = float(block[2] + z_offset)

    seeds = [HOME.copy(), SAFE.copy(), Q_HANDLE.copy(), Q_PULL.copy()]
    approach_far = np.array([0.845, by, bz + 0.03], dtype=float)
    approach_near = np.array([bx + 0.10, by, bz + 0.02], dtype=float)
    grasp = np.array([bx + 0.018, by, bz + 0.02], dtype=float)
    retract = np.array([0.835, by, bz + 0.05], dtype=float)
    lift = np.array([0.79, by, 0.69], dtype=float)

    q_approach_far = best_q(model, data, hand_id, left_id, right_id, approach_far, target_down, seeds, extra_seed=SAFE)
    q_approach_near = best_q(model, data, hand_id, left_id, right_id, approach_near, target_down, seeds, extra_seed=q_approach_far)
    q_grasp = best_q(model, data, hand_id, left_id, right_id, grasp, target_down, seeds, extra_seed=q_approach_near)
    q_retract = best_q(model, data, hand_id, left_id, right_id, retract, target_down, seeds, extra_seed=q_grasp)
    q_lift = best_q(model, data, hand_id, left_id, right_id, lift, target_down, seeds, extra_seed=q_retract)

    move_q(sim, q_approach_far, 255, 140)
    move_q(sim, q_approach_near, 255, 120)
    move_q(sim, q_grasp, 255, 120)
    move_q(sim, q_grasp, 80, 90)
    move_q(sim, q_grasp, 0, 90)
    hold(sim, q_grasp, 0, 80)
    move_q(sim, q_retract, 0, 180)
    hold(sim, q_retract, 0, 80)
    move_q(sim, q_lift, 0, 220)
    hold(sim, q_lift, 0, 240)

    return float(sim.block_position()[2]), bool(sim.has_gripper_block_contact()), q_lift


def main():
    sim = Sim()
    open_drawer(sim)

    plans = [
        (FRONT, (0.0, 0.0), 0.0),
        (FRONT_LEFT, (0.0, 0.0), 0.0),
        (FRONT_RIGHT, (0.0, 0.0), 0.0),
        (FRONT, (0.0, 0.008), 0.0),
        (FRONT, (0.0, -0.008), 0.0),
        (FRONT_LEFT, (0.0, 0.006), 0.004),
        (FRONT_RIGHT, (0.0, -0.006), 0.004),
    ]

    best_q_lift = SAFE.copy()
    best_z = float(sim.block_position()[2])
    for target_down, xy_offset, z_offset in plans:
        final_z, held, q_lift = front_extract_and_lift(sim, target_down, xy_offset, z_offset)
        if final_z > best_z:
            best_z = final_z
            best_q_lift = q_lift
        if final_z >= 0.60 and held:
            best_q_lift = q_lift
            break
        move_q(sim, SAFE, 255, 120)
        hold(sim, SAFE, 255, 60)

    hold(sim, best_q_lift, 0, 220)
    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
