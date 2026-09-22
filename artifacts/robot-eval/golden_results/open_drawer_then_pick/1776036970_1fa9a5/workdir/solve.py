import numpy as np
import mujoco

from sim import Sim


HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
SAFE = np.array([0.0, -0.35, 0.0, -1.90, 0.0, 1.95, 0.785], dtype=float)
SEEDS = [
    HOME.copy(),
    SAFE.copy(),
    np.array([0.08, -0.19, -0.07, -1.88, -0.01, 1.69, 0.5], dtype=float),
    np.array([-0.08, -0.19, 0.07, -1.88, 0.01, 1.69, -0.5], dtype=float),
]
DRAWER_OPEN_CTRL = 1.0


def solve_hand_ik(model, data, hand_id, target_pos, seed, target_down):
    q = seed.copy()
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]
    target_down = np.array(target_down, dtype=float)
    target_down = target_down / max(np.linalg.norm(target_down), 1e-9)

    for _ in range(240):
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)

        hand_pos = data.xpos[hand_id].copy()
        hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
        pos_err = np.array(target_pos, dtype=float) - hand_pos
        ori_err = np.cross(hand_z, target_down)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
        jac = np.vstack([jacp[:, :7], 0.35 * jacr[:, :7]])
        err = np.concatenate([pos_err, 0.35 * ori_err])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), err)
        q = np.clip(q + 0.7 * dq, qmin, qmax)

    return q


def best_q(model, data, hand_id, target_pos, target_down, seeds, extra_seed=None):
    target_down = np.array(target_down, dtype=float)
    target_down = target_down / max(np.linalg.norm(target_down), 1e-9)
    candidates = []
    all_seeds = list(seeds)
    if extra_seed is not None:
        all_seeds.append(extra_seed.copy())

    for seed in all_seeds:
        q = solve_hand_ik(model, data, hand_id, target_pos, seed, target_down)
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)
        hand_pos = data.xpos[hand_id].copy()
        hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
        err = np.linalg.norm(hand_pos - np.array(target_pos, dtype=float)) + 0.2 * np.linalg.norm(np.cross(hand_z, target_down))
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
    hold(sim, HOME, 255, 60)
    hold(sim, SAFE, 255, 60)


def pick_presented_block(sim):
    model = sim.model
    data = sim.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    bx, by, bz = sim.block_position().copy()

    q_above = best_q(model, data, hand_id, [bx, by, bz + 0.16], [0.0, 0.0, -1.0], SEEDS, extra_seed=SAFE)
    q_pregrasp = best_q(model, data, hand_id, [bx, by, bz + 0.08], [0.0, 0.0, -1.0], SEEDS, extra_seed=q_above)
    q_grasp = best_q(model, data, hand_id, [bx, by, bz + 0.03], [0.0, 0.0, -1.0], SEEDS, extra_seed=q_pregrasp)
    q_lift = best_q(model, data, hand_id, [0.68, by, 0.82], [0.0, 0.0, -1.0], SEEDS, extra_seed=q_grasp)

    move_q(sim, q_above, 255, 180)
    move_q(sim, q_pregrasp, 255, 120)
    move_q(sim, q_grasp, 255, 100)
    move_q(sim, q_grasp, 80, 100)
    move_q(sim, q_grasp, 0, 120)
    hold(sim, q_grasp, 0, 320)
    move_q(sim, q_lift, 0, 260)
    hold(sim, q_lift, 0, 300)


def main():
    sim = Sim()
    open_drawer(sim)
    pick_presented_block(sim)
    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
