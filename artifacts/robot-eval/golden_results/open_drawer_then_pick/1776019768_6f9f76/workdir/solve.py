import numpy as np
import mujoco

from sim import Sim


HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
Q_HANDLE = np.array(
    [0.260114734942896, 0.5978935416914477, -0.2548739609310991, -1.7011568239308756, 0.0731417910040021, 3.7525, 0.8723261334629767],
    dtype=float,
)
Q_PULL = np.array(
    [0.12598874856351472, 1.1973462499176968, -0.34850701301211456, -0.625005200213342, -0.022487162744560703, 3.3513450345879296, 1.1390361931507589],
    dtype=float,
)
TARGET_DOWN = np.array([0.0, 0.0, -1.0], dtype=float)
DRAWER_OPEN_CTRL = 1.0


def solve_midpoint_ik(model, data, hand_id, left_id, right_id, target_midpoint, seed):
    q = seed.copy()
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]

    for _ in range(160):
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)

        hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
        midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        pos_err = target_midpoint - midpoint
        ori_err = np.cross(hand_z, TARGET_DOWN)

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


def main():
    sim = Sim()
    model = sim.model
    data = sim.data

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    seeds = [
        HOME.copy(),
        np.array([0.08, -0.19, -0.07, -1.88, -0.01, 1.69, 0.5]),
        np.array([-0.08, -0.19, 0.07, -1.88, 0.01, 1.69, -0.5]),
        Q_HANDLE.copy(),
        Q_PULL.copy(),
    ]

    def best_q(target_xyz, extra_seed=None):
        target = np.array(target_xyz, dtype=float)
        candidates = []
        all_seeds = seeds + ([extra_seed.copy()] if extra_seed is not None else [])
        for seed in all_seeds:
            q = solve_midpoint_ik(model, data, hand_id, left_id, right_id, target, seed)
            data.qpos[:7] = q
            data.qpos[7:9] = 0.04
            mujoco.mj_forward(model, data)
            midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
            hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
            err = np.linalg.norm(midpoint - target) + 0.2 * np.linalg.norm(np.cross(hand_z, TARGET_DOWN))
            candidates.append((err, q))
        return min(candidates, key=lambda item: item[0])[1]

    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.data.ctrl[8] = DRAWER_OPEN_CTRL
    sim.step(200)

    # Drawer opening sequence from validated probe trajectory.
    move_q(sim, Q_HANDLE, 255, 260)
    move_q(sim, Q_HANDLE, 40, 160)
    move_q(sim, Q_PULL, 40, 320)
    hold(sim, Q_PULL, 40, 120)

    # Let the drawer settle open, then release and move to the block.
    move_q(sim, Q_PULL, 255, 100)
    hold(sim, Q_PULL, 255, 40)

    block = sim.block_position().copy()
    above = np.array([block[0], block[1], 0.54], dtype=float)
    grasp = np.array([block[0], block[1], 0.468], dtype=float)
    lift = np.array([block[0], block[1], 0.63], dtype=float)

    q_above = best_q(above, extra_seed=Q_PULL)
    q_grasp = best_q(grasp, extra_seed=q_above)
    q_lift = best_q(lift, extra_seed=q_grasp)

    move_q(sim, q_above, 255, 200)
    move_q(sim, q_grasp, 255, 120)
    move_q(sim, q_grasp, 40, 160)
    move_q(sim, q_grasp, 0, 80)
    hold(sim, q_grasp, 0, 60)
    move_q(sim, q_lift, 0, 180)
    hold(sim, q_lift, 0, 260)

    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
