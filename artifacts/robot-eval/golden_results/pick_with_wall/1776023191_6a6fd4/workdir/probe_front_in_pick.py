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
FRONT_IN = np.array([0.65, 0.0, -0.76], dtype=float)


def solve_midpoint_ik(model, data, hand_id, left_id, right_id, target_midpoint, seed, target_down):
    q = seed.copy()
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]
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


def move_q(sim, q_target, grip, steps, drawer_ctrl=1.0):
    q_start = sim.data.qpos[:7].copy()
    g_start = float(sim.data.ctrl[7])
    d_start = float(sim.data.ctrl[8])
    for i in range(steps):
        a = (i + 1) / steps
        sim.data.ctrl[:7] = (1 - a) * q_start + a * q_target
        sim.data.ctrl[7] = (1 - a) * g_start + a * grip
        sim.data.ctrl[8] = (1 - a) * d_start + a * drawer_ctrl
        sim.step(1)


def hold(sim, q_target, grip, steps, drawer_ctrl=1.0):
    for _ in range(steps):
        sim.data.ctrl[:7] = q_target
        sim.data.ctrl[7] = grip
        sim.data.ctrl[8] = drawer_ctrl
        sim.step(1)


def open_drawer(sim):
    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.data.ctrl[8] = 1.0
    sim.step(200)
    move_q(sim, Q_HANDLE, 255, 260)
    move_q(sim, Q_HANDLE, 40, 160)
    move_q(sim, Q_PULL, 40, 320)
    hold(sim, Q_PULL, 40, 120)
    move_q(sim, Q_PULL, 255, 100)
    hold(sim, Q_PULL, 255, 40)
    move_q(sim, SAFE, 255, 180)
    hold(sim, SAFE, 255, 40)


def main():
    sim = Sim()
    model = sim.model
    data = sim.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    open_drawer(sim)
    block = sim.block_position().copy()
    print("block", np.round(block, 6))
    seeds = [HOME, SAFE, Q_HANDLE, Q_PULL]
    for z_extra in [0.01, 0.015, 0.02, 0.025, 0.03]:
        target = np.array([block[0], block[1], max(block[2] + z_extra, 0.03)], dtype=float)
        best = None
        for seed in seeds:
            q = solve_midpoint_ik(model, data, hand_id, left_id, right_id, target, seed, FRONT_IN)
            data.qpos[:7] = q
            data.qpos[7:9] = 0.04
            mujoco.mj_forward(model, data)
            midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
            err = np.linalg.norm(midpoint - target)
            if best is None or err < best[0]:
                best = (err, q, midpoint, data.xpos[hand_id].copy(), data.xpos[left_id].copy(), data.xpos[right_id].copy())
        print("z", z_extra, "err", round(float(best[0]), 4), "mid", np.round(best[2], 4), "hand", np.round(best[3], 4))
        q = best[1]
        move_q(sim, q, 255, 120)
        move_q(sim, q, 60, 120)
        move_q(sim, q, 0, 120)
        hold(sim, q, 0, 80)
        print(" after close", np.round(sim.block_position(), 4), "contact", sim.has_gripper_block_contact())
        move_q(sim, q, 0, 160)
        hold(sim, q, 0, 160)
        print(" after lift", np.round(sim.block_position(), 4), "contact", sim.has_gripper_block_contact())


if __name__ == "__main__":
    main()
