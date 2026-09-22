import numpy as np
import mujoco

from sim import Sim


HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
Q_HANDLE = np.array(
    [0.260114734942896, 0.5978935416914477, -0.2548739609310991, -1.7011568239308756, 0.0731417910040021, 3.7525, 0.8723261334629767],
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


def main():
    sim = Sim()
    model = sim.model
    data = sim.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.data.ctrl[8] = 1.0
    sim.step(200)
    move_q(sim, Q_HANDLE, 255, 260)
    hold(sim, Q_HANDLE, 255, 220, drawer_ctrl=0.0)
    block = sim.block_position().copy()
    target = np.array([block[0], block[1], max(block[2] + 0.02, 0.03)], dtype=float)
    seed = HOME
    q = solve_midpoint_ik(model, data, hand_id, left_id, right_id, target, seed, FRONT_IN)
    data.qpos[:7] = q
    data.qpos[7:9] = 0.04
    mujoco.mj_forward(model, data)
    print("target", np.round(target, 4))
    print("mid", np.round(0.5 * (data.xpos[left_id] + data.xpos[right_id]), 4), "hand", np.round(data.xpos[hand_id], 4))
    for grips in [(255, 60, 0), (180, 80, 0), (120, 60, 0), (80, 40, 0), (60, 30, 0)]:
        sim2 = Sim()
        sim2.data.ctrl[:7] = HOME
        sim2.data.ctrl[7] = 255
        sim2.data.ctrl[8] = 1.0
        sim2.step(200)
        move_q(sim2, Q_HANDLE, 255, 260)
        hold(sim2, Q_HANDLE, 255, 220, drawer_ctrl=0.0)
        q2 = solve_midpoint_ik(model, sim2.data, hand_id, left_id, right_id, target, HOME, FRONT_IN)
        move_q(sim2, q2, grips[0], 120, drawer_ctrl=0.0)
        move_q(sim2, q2, grips[1], 120, drawer_ctrl=0.0)
        move_q(sim2, q2, grips[2], 120, drawer_ctrl=0.0)
        hold(sim2, q2, grips[2], 120, drawer_ctrl=0.0)
        print({"grips": grips, "end": sim2.block_position().tolist(), "contact": sim2.has_gripper_block_contact()})


if __name__ == "__main__":
    main()
