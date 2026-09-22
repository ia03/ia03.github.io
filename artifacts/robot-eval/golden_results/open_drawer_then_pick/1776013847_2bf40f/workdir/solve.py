import numpy as np
import mujoco

from sim import Sim


OFFSET = np.array([0.0, 0.0, 0.058], dtype=float)
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
R_FRONT = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)
R_TOP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)


def quat_from_mat(R):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    return quat


def solve_pose(sim, target_p, target_R, q_init, iters=350):
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    q = np.array(q_init, dtype=float)
    quat_t = quat_from_mat(target_R)
    for _ in range(iters):
        sim.data.qpos[:7] = q
        sim.data.qvel[:] = 0
        mujoco.mj_forward(sim.model, sim.data)
        R = sim.data.xmat[hand_id].reshape(3, 3).copy()
        p = sim.data.xpos[hand_id].copy() + R @ OFFSET
        e_p = target_p - p
        quat_c = quat_from_mat(R)
        qconj = np.array([quat_c[0], -quat_c[1], -quat_c[2], -quat_c[3]])
        qerr = np.zeros(4)
        mujoco.mju_mulQuat(qerr, quat_t, qconj)
        if qerr[0] < 0:
            qerr *= -1
        e_r = np.zeros(3)
        mujoco.mju_quat2Vel(e_r, qerr, 1.0)
        err = np.concatenate([e_p, 0.3 * e_r])
        jacp = np.zeros((3, sim.model.nv))
        jacr = np.zeros((3, sim.model.nv))
        point = sim.data.xpos[hand_id].copy() + R @ OFFSET
        mujoco.mj_jac(sim.model, sim.data, jacp, jacr, point, hand_id)
        J = np.vstack([jacp[:, :7], 0.3 * jacr[:, :7]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
        q += 0.7 * dq
        for i in range(7):
            lo, hi = sim.model.jnt_range[i]
            q[i] = np.clip(q[i], lo, hi)
    return q


def move_q(sim, q_target, grip, steps=260):
    q_start = sim.data.qpos[:7].copy()
    g_start = float(sim.data.ctrl[7])
    for i in range(steps):
        a = (i + 1) / steps
        sim.data.ctrl[:7] = (1 - a) * q_start + a * q_target
        sim.data.ctrl[7] = (1 - a) * g_start + a * grip
        sim.step(1)


def hold(sim, q_target, grip, steps):
    for _ in range(steps):
        sim.data.ctrl[:7] = q_target
        sim.data.ctrl[7] = grip
        sim.step(1)


def main():
    plan = Sim()
    plan.data.ctrl[:7] = HOME
    plan.data.ctrl[7] = 255
    plan.step(200)

    # Stage 1: open the drawer with the mechanically validated pull.
    q_handle = solve_pose(plan, np.array([0.752, 0.04, 0.44]), R_FRONT, HOME)
    q_pull = solve_pose(plan, np.array([0.92, 0.04, 0.44]), R_FRONT, q_handle)
    q_retract = solve_pose(plan, np.array([0.72, 0.06, 0.52]), R_FRONT, q_pull)

    # Stage 2: top-down grasp the block from the opened drawer and lift it.
    q_above = solve_pose(plan, np.array([0.615, -0.02, 0.56]), R_TOP, q_retract)
    q_grasp = solve_pose(plan, np.array([0.615, -0.02, 0.45]), R_TOP, q_above)
    q_lift = solve_pose(plan, np.array([0.60, -0.02, 0.68]), R_TOP, q_grasp)
    q_finish = solve_pose(plan, np.array([0.56, -0.02, 0.70]), R_TOP, q_lift)

    sim = Sim()
    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.step(200)

    move_q(sim, q_handle, 255, 260)
    move_q(sim, q_handle, 40, 160)
    move_q(sim, q_pull, 40, 320)
    hold(sim, q_pull, 40, 80)
    move_q(sim, q_retract, 255, 220)

    move_q(sim, q_above, 255, 240)
    move_q(sim, q_grasp, 255, 160)
    move_q(sim, q_grasp, 0, 160)
    hold(sim, q_grasp, 0, 80)
    move_q(sim, q_lift, 0, 260)
    move_q(sim, q_finish, 0, 220)
    hold(sim, q_finish, 0, 320)

    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
