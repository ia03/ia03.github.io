import numpy as np
import mujoco

from sim import Sim


OFFSET = np.array([0.0, 0.0, 0.058], dtype=float)
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)


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


def main():
    R = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)
    for z in [0.44, 0.45, 0.46]:
        for y in [0.0, 0.02, 0.04]:
            for x2 in [0.84, 0.88, 0.92, 0.96]:
                sim = Sim()
                sim.data.ctrl[:7] = HOME
                sim.data.ctrl[7] = 255
                sim.step(200)
                q1 = solve_pose(sim, np.array([0.752, y, z]), R, HOME)
                q2 = solve_pose(sim, np.array([x2, y, z]), R, q1)
                move_q(sim, q1, 255, 260)
                d1 = sim.drawer_open_amount()
                move_q(sim, q1, 40, 160)
                d2 = sim.drawer_open_amount()
                move_q(sim, q2, 40, 320)
                d3 = sim.drawer_open_amount()
                print("z", z, "y", y, "x2", x2, "drawer", round(d1, 4), round(d2, 4), round(d3, 4), flush=True)
                if d3 > 0.1:
                    print("q1_full", repr(q1.tolist()), "q2_full", repr(q2.tolist()), flush=True)


if __name__ == "__main__":
    main()
