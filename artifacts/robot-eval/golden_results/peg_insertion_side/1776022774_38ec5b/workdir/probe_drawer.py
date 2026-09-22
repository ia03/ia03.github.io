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


def move_q(sim, q_target, grip, steps=220):
    q_start = sim.data.qpos[:7].copy()
    g_start = float(sim.data.ctrl[7])
    for i in range(steps):
        a = (i + 1) / steps
        sim.data.ctrl[:7] = (1 - a) * q_start + a * q_target
        sim.data.ctrl[7] = (1 - a) * g_start + a * grip
        sim.step(1)


def run_trial(R, p1, p2, grip_close):
    sim = Sim()
    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.step(200)
    q1 = solve_pose(sim, np.array(p1, dtype=float), R, HOME)
    q2 = solve_pose(sim, np.array(p2, dtype=float), R, q1)
    move_q(sim, q1, 255, 240)
    d1 = sim.drawer_open_amount()
    move_q(sim, q1, grip_close, 140)
    d2 = sim.drawer_open_amount()
    move_q(sim, q2, grip_close, 240)
    d3 = sim.drawer_open_amount()
    return d1, d2, d3


def main():
    rots = {
        "front_y": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float),
        "front_neg_y": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float),
        "top_down": np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float),
    }
    for name, R in rots.items():
        for z in [0.44, 0.45, 0.46]:
            for y in [-0.02, 0.0, 0.02]:
                for close in [0, 30, 60, 100]:
                    p1 = [0.752, y, z]
                    p2 = [0.84, y, z]
                    d1, d2, d3 = run_trial(R, p1, p2, close)
                    print(
                        name,
                        "z",
                        z,
                        "y",
                        y,
                        "close",
                        close,
                        "drawer",
                        round(d1, 4),
                        round(d2, 4),
                        round(d3, 4),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
