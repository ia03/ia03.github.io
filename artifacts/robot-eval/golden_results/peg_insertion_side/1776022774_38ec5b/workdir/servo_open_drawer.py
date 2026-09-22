import numpy as np
import mujoco

from sim import Sim


OFFSET = np.array([0.0, 0.0, 0.058], dtype=float)
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
R_FRONT = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)


def quat_from_mat(R):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    return quat


def servo_pose(sim, target_p, target_R, grip, steps, q_nom=None, pos_gain=4.0, rot_gain=1.0, step_scale=0.15):
    if q_nom is None:
        q_nom = HOME
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    quat_t = quat_from_mat(target_R)
    qmin = sim.model.actuator_ctrlrange[:7, 0]
    qmax = sim.model.actuator_ctrlrange[:7, 1]
    for _ in range(steps):
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
        err = np.concatenate([pos_gain * e_p, rot_gain * e_r, 0.04 * (q_nom - sim.data.qpos[:7])])
        jacp = np.zeros((3, sim.model.nv))
        jacr = np.zeros((3, sim.model.nv))
        point = sim.data.xpos[hand_id].copy() + R @ OFFSET
        mujoco.mj_jac(sim.model, sim.data, jacp, jacr, point, hand_id)
        J = np.vstack([pos_gain * jacp[:, :7], rot_gain * jacr[:, :7], 0.04 * np.eye(7)])
        dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(7), J.T @ err)
        sim.data.ctrl[:7] = np.clip(sim.data.qpos[:7] + step_scale * dq, qmin, qmax)
        sim.data.ctrl[7] = grip
        sim.step(1)


def main():
    sim = Sim()
    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.step(200)

    targets = [
        (np.array([0.70, 0.04, 0.48]), 255, 200),
        (np.array([0.752, 0.04, 0.44]), 255, 220),
        (np.array([0.752, 0.04, 0.44]), 40, 180),
        (np.array([0.92, 0.04, 0.44]), 40, 360),
        (np.array([0.92, 0.04, 0.44]), 40, 120),
    ]
    for i, (p, grip, steps) in enumerate(targets, start=1):
        servo_pose(sim, p, R_FRONT, grip, steps)
        print(i, "drawer", sim.drawer_open_amount(), "block", sim.block_position(), flush=True)


if __name__ == "__main__":
    main()
