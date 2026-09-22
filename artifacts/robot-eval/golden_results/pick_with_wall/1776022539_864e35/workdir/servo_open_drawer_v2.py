import numpy as np
import mujoco

from sim import Sim


OFFSET = np.array([0.0, 0.0, 0.058], dtype=float)
HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
R_FRONT = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=float)
Q_HANDLE_HINT = np.array(
    [0.2887739083550356, 0.6178838727508454, -0.2969239199163194, -1.7053069639778142, 0.06592295071522526, 3.7525, 0.9064924581847854],
    dtype=float,
)
Q_PULL_HINT = np.array(
    [0.13213800116989255, 1.214627915621127, -0.37697910858577277, -0.6215977616246623, -0.01917876978463884, 3.361660629985802, 1.164405291264782],
    dtype=float,
)


def quat_from_mat(R):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.reshape(-1))
    return quat


def servo_pose(sim, target_p, target_R, grip, steps, q_nom, pos_gain=6.0, rot_gain=1.5, null_gain=0.08, step_scale=0.18):
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
        err = np.concatenate([pos_gain * e_p, rot_gain * e_r, null_gain * (q_nom - sim.data.qpos[:7])])
        jacp = np.zeros((3, sim.model.nv))
        jacr = np.zeros((3, sim.model.nv))
        point = sim.data.xpos[hand_id].copy() + R @ OFFSET
        mujoco.mj_jac(sim.model, sim.data, jacp, jacr, point, hand_id)
        J = np.vstack([pos_gain * jacp[:, :7], rot_gain * jacr[:, :7], null_gain * np.eye(7)])
        dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(7), J.T @ err)
        sim.data.ctrl[:7] = np.clip(sim.data.qpos[:7] + step_scale * dq, qmin, qmax)
        sim.data.ctrl[7] = grip
        sim.step(1)


def main():
    sim = Sim()
    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.step(200)
    phases = [
        ("approach", np.array([0.70, 0.04, 0.48]), 255, 160, Q_HANDLE_HINT),
        ("handle", np.array([0.752, 0.04, 0.44]), 255, 200, Q_HANDLE_HINT),
        ("close", np.array([0.752, 0.04, 0.44]), 40, 180, Q_HANDLE_HINT),
        ("pull1", np.array([0.82, 0.04, 0.44]), 40, 220, Q_PULL_HINT),
        ("pull2", np.array([0.92, 0.04, 0.44]), 40, 260, Q_PULL_HINT),
        ("hold", np.array([0.92, 0.04, 0.44]), 40, 120, Q_PULL_HINT),
    ]
    for name, pos, grip, steps, q_nom in phases:
        servo_pose(sim, pos, R_FRONT, grip, steps, q_nom)
        print(name, "drawer", sim.drawer_open_amount(), "block", sim.block_position(), flush=True)


if __name__ == "__main__":
    main()
