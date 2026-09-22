import math
import numpy as np
import mujoco

import sim


ARM_JOINTS = list(range(7))
FINGER_ACT = 7


def quat_from_mat(mat):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    return q


def mat_from_quat(q):
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


def hand_pose(s):
    bid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    return s.data.xpos[bid].copy(), s.data.xmat[bid].reshape(3, 3).copy()


def body_jacobian(s, body_name="hand"):
    bid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jacp = np.zeros((3, s.model.nv))
    jacr = np.zeros((3, s.model.nv))
    mujoco.mj_jacBody(s.model, s.data, jacp, jacr, bid)
    return jacp[:, ARM_JOINTS], jacr[:, ARM_JOINTS]


def ik_to_target(s, target_pos, target_mat=None, n_iter=120, tol=1e-4):
    q = s.data.qpos.copy()
    for _ in range(n_iter):
        mujoco.mj_forward(s.model, s.data)
        pos, mat = hand_pose(s)
        pos_err = target_pos - pos
        if target_mat is None:
            err = pos_err
            if np.linalg.norm(err) < tol:
                break
            jacp, _ = body_jacobian(s)
            J = jacp
        else:
            # Orientation error in axis-angle form.
            rel = target_mat @ mat.T
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, rel.reshape(-1))
            if quat[0] < 0:
                quat *= -1
            ang = 2.0 * math.acos(np.clip(quat[0], -1.0, 1.0))
            axis = quat[1:]
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-9 or ang < 1e-6:
                ori_err = np.zeros(3)
            else:
                ori_err = axis / axis_norm * ang
            err = np.concatenate([pos_err, 0.35 * ori_err])
            if np.linalg.norm(pos_err) < tol and np.linalg.norm(ori_err) < 5e-3:
                break
            jacp, jacr = body_jacobian(s)
            J = np.vstack([jacp, 0.35 * jacr])

        # Damped least squares on active joints only.
        lam = 1e-2
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), err)
        q[ARM_JOINTS] = np.clip(
            q[ARM_JOINTS] + dq,
            s.model.jnt_range[:7, 0],
            s.model.jnt_range[:7, 1],
        )
        s.data.qpos[:] = q
        mujoco.mj_forward(s.model, s.data)
    return q[ARM_JOINTS].copy()


def run_waypoint(s, q_target, finger_target, steps=180):
    ctrl = np.zeros(s.model.nu)
    ctrl[:7] = q_target
    ctrl[7] = finger_target
    for _ in range(steps):
        s.data.ctrl[:] = ctrl
        s.step(1)


def main():
    s = sim.Sim()
    mujoco.mj_forward(s.model, s.data)

    # Save an early plausible attempt so the replay artifact exists.
    s.save_final_state("/work/final_state.npz")

    hand_pos, hand_mat = hand_pose(s)
    cup = s.cup_position().copy()
    bin_center = np.array(sim.BIN_CENTER)

    # Keep the gripper in the same orientation for a simple top grasp.
    orient = hand_mat.copy()

    pre_grasp = cup + np.array([0.0, 0.0, 0.14])
    grasp = cup + np.array([0.0, 0.0, 0.055])
    lift = np.array([cup[0], cup[1], 0.58])
    above_bin = np.array([bin_center[0], bin_center[1], 0.60])
    place = np.array([bin_center[0], bin_center[1], 0.455])
    retreat = np.array([bin_center[0] - 0.05, bin_center[1], 0.62])

    q = s.data.qpos[:7].copy()
    for target_pos, finger, steps in [
        (pre_grasp, 0, 220),
        (grasp, 0, 160),
        (grasp, 255, 220),
        (lift, 255, 260),
        (above_bin, 255, 280),
        (place, 255, 180),
        (place, 0, 160),
        (retreat, 0, 180),
    ]:
        q = ik_to_target(s, target_pos, orient)
        run_waypoint(s, q, finger, steps=steps)
        print(
            "t", round(float(s.data.time), 3),
            "cup", np.round(s.cup_position(), 4),
            "contact", s.has_gripper_cup_contact(),
            "finger", np.round(s.data.qpos[7:9], 4),
        )

    # Let the object settle after release, then save the trajectory/state.
    for _ in range(500):
        s.data.ctrl[:] = np.array([*q, 0.0])
        s.step(1)

    print("final cup", np.round(s.cup_position(), 4))
    print("final contact", s.has_gripper_cup_contact())
    s.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
