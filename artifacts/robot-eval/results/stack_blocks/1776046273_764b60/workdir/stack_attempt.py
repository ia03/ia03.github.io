import numpy as np
import mujoco

import sim


ARM_DOF = 7
HAND_BODY = "hand"


def arm_qpos(data):
    return data.qpos[:ARM_DOF].copy()


def set_arm_ctrl(s, q_arm, gripper=None):
    s.data.ctrl[:ARM_DOF] = q_arm
    if gripper is not None:
        s.data.ctrl[7] = gripper


def hand_pos(s):
    bid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    return s.data.xpos[bid].copy()


def hand_rot(s):
    bid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    return s.data.xmat[bid].reshape(3, 3).copy()


def rot_error(R_cur, R_des):
    R_err = R_des @ R_cur.T
    return 0.5 * np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )


def resolved_rate_step(s, q, target_pos, target_rot, pos_gain=0.18, ori_gain=0.08, damp=1e-2):
    bid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    s.data.qpos[:ARM_DOF] = q
    mujoco.mj_forward(s.model, s.data)
    cur = s.data.xpos[bid].copy()
    R_cur = s.data.xmat[bid].reshape(3, 3).copy()
    pos_err = target_pos - cur
    ori_err = rot_error(R_cur, target_rot)

    jacp = np.zeros((3, s.model.nv))
    jacr = np.zeros((3, s.model.nv))
    mujoco.mj_jacBody(s.model, s.data, jacp, jacr, bid)
    J = np.vstack([pos_gain * jacp[:, :ARM_DOF], ori_gain * jacr[:, :ARM_DOF]])
    err = np.concatenate([pos_gain * pos_err, ori_gain * ori_err])
    dq = J.T @ np.linalg.solve(J @ J.T + (damp**2) * np.eye(6), err)
    q = q + dq
    low = s.model.jnt_range[:ARM_DOF, 0]
    high = s.model.jnt_range[:ARM_DOF, 1]
    return np.minimum(np.maximum(q, low), high)


def move_arm(s, target_pos, target_rot, steps=180, gripper=None):
    q = arm_qpos(s.data)
    for _ in range(steps):
        q = resolved_rate_step(s, q, np.array(target_pos, dtype=float), target_rot)
        set_arm_ctrl(s, q, gripper=gripper)
        s.step(1)
    return q


def move_cartesian(s, target_pos, target_rot, segments=25, settle=8, gripper=None):
    start = hand_pos(s)
    q = arm_qpos(s.data)
    for i in range(segments):
        a = (i + 1) / segments
        waypoint = (1 - a) * start + a * np.array(target_pos, dtype=float)
        for _ in range(settle):
            q = resolved_rate_step(s, q, waypoint, target_rot)
            set_arm_ctrl(s, q, gripper=gripper)
            s.step(1)
    return q


def hold(s, steps, gripper=None):
    q = arm_qpos(s.data)
    for _ in range(steps):
        set_arm_ctrl(s, q, gripper=gripper)
        s.step(1)


def squeeze_grasp(s, start_pos, end_pos, target_rot, steps=30):
    q = arm_qpos(s.data)
    start = np.array(start_pos, dtype=float)
    end = np.array(end_pos, dtype=float)
    for i in range(steps):
        a = (i + 1) / steps
        waypoint = (1 - a) * start + a * end
        q = resolved_rate_step(s, q, waypoint, target_rot)
        gripper = int(round(255 * (1 - a)))
        set_arm_ctrl(s, q, gripper=gripper)
        s.step(1)
    return q


def run():
    s = sim.Sim()
    hand_target_rot = hand_rot(s)

    # Open the gripper and let the arm settle at the start pose.
    hold(s, 50, gripper=255)

    red = s.block_positions()["red"]
    green = s.block_positions()["green"]

    # Keep the hand high while translating laterally, then descend vertically.
    safe_above_red = red + np.array([0.0, 0.0, 0.34])
    pregrasp_red = red + np.array([0.0, 0.0, 0.155])
    grasp_red = red + np.array([0.0, 0.0, 0.112])
    lift_red = red + np.array([0.0, 0.0, 0.30])

    move_cartesian(s, safe_above_red, hand_target_rot, segments=30, settle=6, gripper=255)
    move_cartesian(s, pregrasp_red, hand_target_rot, segments=12, settle=6, gripper=255)
    squeeze_grasp(s, pregrasp_red, grasp_red, hand_target_rot, steps=28)
    hold(s, 160, gripper=0)
    move_cartesian(s, lift_red, hand_target_rot, segments=18, settle=8, gripper=0)

    # Move above green and place the block gently.
    safe_above_green = green + np.array([0.0, 0.0, 0.36])
    place_height = green + np.array([0.0, 0.0, 0.26])
    retreat = green + np.array([0.16, -0.06, 0.28])

    move_cartesian(s, safe_above_green, hand_target_rot, segments=32, settle=6, gripper=0)
    # Lower in small increments until the carried block is near the desired stack height.
    desired_red_z = green[2] + 0.050
    current_target = hand_pos(s)
    for _ in range(24):
        if s.block_positions()["red"][2] <= desired_red_z + 0.002:
            break
        current_target = current_target + np.array([0.0, 0.0, -0.008])
        move_cartesian(s, current_target, hand_target_rot, segments=5, settle=5, gripper=0)

    # Release, then withdraw after the block has time to settle on the green block.
    hold(s, 160, gripper=255)
    move_cartesian(s, retreat, hand_target_rot, segments=20, settle=8, gripper=255)
    hold(s, 400, gripper=255)

    print("final positions", s.block_positions())
    print("time", s.data.time)
    s.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    run()
