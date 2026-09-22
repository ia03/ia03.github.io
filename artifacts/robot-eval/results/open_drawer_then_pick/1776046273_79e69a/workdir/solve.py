import numpy as np
import mujoco

import sim


ARM_DOF = 7
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.0


def arm_qpos(data):
    return data.qpos[:ARM_DOF].copy()


def set_ctrl(s, arm, fingers, drawer):
    s.data.ctrl[:ARM_DOF] = arm
    s.data.ctrl[7] = fingers
    s.data.ctrl[8] = drawer


def solve_hand_position(model, data, target_pos, q_nominal, steps=80, damping=1e-2):
    """Position-only IK for the hand body."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    q = q_nominal.copy()
    for _ in range(steps):
        data.qpos[:ARM_DOF] = q
        mujoco.mj_forward(model, data)
        pos = data.xpos[bid].copy()
        err = target_pos - pos
        if np.linalg.norm(err) < 1e-4:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, bid)
        J = jacp[:, :ARM_DOF]
        # Damped least squares with a weak pull toward the nominal posture.
        A = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)
        dq += 0.02 * (q_nominal - q)
        q = q + np.clip(dq, -0.08, 0.08)
        q = np.clip(q, model.jnt_range[:ARM_DOF, 0], model.jnt_range[:ARM_DOF, 1])
    return q


def rotvec_from_matrix(R):
    """Convert a rotation matrix to an axis-angle vector."""
    trace = np.trace(R)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    if np.pi - theta < 1e-5:
        # Near-180 degree rotations need a more careful axis extraction.
        axis = np.sqrt(np.maximum(np.diag(R) + 1.0, 0.0))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis /= np.linalg.norm(axis)
        return axis * theta
    skew = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    axis = skew / (2.0 * np.sin(theta))
    return axis * theta


def solve_hand_pose(model, data, target_pos, target_rot, q_nominal, steps=100, damping=1e-2):
    """6D IK for the hand body pose, with a weak pull toward the nominal posture."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    q = q_nominal.copy()
    for _ in range(steps):
        data.qpos[:ARM_DOF] = q
        mujoco.mj_forward(model, data)
        pos = data.xpos[bid].copy()
        cur_rot = np.array(data.xmat[bid]).reshape(3, 3)
        err_pos = target_pos - pos
        err_rot = rotvec_from_matrix(target_rot @ cur_rot.T)
        err = np.concatenate([err_pos, 0.7 * err_rot])
        if np.linalg.norm(err_pos) < 1e-4 and np.linalg.norm(err_rot) < 1e-3:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, bid)
        J = np.vstack([jacp[:, :ARM_DOF], 0.7 * jacr[:, :ARM_DOF]])
        A = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(A, err)
        dq += 0.02 * (q_nominal - q)
        q = q + np.clip(dq, -0.07, 0.07)
        q = np.clip(q, model.jnt_range[:ARM_DOF, 0], model.jnt_range[:ARM_DOF, 1])
    return q


def run_stage(s, arm_target, finger_target, drawer_target, steps, settle=0):
    for _ in range(steps):
        set_ctrl(s, arm_target, finger_target, drawer_target)
        s.step(1)
    for _ in range(settle):
        set_ctrl(s, arm_target, finger_target, drawer_target)
        s.step(1)


def main():
    s = sim.Sim()
    m, d = s.model, s.data

    # Home posture from the Panda keyframe. This wrist orientation is a good
    # match for a front grasp because the jaws close along world x here.
    home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
    open_fingers = FINGER_OPEN
    closed_fingers = FINGER_CLOSED
    drawer_open = 1.0
    home_rot = None
    # Record the home wrist orientation from the actual model state so later
    # IK keeps the same grasp-friendly frame.
    d.qpos[:ARM_DOF] = home
    mujoco.mj_forward(m, d)
    hand_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    home_rot = np.array(d.xmat[hand_bid]).reshape(3, 3)

    # Stage 1: move to the home arm posture while opening the drawer.
    run_stage(s, home, open_fingers, drawer_open, steps=250)

    # Stage 2: move in front of the block with the fingers open.
    q_pre = solve_hand_pose(m, d, np.array([0.77, 0.0, 0.56]), home_rot, home)
    run_stage(s, q_pre, open_fingers, drawer_open, steps=220)

    # Stage 3: descend to a grasp pose around the block.
    q_grasp = solve_hand_pose(m, d, np.array([0.77, 0.0, 0.50]), home_rot, q_pre)
    run_stage(s, q_grasp, open_fingers, drawer_open, steps=220)

    # Stage 4: close the fingers to capture the block.
    run_stage(s, q_grasp, 0.02, drawer_open, steps=200, settle=120)

    # Stage 5: lift straight up while keeping the drawer open.
    q_pull = solve_hand_pose(m, d, np.array([0.80, 0.0, 0.50]), home_rot, q_grasp)
    run_stage(s, q_pull, 0.02, drawer_open, steps=280, settle=160)
    q_lift = solve_hand_pose(m, d, np.array([0.80, 0.0, 0.64]), home_rot, q_pull)
    run_stage(s, q_lift, 0.02, drawer_open, steps=320, settle=200)

    # Leave the hand holding the block for additional settle time.
    run_stage(s, q_lift, 0.02, drawer_open, steps=220, settle=320)

    print("drawer", s.drawer_open_amount())
    print("block", s.block_position())
    print("contact", s.has_gripper_block_contact())
    s.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
