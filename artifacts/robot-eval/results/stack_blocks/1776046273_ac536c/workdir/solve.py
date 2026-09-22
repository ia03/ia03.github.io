import math
import numpy as np
import mujoco

import sim


ARM_DOF = 7
FINGER_CTRL = 7


def mat_from_xmat(xmat):
    return np.asarray(xmat, dtype=float).reshape(3, 3).copy()


def rotvec_from_matrix(r):
    trace = float(np.trace(r))
    c = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    angle = math.acos(c)
    if angle < 1e-8:
        return np.zeros(3)
    s = math.sin(angle)
    if abs(s) < 1e-8:
        return np.zeros(3)
    axis = np.array(
        [
            r[2, 1] - r[1, 2],
            r[0, 2] - r[2, 0],
            r[1, 0] - r[0, 1],
        ],
        dtype=float,
    )
    axis /= (2.0 * s)
    return axis * angle


def clip_arm_q(model, q):
    q = np.asarray(q, dtype=float).copy()
    for i in range(ARM_DOF):
        lo, hi = model.jnt_range[i]
        q[i] = np.clip(q[i], lo, hi)
    return q


def set_arm_q(simobj, q):
    simobj.data.qpos[:ARM_DOF] = q[:ARM_DOF]
    mujoco.mj_forward(simobj.model, simobj.data)


def ik_to_hand_pose(simobj, q_seed, target_pos, target_rot, iters=80):
    model = simobj.model
    data = simobj.data
    q = np.asarray(q_seed, dtype=float).copy()
    hand_bid = simobj.hand_body_id
    damping = 1e-3
    step = 0.35
    rot_weight = 0.4

    for _ in range(iters):
        set_arm_q(simobj, q)
        cur_pos = data.xpos[hand_bid].copy()
        cur_rot = mat_from_xmat(data.xmat[hand_bid])
        pos_err = target_pos - cur_pos
        rot_err = rotvec_from_matrix(target_rot @ cur_rot.T)
        if np.linalg.norm(pos_err) < 5e-4 and np.linalg.norm(rot_err) < 5e-3:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_bid)
        J = np.vstack([jacp[:, :ARM_DOF], rot_weight * jacr[:, :ARM_DOF]])
        e = np.concatenate([pos_err, rot_weight * rot_err])

        # Damped least squares solve for a stable pose update.
        A = J @ J.T + (damping ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(A, e)
        q = clip_arm_q(model, q + step * dq)

    set_arm_q(simobj, q)
    return q


def drive_to_q(simobj, q_start, q_goal, finger_ctrl, steps=120):
    q_start = np.asarray(q_start, dtype=float)
    q_goal = np.asarray(q_goal, dtype=float)
    for i in range(steps):
        t = (i + 1) / steps
        q_cmd = (1.0 - t) * q_start + t * q_goal
        simobj.data.ctrl[:ARM_DOF] = q_cmd[:ARM_DOF]
        simobj.data.ctrl[FINGER_CTRL] = finger_ctrl
        simobj.step(1)


def hold(simobj, q_cmd, finger_ctrl, steps=100):
    q_cmd = np.asarray(q_cmd, dtype=float)
    for _ in range(steps):
        simobj.data.ctrl[:ARM_DOF] = q_cmd[:ARM_DOF]
        simobj.data.ctrl[FINGER_CTRL] = finger_ctrl
        simobj.step(1)


def main():
    s = sim.Sim()
    s.hand_body_id = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    red = s.block_positions()["red"].copy()
    green = s.block_positions()["green"].copy()
    initial_rot = mat_from_xmat(s.data.xmat[s.hand_body_id])

    # First save a plausible replayable artifact immediately, then improve it.
    s.save_final_state("/work/final_state.npz")

    q = s.data.qpos[:ARM_DOF].copy()
    open_grip = 255.0
    closed_grip = 0.0
    hand_lift = 0.0584

    waypoints = [
        (np.array([red[0], red[1], 0.62]), open_grip),
        (np.array([red[0], red[1], 0.50]), open_grip),
        (np.array([red[0], red[1], red[2] + hand_lift]), open_grip),
        (np.array([red[0], red[1], red[2] + hand_lift]), closed_grip),
        (np.array([red[0], red[1], 0.66]), closed_grip),
        (np.array([green[0], green[1], 0.66]), closed_grip),
        (np.array([green[0], green[1], 0.55]), closed_grip),
        (np.array([green[0], green[1], green[2] + 0.05 + hand_lift]), closed_grip),
        (np.array([green[0], green[1], green[2] + 0.05 + hand_lift]), open_grip),
        (np.array([green[0], green[1], 0.70]), open_grip),
    ]

    for idx, (pos, grip) in enumerate(waypoints):
        q_goal = ik_to_hand_pose(s, q, pos, initial_rot, iters=90)
        # Longer dwell at grasp and release points.
        seg_steps = 160 if idx in (2, 3, 7, 8) else 120
        drive_to_q(s, q, q_goal, grip, steps=seg_steps)
        q = q_goal

        # Let the grasp settle before transitioning.
        if idx == 3:
            hold(s, q, closed_grip, steps=80)
        if idx == 8:
            hold(s, q, open_grip, steps=80)

    # Retract slightly and let the stack settle before saving.
    q_retract = ik_to_hand_pose(s, q, np.array([green[0] - 0.12, green[1] - 0.06, 0.72]), initial_rot, iters=80)
    drive_to_q(s, q, q_retract, open_grip, steps=150)
    q = q_retract
    hold(s, q, open_grip, steps=500)

    s.save_final_state("/work/final_state.npz")
    print("final red", s.block_positions()["red"])
    print("final green", s.block_positions()["green"])
    print("saved /work/final_state.npz")


if __name__ == "__main__":
    main()
