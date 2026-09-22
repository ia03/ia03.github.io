import math
import os

import imageio.v2 as imageio
import mujoco
import numpy as np

import sim


ARM_JOINTS = list(range(7))
FINGER_ACT = 7


def rotmat_from_body(data, body_id):
    return np.array(data.xmat[body_id]).reshape(3, 3).copy()


def rot_error(R_cur, R_des):
    R_err = R_des @ R_cur.T
    return 0.5 * np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )


def rotz(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def solve_ik(model, q_seed, pos_target, rot_target, body_id, iters=50, damp=1e-3):
    data = mujoco.MjData(model)
    q = q_seed.copy()
    for _ in range(iters):
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        pos = np.array(data.xpos[body_id]).copy()
        R = rotmat_from_body(data, body_id)
        e_pos = pos_target - pos
        e_rot = rot_error(R, rot_target)
        err = np.concatenate([e_pos, 0.35 * e_rot])
        if np.linalg.norm(e_pos) < 1e-4 and np.linalg.norm(e_rot) < 2e-3:
            break
        Jp = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, Jp, Jr, body_id)
        J = np.vstack([Jp[:, :7], 0.35 * Jr[:, :7]])
        A = J @ J.T + damp * np.eye(6)
        dq = J.T @ np.linalg.solve(A, err)
        q[:7] += dq
        for j in range(7):
            jid = j
            lo = model.jnt_range[jid, 0]
            hi = model.jnt_range[jid, 1]
            q[j] = np.clip(q[j], lo, hi)
    return q


def best_ik(model, q_seed, pos_target, body_id, base_R, yaws):
    best_q = None
    best_err = None
    for yaw in yaws:
        q = solve_ik(model, q_seed, pos_target, rotz(yaw) @ base_R, body_id, iters=80)
        data = mujoco.MjData(model)
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        err = np.linalg.norm(np.array(data.xpos[body_id]) - pos_target)
        if best_err is None or err < best_err:
            best_err = err
            best_q = q
    return best_q


def run_segment(s, q_start, q_goal, finger_goal, steps):
    for t in range(steps):
        a = (t + 1) / steps
        q_cmd = (1 - a) * q_start + a * q_goal
        s.data.ctrl[:7] = q_cmd[:7]
        s.data.ctrl[FINGER_ACT] = finger_goal
        s.step(1)


def main():
    s = sim.Sim()
    hand_id = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    init_q = s.data.qpos.copy()
    init_R = rotmat_from_body(s.data, hand_id)
    cup = np.array(sim.CUP_INIT_POS)
    bin_center = np.array(sim.BIN_CENTER)

    above_cup = cup + np.array([0.0, 0.0, 0.18])
    grasp = cup + np.array([0.0, 0.0, -0.01])
    lift = cup + np.array([0.0, 0.0, 0.20])
    above_bin = bin_center + np.array([0.0, 0.0, 0.17])
    drop = bin_center + np.array([0.0, 0.0, 0.17])
    retreat = np.array([0.30, -0.30, 0.85])

    q_above_cup = best_ik(s.model, init_q, above_cup, hand_id, init_R, np.linspace(-0.7, -0.2, 11))
    q_grasp = best_ik(s.model, q_above_cup, grasp, hand_id, init_R, np.linspace(-0.7, -0.2, 11))
    q_lift = best_ik(s.model, q_grasp, lift, hand_id, init_R, np.linspace(-0.7, -0.2, 11))
    q_above_bin = best_ik(s.model, q_lift, above_bin, hand_id, init_R, np.linspace(-0.3, 0.3, 7))
    q_drop = best_ik(s.model, q_above_bin, drop, hand_id, init_R, np.linspace(-0.3, 0.3, 7))
    q_retreat = best_ik(s.model, q_drop, retreat, hand_id, init_R, np.linspace(-0.7, 0.3, 11))

    # Open, approach, close, lift, carry, release, retreat, settle.
    s.data.ctrl[:7] = init_q[:7]
    s.data.ctrl[FINGER_ACT] = 255.0
    s.step(10)

    run_segment(s, init_q, q_above_cup, 255.0, 120)
    run_segment(s, q_above_cup, q_grasp, 255.0, 100)
    run_segment(s, q_grasp, q_grasp, 0.0, 140)
    run_segment(s, q_grasp, q_lift, 0.0, 120)
    run_segment(s, q_lift, q_above_bin, 0.0, 170)
    run_segment(s, q_above_bin, q_drop, 0.0, 80)
    run_segment(s, q_drop, q_drop, 255.0, 90)
    run_segment(s, q_drop, q_retreat, 255.0, 120)
    run_segment(s, q_retreat, q_retreat, 255.0, 500)

    # Save a quick snapshot for debugging.
    imageio.imwrite("/work/final_frame.png", s.render(640, 480))
    s.save_final_state("/work/final_state.npz")

    trace = np.array(s._trace, dtype=object)
    print("time", s.data.time)
    print("cup", s.cup_position())
    print("contact", s.has_gripper_cup_contact())
    print("saved", os.path.getsize("/work/final_state.npz"))


if __name__ == "__main__":
    main()
