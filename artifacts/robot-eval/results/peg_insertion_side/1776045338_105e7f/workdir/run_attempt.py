import math
import os

import imageio.v2 as iio
import mujoco
import numpy as np

from sim import Sim


HAND_BODY = "hand"
PEG_BODY = "peg"


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def mat_from_body(data, bid):
    return data.xmat[bid].reshape(3, 3).copy()


def pose_error(current_mat, target_mat):
    rot = current_mat.T @ target_mat
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.reshape(-1))
    err = np.zeros(3)
    mujoco.mju_quat2Vel(err, quat, 1.0)
    return current_mat @ err


def solve_ik(sim, target_pos, target_mat=None, steps=120, pos_gain=4.0, rot_gain=2.0):
    model, data = sim.model, sim.data
    hand_bid = body_id(model, HAND_BODY)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    eye = np.eye(7)
    for _ in range(steps):
        mujoco.mj_forward(model, data)
        pos = data.xpos[hand_bid].copy()
        pos_err = target_pos - pos
        if target_mat is None:
            err = pos_gain * pos_err
            mujoco.mj_jacBody(model, data, jacp, None, hand_bid)
            J = jacp[:, :7]
        else:
            cur_mat = mat_from_body(data, hand_bid)
            rot_err = pose_error(cur_mat, target_mat)
            err = np.concatenate([pos_gain * pos_err, rot_gain * rot_err])
            mujoco.mj_jacBody(model, data, jacp, jacr, hand_bid)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(J.shape[0]), err)
        q = data.qpos[:7].copy()
        q += np.clip(dq, -0.08, 0.08)
        q = np.clip(q, model.actuator_ctrlrange[:7, 0], model.actuator_ctrlrange[:7, 1])
        data.qpos[:7] = q
        data.qvel[:7] = 0
    mujoco.mj_forward(model, data)
    return data.qpos[:7].copy()


def run_hold(sim, q_arm, grip, steps, save_every=None):
    sim.data.ctrl[:7] = q_arm
    sim.data.ctrl[7] = grip
    frames = []
    for i in range(steps):
        sim.data.ctrl[:7] = q_arm
        sim.data.ctrl[7] = grip
        sim.step()
        if save_every is not None and i % save_every == 0:
            frames.append(sim.render(640, 480))
    return frames


def main():
    sim = Sim()
    model, data = sim.model, sim.data
    hand_bid = body_id(model, HAND_BODY)
    peg_bid = body_id(model, PEG_BODY)

    peg0 = data.xpos[peg_bid].copy()
    waypoints = [
        (peg0 + np.array([-0.06, 0.00, 0.16]), 0.0, 220),
        (peg0 + np.array([-0.06, 0.00, 0.10]), 0.0, 180),
        (peg0 + np.array([-0.055, 0.00, 0.055]), 0.0, 220),
        (peg0 + np.array([-0.050, 0.00, 0.030]), 0.0, 220),
    ]

    frames = [sim.render(640, 480)]
    best_x = sim.peg_position()[0]
    best = None

    for pos, grip, settle in waypoints:
        q = solve_ik(sim, pos, target_mat=None, steps=260)
        frames += run_hold(sim, q, grip, settle, save_every=20)
        peg = sim.peg_position().copy()
        if peg[0] > best_x:
            best_x = peg[0]
            best = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), list(sim._ctrl_trace))

    sim.save_final_state("/work/final_state.npz")

    push_offsets = [
        np.array([-0.045, 0.000, 0.030]),
        np.array([-0.035, 0.000, 0.030]),
        np.array([-0.025, 0.000, 0.028]),
        np.array([-0.015, 0.000, 0.026]),
        np.array([-0.005, 0.000, 0.024]),
        np.array([0.005, 0.000, 0.022]),
        np.array([0.015, 0.000, 0.020]),
        np.array([0.025, 0.000, 0.018]),
        np.array([0.035, 0.000, 0.016]),
        np.array([0.045, 0.000, 0.014]),
        np.array([0.055, 0.000, 0.012]),
    ]
    for offset in push_offsets:
        peg = data.xpos[peg_bid].copy()
        q = solve_ik(sim, peg + offset, target_mat=None, steps=180)
        frames += run_hold(sim, q, 0.0, 120, save_every=15)
        peg = sim.peg_position().copy()
        if peg[0] > best_x:
            best_x = peg[0]
            sim.save_final_state("/work/final_state.npz")

    q = solve_ik(sim, sim.peg_position() + np.array([0.03, 0.0, 0.03]), target_mat=None, steps=120)
    frames += run_hold(sim, q, 0.0, 260, save_every=20)
    sim.save_final_state("/work/final_state.npz")

    os.makedirs("/work/frames", exist_ok=True)
    for idx, frame in enumerate(frames):
        iio.imwrite(f"/work/frames/frame_{idx:03d}.png", frame)
    iio.mimwrite("/work/attempt.mp4", frames, fps=20)

    peg = sim.peg_position()
    peg_mat = mat_from_body(data, peg_bid)
    print("final peg pos", peg)
    print("alignment", peg_mat[0, 0])
    print("hand pos", data.xpos[hand_bid])
    print("saved", os.path.exists("/work/final_state.npz"))


if __name__ == "__main__":
    main()
