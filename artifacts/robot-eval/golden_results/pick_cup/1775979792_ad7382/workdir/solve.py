import os
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim, SUCCESS_CUP_Z


LEFT_PAD_GEOM = 69
RIGHT_PAD_GEOM = 77
SEED_Q = np.array([1.243, 1.09, -2.028, -1.624, 0.675, 1.09, -2.22], dtype=float)


@dataclass
class EvalResult:
    final_z: float
    contact_fraction: float
    max_z: float
    path: str


def quat_from_matrix(mat):
    q = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    return q


def orientation_error(current, target):
    q_conj = np.array([current[0], -current[1], -current[2], -current[3]])
    q_err = np.empty(4, dtype=float)
    mujoco.mju_mulQuat(q_err, target, q_conj)
    if q_err[0] < 0:
        q_err *= -1
    return 2.0 * q_err[1:]


def clamp_ctrl(sim, ctrl):
    lo = sim.model.actuator_ctrlrange[:, 0]
    hi = sim.model.actuator_ctrlrange[:, 1]
    return np.clip(ctrl, lo, hi)


def set_arm_q(sim, q):
    sim.data.qpos[:7] = q
    sim.data.qvel[:] = 0
    sim.data.ctrl[:7] = q
    mujoco.mj_forward(sim.model, sim.data)


def geom_pos_jac(sim, geom_id):
    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    mujoco.mj_jacGeom(sim.model, sim.data, jacp, jacr, geom_id)
    return jacp[:, :7]


def body_pose_jac(sim, body_id):
    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, body_id)
    return jacp[:, :7], jacr[:, :7]


def solve_pad_ik(sim, left_target, right_target, hand_quat=None, q_init=None, max_iters=200):
    if q_init is None:
        q = sim.data.qpos[:7].copy()
    else:
        q = q_init.copy()
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    for _ in range(max_iters):
        set_arm_q(sim, q)
        left = sim.data.geom_xpos[LEFT_PAD_GEOM].copy()
        right = sim.data.geom_xpos[RIGHT_PAD_GEOM].copy()
        err = np.concatenate([left_target - left, right_target - right])
        blocks = [geom_pos_jac(sim, LEFT_PAD_GEOM), geom_pos_jac(sim, RIGHT_PAD_GEOM)]
        if hand_quat is not None:
            hand_quat_curr = np.empty(4, dtype=float)
            mujoco.mju_mat2Quat(hand_quat_curr, sim.data.xmat[hand_id])
            oerr = orientation_error(hand_quat_curr, hand_quat)
            _, jacr = body_pose_jac(sim, hand_id)
            err = np.concatenate([err, 0.25 * oerr])
            blocks.append(0.25 * jacr)
        if np.linalg.norm(err) < 1e-4:
            break
        J = np.vstack(blocks)
        lam = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), err)
        q = np.clip(q + np.clip(dq, -0.08, 0.08), sim.model.actuator_ctrlrange[:7, 0], sim.model.actuator_ctrlrange[:7, 1])
    set_arm_q(sim, q)
    return q


def body_contacts_with_cup(sim):
    cup_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    left_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    finger = False
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        b1 = sim.model.geom_bodyid[c.geom1]
        b2 = sim.model.geom_bodyid[c.geom2]
        pair = {b1, b2}
        if cup_bid in pair and (left_bid in pair or right_bid in pair):
            finger = True
    return finger


def replay_eval(ctrl_trace):
    sim = Sim()
    for ctrl in ctrl_trace:
        sim.data.ctrl[:] = ctrl
        sim.step(1)
    max_z = sim.cup_position()[2]
    finger_hits = 0
    settle = 500
    hold_ctrl = ctrl_trace[-1] if len(ctrl_trace) else sim.data.ctrl.copy()
    for _ in range(settle):
        sim.data.ctrl[:] = hold_ctrl
        sim.step(1)
        max_z = max(max_z, sim.cup_position()[2])
        finger_hits += int(body_contacts_with_cup(sim))
    return EvalResult(
        final_z=float(sim.cup_position()[2]),
        contact_fraction=finger_hits / settle,
        max_z=float(max_z),
        path="",
    )


def move_ctrl(sim, q_target, grip, steps):
    q_start = sim.data.ctrl[:7].copy()
    g_start = sim.data.ctrl[7]
    for i in range(steps):
        a = (i + 1) / steps
        sim.data.ctrl[:7] = (1 - a) * q_start + a * q_target
        sim.data.ctrl[7] = (1 - a) * g_start + a * grip
        sim.step(1)


def settle(sim, steps):
    sim.step(steps)


def make_attempt(params, save_path="/work/final_state.npz", render_dir=None):
    sim = Sim()
    cup = sim.cup_position().copy()

    sim.data.ctrl[:7] = sim.data.qpos[:7]
    sim.data.ctrl[7] = 255.0
    sim.step(20)

    y_open = params["y_open"]
    y_close = params["y_close"]
    z_high = params["z_high"]
    z_grasp = params["z_grasp"]
    z_lift = params["z_lift"]
    x_bias = params["x_bias"]
    q_seed = params.get("q_seed", SEED_Q)

    pre_left = cup + np.array([x_bias, y_open, z_high - cup[2]])
    pre_right = cup + np.array([x_bias, -y_open, z_high - cup[2]])
    q_pre = solve_pad_ik(sim, pre_left, pre_right, q_init=q_seed)

    grasp_left = cup + np.array([x_bias, y_close, z_grasp - cup[2]])
    grasp_right = cup + np.array([x_bias, -y_close, z_grasp - cup[2]])
    q_grasp = solve_pad_ik(sim, grasp_left, grasp_right, q_init=q_pre)

    lift_left = cup + np.array([x_bias, y_close, z_lift - cup[2]])
    lift_right = cup + np.array([x_bias, -y_close, z_lift - cup[2]])
    q_lift = solve_pad_ik(sim, lift_left, lift_right, q_init=q_grasp)

    sim.reset()
    sim.data.ctrl[:7] = sim.data.qpos[:7]
    sim.data.ctrl[7] = 255.0
    sim.step(50)
    move_ctrl(sim, q_pre, 255.0, params["steps_pre"])
    settle(sim, params["hold_pre"])
    move_ctrl(sim, q_grasp, 255.0, params["steps_descend"])
    settle(sim, params["hold_descend"])
    move_ctrl(sim, q_grasp, 0.0, params["steps_close"])
    settle(sim, params["hold_close"])
    move_ctrl(sim, q_lift, 0.0, params["steps_lift"])
    settle(sim, params["hold_lift"])
    sim.save_final_state(save_path)

    if render_dir:
        os.makedirs(render_dir, exist_ok=True)
        from PIL import Image

        Image.fromarray(sim.render()).save(os.path.join(render_dir, "final.png"))

    data = np.load(save_path)
    result = replay_eval(data["ctrl_trace"])
    result.path = save_path
    return result


def main():
    params = {
        "q_seed": SEED_Q,
        "x_bias": 0.0,
        "y_open": 0.0455,
        "y_close": 0.0455,
        "z_high": 0.49,
        "z_grasp": 0.445,
        "z_lift": 0.60,
        "steps_pre": 200,
        "hold_pre": 50,
        "steps_descend": 120,
        "hold_descend": 30,
        "steps_close": 120,
        "hold_close": 80,
        "steps_lift": 200,
        "hold_lift": 200,
    }
    result = make_attempt(params)
    print(
        f"final_z={result.final_z:.3f} max_z={result.max_z:.3f} "
        f"contact={result.contact_fraction:.3f} pass="
        f"{result.final_z >= SUCCESS_CUP_Z and result.contact_fraction >= 0.5}"
    )


if __name__ == "__main__":
    main()
