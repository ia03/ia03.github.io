import time

import mujoco
import numpy as np

import sim


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=float)


def now_utc():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def joint_indices(model, names):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]


def joint_qpos_adr(model, joint_ids):
    return [model.jnt_qposadr[j] for j in joint_ids]


def joint_dof_adr(model, joint_ids):
    return [model.jnt_dofadr[j] for j in joint_ids]


def set_arm_qpos(data, model, q_arm):
    for idx, q in zip(ARM_QPOS_ADR, q_arm):
        data.qpos[idx] = q


def get_arm_qpos(data):
    return np.array([data.qpos[idx] for idx in ARM_QPOS_ADR], dtype=float)


def clip_arm_qpos(model, q):
    out = q.copy()
    for i, j in enumerate(ARM_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = model.jnt_range[jid]
        out[i] = np.clip(out[i], lo + 1e-3, hi - 1e-3)
    return out


def solve_ik_to_pos(sim_obj, target_pos, q_seed=None, max_iters=100, tol=2e-3, damping=1e-2):
    model = sim_obj.model
    data = sim_obj.data
    if q_seed is None:
        q = get_arm_qpos(data)
    else:
        q = np.array(q_seed, dtype=float).copy()
    for _ in range(max_iters):
        set_arm_qpos(data, model, q)
        mujoco.mj_forward(model, data)
        pos = data.xpos[sim_obj.hand_id].copy()
        err = target_pos - pos
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, sim_obj.hand_id)
        cols = ARM_DOF_ADR
        J = jacp[:, cols]
        dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(3), err)
        step_norm = np.linalg.norm(dq)
        if step_norm > 0.18:
            dq *= 0.18 / step_norm
        q = clip_arm_qpos(model, q + dq)
    set_arm_qpos(data, model, q)
    mujoco.mj_forward(model, data)
    return q


def step_ctrl(sim_obj, ctrl, steps, settle=False):
    ctrl = np.asarray(ctrl, dtype=float).copy()
    if settle:
        ctrl[:7] = ctrl[:7]
    sim_obj.data.ctrl[:] = ctrl
    sim_obj.step(steps)


def open_gripper_ctrl():
    return np.array([0, 0, 0, 0, 0, 0, 0, 255], dtype=float)


def close_gripper_ctrl():
    return np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)


def arm_ctrl_from_q(q_arm, finger=255):
    ctrl = np.zeros(8, dtype=float)
    ctrl[:7] = q_arm
    ctrl[7] = finger
    return ctrl


def run_policy():
    s = sim.Sim()
    s.hand_id = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    print("start", now_utc())
    print("initial blocks", s.block_positions())

    # Move to the built-in home pose before attempting the grasp.
    s.data.ctrl[:] = arm_ctrl_from_q(HOME_Q, 255)
    s.step(250)
    s.data.ctrl[:] = arm_ctrl_from_q(HOME_Q, 255)
    s.step(100)

    red = s.block_positions()["red"]
    green = s.block_positions()["green"]

    # Approach from above with a modest offset.
    q = solve_ik_to_pos(s, np.array([red[0], red[1], 0.64]), q_seed=HOME_Q)
    step_ctrl(s, arm_ctrl_from_q(q, 255), 160)

    q = solve_ik_to_pos(s, np.array([red[0], red[1], 0.50]), q_seed=get_arm_qpos(s.data))
    step_ctrl(s, arm_ctrl_from_q(q, 255), 160)

    # Close slowly to capture the red block.
    for finger in [220, 180, 140, 100, 60, 20, 0]:
        step_ctrl(s, arm_ctrl_from_q(get_arm_qpos(s.data), finger), 40)

    # Lift and check the grasp.
    q = solve_ik_to_pos(s, np.array([red[0], red[1], 0.66]), q_seed=get_arm_qpos(s.data))
    step_ctrl(s, arm_ctrl_from_q(q, 0), 220)

    # Move above the green block with the object in hand.
    q = solve_ik_to_pos(s, np.array([green[0], green[1], 0.66]), q_seed=get_arm_qpos(s.data))
    step_ctrl(s, arm_ctrl_from_q(q, 0), 220)

    # Descend for placement.
    q = solve_ik_to_pos(s, np.array([green[0], green[1], 0.535]), q_seed=get_arm_qpos(s.data))
    step_ctrl(s, arm_ctrl_from_q(q, 0), 160)

    # Release and settle for a while.
    step_ctrl(s, arm_ctrl_from_q(get_arm_qpos(s.data), 255), 80)
    q = solve_ik_to_pos(s, np.array([green[0] - 0.12, green[1], 0.75]), q_seed=get_arm_qpos(s.data))
    step_ctrl(s, arm_ctrl_from_q(q, 255), 260)
    s.data.ctrl[:] = arm_ctrl_from_q(get_arm_qpos(s.data), 255)
    s.step(500)

    print("final blocks", s.block_positions())
    print("time", s.data.time)
    s.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")
    return s


if __name__ == "__main__":
    model_tmp = sim.Sim().model
    ARM_QPOS_ADR = joint_qpos_adr(model_tmp, joint_indices(model_tmp, ARM_JOINTS))
    ARM_DOF_ADR = joint_dof_adr(model_tmp, joint_indices(model_tmp, ARM_JOINTS))
    run_policy()
