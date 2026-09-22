import math
import numpy as np
import mujoco

import sim


ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_DOF_ADR = None
ARM_QPOS_ADR = None


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def joint_ids(model):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]


def setup_arm_indices(model):
    jids = joint_ids(model)
    qpos = [model.jnt_qposadr[j] for j in jids]
    dof = [model.jnt_dofadr[j] for j in jids]
    return np.array(qpos, dtype=int), np.array(dof, dtype=int)


def ik_pose(model, data, target_pos, q_rest, hand_bid, arm_dofs, iters=8):
    """Damped least-squares IK on hand position, with a soft posture prior."""
    q = data.qpos.copy()
    lam = 1e-3
    pos = np.asarray(target_pos, dtype=float)
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        pos_err = pos - data.xpos[hand_bid]
        if np.linalg.norm(pos_err) < 1e-3:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_bid)
        J = jacp[:, arm_dofs]
        A = J @ J.T + lam * np.eye(3)
        err = pos_err
        dq = J.T @ np.linalg.solve(A, err)
        dq += 0.2 * (q_rest - q[arm_qpos_adr])
        step = np.clip(dq, -0.05, 0.05)
        q[arm_qpos_adr] = q[arm_qpos_adr] + step
        for i, jid in enumerate(arm_joint_ids):
            lo, hi = model.jnt_range[jid]
            q[arm_qpos_adr[i]] = np.clip(q[arm_qpos_adr[i]], lo, hi)
        data.qpos[:] = q
    mujoco.mj_forward(model, data)
    return data.qpos[arm_qpos_adr].copy()


def interp(a, b, t):
    return (1.0 - t) * np.asarray(a, float) + t * np.asarray(b, float)


def run():
    s = sim.Sim()
    model = s.model
    data = s.data

    global arm_joint_ids, arm_qpos_adr, ARM_DOF_ADR
    arm_joint_ids = joint_ids(model)
    arm_qpos_adr, ARM_DOF_ADR = setup_arm_indices(model)

    hand_bid = body_id(model, "hand")
    q_rest = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])
    q_rest = np.clip(q_rest, [model.jnt_range[j][0] for j in arm_joint_ids], [model.jnt_range[j][1] for j in arm_joint_ids])

    phases = [
        (160, np.array([0.49, -0.10, 0.67]), 255),  # hover above the cup
        (100, np.array([0.50, -0.10, 0.59]), 255),  # descend to grasp height
        (90, np.array([0.50, -0.10, 0.59]), 0),     # close
        (150, np.array([0.50, -0.10, 0.72]), 0),    # lift straight up
        (220, np.array([0.70, 0.14, 0.72]), 0),     # move above bin
        (130, np.array([0.70, 0.14, 0.54]), 0),     # lower into bin
        (100, np.array([0.70, 0.14, 0.54]), 255),   # release
        (140, np.array([0.62, 0.04, 0.78]), 255),   # retreat away
        (120, np.array([0.62, 0.04, 0.78]), 255),   # hold
    ]

    # Build a smooth target trajectory by linearly interpolating within each phase.
    control_steps = 0
    for duration, target, grip in phases:
        start = data.xpos[hand_bid].copy()
        for i in range(duration):
            t = (i + 1) / duration
            tgt = interp(start, target, t)
            q_des = ik_pose(model, data, tgt, q_rest, hand_bid, ARM_DOF_ADR, iters=10)
            data.ctrl[:7] = q_des
            data.ctrl[7] = grip
            s.step(1)
            control_steps += 1

    # Keep the final parked pose for a short settle-like control segment.
    for _ in range(40):
        data.ctrl[:7] = data.qpos[:7]
        data.ctrl[7] = 255
        s.step(1)
        control_steps += 1

    print("steps", control_steps)
    print("cup pos", s.cup_position())
    print("contact", s.has_gripper_cup_contact())
    print("time", float(data.time))
    s.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    run()
