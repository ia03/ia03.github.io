import math
import os
import time

import numpy as np
import mujoco

import sim


HAND_BODY = "hand"
ARM_DOF_QPOS = np.arange(7)
ARM_DOF_QVEL = np.arange(7)
FINGER_CTRL = 7
DRAWER_CTRL = 8


def rot_err(current_R, target_R):
    # Small-angle orientation error in world coordinates.
    return 0.5 * (
        np.cross(current_R[:, 0], target_R[:, 0])
        + np.cross(current_R[:, 1], target_R[:, 1])
        + np.cross(current_R[:, 2], target_R[:, 2])
    )


def ik_target_step(s, target_pos, target_R, finger_ctrl, drawer_ctrl, step_gain=1.0, rot_weight=0.0):
    model = s.model
    data = s.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
    if rot_weight > 0.0:
        J = np.vstack([jacp[:, ARM_DOF_QVEL], rot_weight * jacr[:, ARM_DOF_QVEL]])
    else:
        J = jacp[:, ARM_DOF_QVEL]

    current_pos = data.xpos[hand_id].copy()
    current_R = data.xmat[hand_id].reshape(3, 3).copy()
    if rot_weight > 0.0:
        e = np.concatenate([target_pos - current_pos, rot_weight * rot_err(current_R, target_R)])
    else:
        e = target_pos - current_pos

    # Damped least-squares with a conservative step to keep motion stable.
    lam = 1e-3
    A = J @ J.T + lam * np.eye(J.shape[0])
    dq = J.T @ np.linalg.solve(A, e)
    dq = np.clip(dq, -0.35, 0.35)

    arm_q = data.qpos[ARM_DOF_QPOS].copy()
    target_q = arm_q + step_gain * dq

    for i in range(7):
        lo, hi = model.actuator_ctrlrange[i]
        target_q[i] = np.clip(target_q[i], lo, hi)

    data.ctrl[:7] = target_q
    data.ctrl[FINGER_CTRL] = finger_ctrl
    data.ctrl[DRAWER_CTRL] = drawer_ctrl


def hold_pose(s, arm_q, finger_ctrl, drawer_ctrl):
    s.data.ctrl[:7] = arm_q
    s.data.ctrl[FINGER_CTRL] = finger_ctrl
    s.data.ctrl[DRAWER_CTRL] = drawer_ctrl


def run_stage(
    s,
    n_steps,
    target_pos=None,
    target_R=None,
    finger_ctrl=0.0,
    drawer_ctrl=1.0,
    log_every=100,
    label="",
    target_pos_fn=None,
    rot_weight=0.08,
):
    for t in range(n_steps):
        if target_pos_fn is not None:
            ik_target_step(
                s,
                target_pos_fn(s),
                target_R,
                finger_ctrl,
                drawer_ctrl,
                rot_weight=rot_weight,
            )
        elif target_pos is not None:
            ik_target_step(
                s,
                target_pos,
                target_R,
                finger_ctrl,
                drawer_ctrl,
                rot_weight=rot_weight,
            )
        else:
            hold_pose(s, s.data.qpos[:7].copy(), finger_ctrl, drawer_ctrl)
        s.step(1)
        if log_every and (t % log_every == 0 or t == n_steps - 1):
            print(
                f"{label} step={t+1}/{n_steps} time={s.data.time:.2f} "
                f"drawer={s.drawer_open_amount():.3f} block_z={s.block_position()[2]:.3f} "
                f"contact={s.has_gripper_block_contact()}"
            )


def main():
    s = sim.Sim()
    hand_id = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    target_R = s.data.xmat[hand_id].reshape(3, 3).copy()
    print("start utc", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    print("initial block", s.block_position())
    print("initial hand", s.data.xpos[hand_id])

    # Stage 1: open the drawer.
    s.data.ctrl[:] = 0.0
    s.data.ctrl[DRAWER_CTRL] = 1.0
    s.data.ctrl[FINGER_CTRL] = 255.0
    s.step(300)
    print("drawer open after stage1", s.drawer_open_amount())
    s.save_final_state("/work/final_state.npz")
    print("saved early state")

    # Stage 2: move above the live block position with the gripper open.
    pregrasp_fn = lambda simobj: simobj.block_position() + np.array([0.0, 0.0, 0.18])
    run_stage(
        s,
        260,
        target_R=target_R,
        finger_ctrl=255.0,
        drawer_ctrl=1.0,
        label="pregrasp",
        target_pos_fn=pregrasp_fn,
        rot_weight=0.0,
    )

    # Stage 3: descend onto the block.
    grasp_fn = lambda simobj: simobj.block_position() + np.array([0.0, 0.0, -0.02])
    run_stage(
        s,
        220,
        target_R=target_R,
        finger_ctrl=255.0,
        drawer_ctrl=1.0,
        label="descend",
        target_pos_fn=grasp_fn,
        rot_weight=0.0,
    )

    # Stage 4: close the gripper around the block.
    run_stage(
        s,
        180,
        target_R=target_R,
        finger_ctrl=0.0,
        drawer_ctrl=1.0,
        label="close",
        target_pos_fn=grasp_fn,
        rot_weight=0.0,
    )

    # Stage 5: lift while holding the drawer open.
    lift_fn = lambda simobj: simobj.block_position() + np.array([0.0, 0.0, 0.25])
    run_stage(
        s,
        300,
        target_R=target_R,
        finger_ctrl=0.0,
        drawer_ctrl=1.0,
        label="lift",
        target_pos_fn=lift_fn,
        rot_weight=0.0,
    )

    # Stage 6: stabilize the held block before saving.
    run_stage(
        s,
        160,
        target_R=target_R,
        finger_ctrl=0.0,
        drawer_ctrl=1.0,
        label="stabilize",
        target_pos_fn=lift_fn,
        rot_weight=0.0,
    )

    print("final drawer", s.drawer_open_amount())
    print("final block", s.block_position())
    print("final contact", s.has_gripper_block_contact())
    s.save_final_state("/work/final_state.npz")
    print("saved final state")


if __name__ == "__main__":
    main()
