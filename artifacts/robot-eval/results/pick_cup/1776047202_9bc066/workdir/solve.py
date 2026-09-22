import numpy as np
import mujoco

from sim import Sim, SUCCESS_CUP_Z


def rot_err(R_current: np.ndarray, R_desired: np.ndarray) -> np.ndarray:
    # Orientation error used by many operational-space IK controllers.
    return 0.5 * (
        np.cross(R_current[:, 0], R_desired[:, 0])
        + np.cross(R_current[:, 1], R_desired[:, 1])
        + np.cross(R_current[:, 2], R_desired[:, 2])
    )


def damped_ls(J: np.ndarray, e: np.ndarray, damping: float) -> np.ndarray:
    # dq = J^T (J J^T + l^2 I)^-1 e
    JJt = J @ J.T
    return J.T @ np.linalg.solve(JJt + (damping**2) * np.eye(JJt.shape[0]), e)


def drive_hand_to(
    sim: Sim,
    hand_id: int,
    pos_w: np.ndarray,
    R_w: np.ndarray,
    steps: int,
    pos_gain: float = 2.5,
    ori_gain: float = 1.5,
    damping: float = 0.08,
    dq_step: float = 0.12,
):
    m, d = sim.model, sim.data
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))

    for _ in range(steps):
        mujoco.mj_jacBody(m, d, jacp, jacr, hand_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])

        p_cur = d.xpos[hand_id].copy()
        R_cur = d.xmat[hand_id].reshape(3, 3).copy()

        e_pos = pos_gain * (pos_w - p_cur)
        e_ori = ori_gain * rot_err(R_cur, R_w)
        e = np.concatenate([e_pos, e_ori])

        dq = damped_ls(J, e, damping=damping)
        dq = np.clip(dq, -dq_step, dq_step)
        q_des = d.qpos[:7] + dq

        # Clip to actuator ctrlrange (joint limits).
        lo, hi = m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1]
        sim.data.ctrl[:7] = np.clip(q_des, lo, hi)
        sim.step(1)


def drive_hand_pose_to(
    sim: Sim,
    hand_id: int,
    pos_w: np.ndarray,
    R_w: np.ndarray,
    steps: int,
    pos_gain: float = 2.2,
    ori_gain: float = 2.0,
    damping: float = 0.07,
    dq_step: float = 0.10,
    q_rest: np.ndarray | None = None,
    null_gain: float = 0.08,
):
    m, d = sim.model, sim.data
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    if q_rest is None:
        q_rest = d.qpos[:7].copy()

    for _ in range(steps):
        mujoco.mj_jacBody(m, d, jacp, jacr, hand_id)
        Jp = jacp[:, :7]
        Jr = jacr[:, :7]
        J = np.vstack([Jp, Jr])

        p_cur = d.xpos[hand_id].copy()
        R_cur = d.xmat[hand_id].reshape(3, 3).copy()

        e_pos = pos_gain * (pos_w - p_cur)
        e_ori = ori_gain * rot_err(R_cur, R_w)
        e = np.concatenate([e_pos, e_ori])

        JJt = J @ J.T
        inv = np.linalg.solve(JJt + (damping**2) * np.eye(6), np.eye(6))
        J_pinv = J.T @ inv

        dq = J_pinv @ e
        dq += (np.eye(7) - J_pinv @ J) @ (null_gain * (q_rest - d.qpos[:7]))

        dq = np.clip(dq, -dq_step, dq_step)
        q_des = d.qpos[:7] + dq

        lo, hi = m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1]
        d.ctrl[:7] = np.clip(q_des, lo, hi)
        sim.step(1)


def drive_finger_midpoint_to(
    sim: Sim,
    left_finger_id: int,
    right_finger_id: int,
    target_mid_pos_w: np.ndarray,
    steps: int,
    pos_gain: float = 2.2,
    damping: float = 0.06,
    dq_step: float = 0.10,
    q_rest: np.ndarray | None = None,
    null_gain: float = 0.15,
):
    m, d = sim.model, sim.data
    jacp_l = np.zeros((3, m.nv))
    jacr_l = np.zeros((3, m.nv))
    jacp_r = np.zeros((3, m.nv))
    jacr_r = np.zeros((3, m.nv))

    if q_rest is None:
        q_rest = d.qpos[:7].copy()

    for _ in range(steps):
        mujoco.mj_jacBody(m, d, jacp_l, jacr_l, left_finger_id)
        mujoco.mj_jacBody(m, d, jacp_r, jacr_r, right_finger_id)

        J = 0.5 * (jacp_l[:, :7] + jacp_r[:, :7])

        p_l = d.xpos[left_finger_id].copy()
        p_r = d.xpos[right_finger_id].copy()
        p_mid = 0.5 * (p_l + p_r)

        e = pos_gain * (target_mid_pos_w - p_mid)

        JJt = J @ J.T
        inv = np.linalg.solve(JJt + (damping**2) * np.eye(3), np.eye(3))
        J_pinv = J.T @ inv

        dq_task = J_pinv @ e
        dq = dq_task

        # Gentle nullspace pull toward a reasonable configuration.
        I = np.eye(7)
        dq += (I - J_pinv @ J) @ (null_gain * (q_rest - d.qpos[:7]))

        dq = np.clip(dq, -dq_step, dq_step)
        q_des = d.qpos[:7] + dq

        lo, hi = m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1]
        d.ctrl[:7] = np.clip(q_des, lo, hi)
        sim.step(1)


def fingertip_geom_ids(m: mujoco.MjModel, finger_body_id: int) -> list[int]:
    geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == finger_body_id]
    if not geoms:
        raise RuntimeError(f"no geoms found for body_id={finger_body_id}")

    zmax = max(float(m.geom_pos[i][2]) for i in geoms)
    near = [i for i in geoms if abs(float(m.geom_pos[i][2]) - zmax) < 1e-6]
    if len(near) <= 4:
        return near

    # If there are many at the fingertip, pick a few spread in local-x.
    near.sort(key=lambda i: abs(float(m.geom_pos[i][0])), reverse=True)
    return near[:4]


def drive_grasp_center_to(
    sim: Sim,
    left_tip_geoms: list[int],
    right_tip_geoms: list[int],
    target_pos_w: np.ndarray,
    steps: int,
    pos_gain: float = 2.0,
    damping: float = 0.05,
    dq_step: float = 0.08,
    q_rest: np.ndarray | None = None,
    null_gain: float = 0.12,
):
    m, d = sim.model, sim.data
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))

    if q_rest is None:
        q_rest = d.qpos[:7].copy()

    tip_geoms = list(left_tip_geoms) + list(right_tip_geoms)
    n = len(tip_geoms)

    for _ in range(steps):
        p = np.zeros(3)
        J = np.zeros((3, 7))
        for gid in tip_geoms:
            mujoco.mj_jacGeom(m, d, jacp, jacr, gid)
            p += d.geom_xpos[gid]
            J += jacp[:, :7]
        p /= n
        J /= n

        e = pos_gain * (target_pos_w - p)

        JJt = J @ J.T
        inv = np.linalg.solve(JJt + (damping**2) * np.eye(3), np.eye(3))
        J_pinv = J.T @ inv

        dq = J_pinv @ e
        dq += (np.eye(7) - J_pinv @ J) @ (null_gain * (q_rest - d.qpos[:7]))

        dq = np.clip(dq, -dq_step, dq_step)
        q_des = d.qpos[:7] + dq
        lo, hi = m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1]
        d.ctrl[:7] = np.clip(q_des, lo, hi)
        sim.step(1)


def main(save_path: str = "/work/final_state.npz"):
    sim = Sim()
    m, d = sim.model, sim.data

    hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_finger_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    left_tip = fingertip_geom_ids(m, left_finger_id)
    right_tip = fingertip_geom_ids(m, right_finger_id)

    # Keep a constant "palm down" orientation based on the canonical start.
    # Right-handed frame with z pointing down.
    R_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)

    # Move to a pregrasp configuration that puts the gripper near the cup,
    # then do small Cartesian refinement with differential IK.
    q_pre = np.array([0.60, -0.02, -0.29, -1.87, -1.03, 1.22, 1.90], dtype=float)
    d.ctrl[:7] = q_pre
    d.ctrl[7] = 255.0
    sim.step(900)

    cup0 = sim.cup_position()

    # Calibrate the world-space offset from hand origin to fingertip grasp center.
    p_tip = np.zeros(3)
    for gid in list(left_tip) + list(right_tip):
        p_tip += d.geom_xpos[gid]
    p_tip /= (len(left_tip) + len(right_tip))
    hand_p = d.xpos[hand_id].copy()
    delta = p_tip - hand_p

    # Stage 1: go above the cup with a stable palm-down orientation.
    grasp_center_above = cup0 + np.array([0.0, 0.0, 0.16])
    hand_above = grasp_center_above - delta
    drive_hand_pose_to(sim, hand_id, hand_above, R_down, steps=1300, q_rest=q_pre, dq_step=0.10)

    # Stage 2: descend to grasp height.
    grasp_center = cup0 + np.array([0.0, 0.0, 0.02])
    hand_grasp = grasp_center - delta
    drive_hand_pose_to(sim, hand_id, hand_grasp, R_down, steps=1500, q_rest=q_pre, dq_step=0.06)

    # Stage 3: close gripper and squeeze briefly.
    d.ctrl[7] = 0.0
    drive_hand_pose_to(sim, hand_id, hand_grasp, R_down, steps=900, q_rest=q_pre, dq_step=0.04)

    # Stage 4: lift.
    lift_center = cup0 + np.array([0.0, 0.0, 0.27])
    hand_lift = lift_center - delta
    drive_hand_pose_to(sim, hand_id, hand_lift, R_down, steps=1800, q_rest=q_pre, dq_step=0.08)

    # Stage 5: hold.
    drive_hand_pose_to(sim, hand_id, hand_lift, R_down, steps=900, q_rest=q_pre, dq_step=0.03, null_gain=0.05)

    sim.save_final_state(save_path)
    print("saved", save_path)
    print("final cup pos", sim.cup_position())
    print("final finger qpos", d.qpos[7:9].copy())


if __name__ == "__main__":
    main()
