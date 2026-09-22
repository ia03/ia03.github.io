import numpy as np
import mujoco

from sim import Sim, BIN_CENTER


def _clip_to_ctrlrange(sim: Sim, q: np.ndarray) -> np.ndarray:
    q = q.copy()
    for i in range(7):
        if sim.model.actuator_ctrllimited[i]:
            lo, hi = sim.model.actuator_ctrlrange[i]
            q[i] = float(np.clip(q[i], lo, hi))
    return q


def ik_step_hand(
    sim: Sim,
    target_pos: np.ndarray,
    *,
    target_z_axis: np.ndarray | None = None,
    kp_pos: float = 8.0,
    kp_ori: float = 0.0,
    damping: float = 2e-2,
    max_dq: float = 0.08,
) -> np.ndarray:
    """One damped least-squares IK update for the Panda arm joints (7 dof)."""
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cur_pos = sim.data.xpos[hand_id].copy()
    pos_err = (target_pos - cur_pos)

    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, hand_id)
    Jp = jacp[:, :7]

    if kp_ori > 0.0 and target_z_axis is not None:
        R = sim.data.xmat[hand_id].reshape(3, 3)
        z_cur = R[:, 2].copy()
        ori_err = np.cross(z_cur, target_z_axis)
        Jr = jacr[:, :7]
        J = np.vstack([Jp, Jr])  # 6x7
        v = np.concatenate([kp_pos * pos_err, kp_ori * ori_err], axis=0)
        A = J @ J.T + (damping * damping) * np.eye(6)
        dq = J.T @ np.linalg.solve(A, v)
    else:
        J = Jp  # 3x7
        v = kp_pos * pos_err
        A = J @ J.T + (damping * damping) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, v)
    dq = np.clip(dq, -max_dq, max_dq)

    q = sim.data.qpos[:7].copy()
    q = q + dq
    return _clip_to_ctrlrange(sim, q)


def finger_tip_geoms(sim: Sim) -> tuple[int, int]:
    """Heuristically pick a 'tip' geom per finger (farthest from hand)."""
    m = sim.model
    hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    left_geoms = [gid for gid in range(m.ngeom) if int(m.geom_bodyid[gid]) == left_body]
    right_geoms = [gid for gid in range(m.ngeom) if int(m.geom_bodyid[gid]) == right_body]
    if not left_geoms or not right_geoms:
        raise RuntimeError("Could not find finger geoms")

    def pick_far(geoms: list[int]) -> int:
        best = geoms[0]
        best_d = -1.0
        hand_pos = sim.data.xpos[hand_id]
        for gid in geoms:
            d = float(np.linalg.norm(sim.data.geom_xpos[gid] - hand_pos))
            if d > best_d:
                best_d = d
                best = gid
        return best

    return pick_far(left_geoms), pick_far(right_geoms)


def ik_step_grasp_midpoint(
    sim: Sim,
    left_tip_geom: int,
    right_tip_geom: int,
    target_pos: np.ndarray,
    *,
    kp_pos: float = 8.0,
    damping: float = 1e-2,
    max_dq: float = 0.12,
) -> np.ndarray:
    """Position-only IK on the midpoint between two fingertip geoms."""
    pL = sim.data.geom_xpos[left_tip_geom].copy()
    pR = sim.data.geom_xpos[right_tip_geom].copy()
    cur = 0.5 * (pL + pR)
    err = target_pos - cur

    jacpL = np.zeros((3, sim.model.nv))
    jacrL = np.zeros((3, sim.model.nv))
    jacpR = np.zeros((3, sim.model.nv))
    jacrR = np.zeros((3, sim.model.nv))
    mujoco.mj_jacGeom(sim.model, sim.data, jacpL, jacrL, left_tip_geom)
    mujoco.mj_jacGeom(sim.model, sim.data, jacpR, jacrR, right_tip_geom)
    J = 0.5 * (jacpL[:, :7] + jacpR[:, :7])  # 3x7 midpoint Jacobian

    v = kp_pos * err
    A = J @ J.T + (damping * damping) * np.eye(3)
    dq = J.T @ np.linalg.solve(A, v)
    dq = np.clip(dq, -max_dq, max_dq)

    q = sim.data.qpos[:7].copy() + dq
    return _clip_to_ctrlrange(sim, q)


def ik_step_midpoint_with_hand_z(
    sim: Sim,
    left_tip_geom: int,
    right_tip_geom: int,
    target_pos: np.ndarray,
    *,
    target_hand_z_axis: np.ndarray = np.array([0.0, 0.0, -1.0]),
    kp_pos: float = 6.0,
    kp_ori: float = 2.0,
    damping: float = 2e-2,
    max_dq: float = 0.10,
) -> np.ndarray:
    """IK for fingertip midpoint position + hand z-axis alignment (6D objective)."""
    # Midpoint position + jacobian
    pL = sim.data.geom_xpos[left_tip_geom].copy()
    pR = sim.data.geom_xpos[right_tip_geom].copy()
    cur = 0.5 * (pL + pR)
    pos_err = target_pos - cur

    jacpL = np.zeros((3, sim.model.nv))
    jacrL = np.zeros((3, sim.model.nv))
    jacpR = np.zeros((3, sim.model.nv))
    jacrR = np.zeros((3, sim.model.nv))
    mujoco.mj_jacGeom(sim.model, sim.data, jacpL, jacrL, left_tip_geom)
    mujoco.mj_jacGeom(sim.model, sim.data, jacpR, jacrR, right_tip_geom)
    Jp = 0.5 * (jacpL[:, :7] + jacpR[:, :7])  # 3x7

    # Hand orientation (z axis) + jacobian
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    R = sim.data.xmat[hand_id].reshape(3, 3)
    z_cur = R[:, 2].copy()
    ori_err = np.cross(z_cur, target_hand_z_axis)
    jacpH = np.zeros((3, sim.model.nv))
    jacrH = np.zeros((3, sim.model.nv))
    mujoco.mj_jacBody(sim.model, sim.data, jacpH, jacrH, hand_id)
    Jr = jacrH[:, :7]

    J = np.vstack([Jp, Jr])  # 6x7
    v = np.concatenate([kp_pos * pos_err, kp_ori * ori_err], axis=0)
    A = J @ J.T + (damping * damping) * np.eye(6)
    dq = J.T @ np.linalg.solve(A, v)
    dq = np.clip(dq, -max_dq, max_dq)

    q = sim.data.qpos[:7].copy() + dq
    return _clip_to_ctrlrange(sim, q)


def hold_ctrl(sim: Sim, q: np.ndarray, gripper: float):
    sim.data.ctrl[:7] = q
    sim.data.ctrl[7] = float(gripper)


def run_script(sim: Sim, *, save_early_path: str = "/work/final_state.npz"):
    left_tip, right_tip = finger_tip_geoms(sim)

    cup_pregrasp = np.array([0.48, -0.12, 0.54], dtype=float)
    cup_grasp = np.array([0.48, -0.12, 0.44], dtype=float)
    bin_xy = np.array([BIN_CENTER[0], BIN_CENTER[1]], dtype=float)
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    # Phase 0: move above cup, gripper open (early save after we're en route).
    for t in range(900):
        q = ik_step_grasp_midpoint(sim, left_tip, right_tip, cup_pregrasp)
        hold_ctrl(sim, q, 255.0)
        sim.step(1)
        if t == 400:
            sim.save_final_state(save_early_path)

    # Phase 1: descend to grasp.
    for _ in range(900):
        q = ik_step_grasp_midpoint(sim, left_tip, right_tip, cup_grasp, kp_pos=6.0)
        hold_ctrl(sim, q, 255.0)
        sim.step(1)

    # Phase 1b: small XY spiral to seat the cup between fingers before closing.
    for k in range(500):
        ang = 2.0 * np.pi * (k / 200.0)
        r = 0.010 * min(1.0, k / 350.0)
        target = cup_grasp.copy()
        target[0] += r * np.cos(ang)
        target[1] += r * np.sin(ang)
        q = ik_step_grasp_midpoint(sim, left_tip, right_tip, target, kp_pos=5.0)
        hold_ctrl(sim, q, 255.0)
        sim.step(1)

    # Phase 2: close gripper while holding pose.
    for k in range(1200):
        q = ik_step_grasp_midpoint(sim, left_tip, right_tip, cup_grasp, kp_pos=4.5)
        grip = float(np.interp(k, [0, 900, 1200], [255.0, 0.0, 0.0]))
        hold_ctrl(sim, q, grip)
        sim.step(1)

    # Phase 2b: hold closed briefly to stabilize grasp.
    for _ in range(500):
        q = ik_step_grasp_midpoint(sim, left_tip, right_tip, cup_grasp, kp_pos=3.5)
        hold_ctrl(sim, q, 0.0)
        sim.step(1)

    # Phase 3: lift using cup feedback to guarantee z >= 0.50.
    desired_cup_z = 0.56
    for _ in range(1100):
        cup = sim.cup_position()
        midpoint = 0.5 * (sim.data.geom_xpos[left_tip] + sim.data.geom_xpos[right_tip])
        target = midpoint.copy()
        target[:2] += (np.array([0.48, -0.12]) - cup[:2])
        target[2] += (desired_cup_z - cup[2])
        q = ik_step_midpoint_with_hand_z(sim, left_tip, right_tip, target, kp_pos=5.5, kp_ori=2.0)
        hold_ctrl(sim, q, 0.0)
        sim.step(1)

    # Phase 4: translate cup above bin center at safe height.
    desired_cup_z = 0.56
    for _ in range(2200):
        cup = sim.cup_position()
        midpoint = 0.5 * (sim.data.geom_xpos[left_tip] + sim.data.geom_xpos[right_tip])
        target = midpoint.copy()
        target[:2] += (bin_xy - cup[:2])
        target[2] += (desired_cup_z - cup[2])
        q = ik_step_midpoint_with_hand_z(sim, left_tip, right_tip, target, kp_pos=4.5, kp_ori=2.0)
        hold_ctrl(sim, q, 0.0)
        sim.step(1)

    # Phase 5: lower cup into bin.
    desired_cup_z = 0.45
    for _ in range(1200):
        cup = sim.cup_position()
        midpoint = 0.5 * (sim.data.geom_xpos[left_tip] + sim.data.geom_xpos[right_tip])
        target = midpoint.copy()
        target[:2] += (bin_xy - cup[:2])
        target[2] += (desired_cup_z - cup[2])
        q = ik_step_midpoint_with_hand_z(sim, left_tip, right_tip, target, kp_pos=4.0, kp_ori=2.0)
        hold_ctrl(sim, q, 0.0)
        sim.step(1)

    # Phase 5b: dwell at low height to damp motion before release.
    for _ in range(700):
        cup = sim.cup_position()
        midpoint = 0.5 * (sim.data.geom_xpos[left_tip] + sim.data.geom_xpos[right_tip])
        target = midpoint.copy()
        target[:2] += (bin_xy - cup[:2])
        target[2] += (desired_cup_z - cup[2])
        q = ik_step_midpoint_with_hand_z(sim, left_tip, right_tip, target, kp_pos=3.0, kp_ori=2.0)
        hold_ctrl(sim, q, 0.0)
        sim.step(1)

    # Phase 6: release in place (no lift) then retreat upward to eliminate contact at end.
    for k in range(800):
        cup = sim.cup_position()
        midpoint = 0.5 * (sim.data.geom_xpos[left_tip] + sim.data.geom_xpos[right_tip])
        target = midpoint.copy()
        target[:2] += (bin_xy - cup[:2])
        target[2] += (desired_cup_z - cup[2])
        q = ik_step_midpoint_with_hand_z(sim, left_tip, right_tip, target, kp_pos=2.8, kp_ori=2.0)
        grip = float(np.interp(k, [0, 650, 800], [0.0, 255.0, 255.0]))
        hold_ctrl(sim, q, grip)
        sim.step(1)

    # Phase 7: retreat straight up with gripper open and no XY correction.
    for k in range(700):
        midpoint = 0.5 * (sim.data.geom_xpos[left_tip] + sim.data.geom_xpos[right_tip])
        target = midpoint.copy()
        target[2] += float(np.interp(k, [0, 700], [0.02, 0.20]))
        q = ik_step_midpoint_with_hand_z(sim, left_tip, right_tip, target, kp_pos=3.5, kp_ori=2.0)
        hold_ctrl(sim, q, 255.0)
        sim.step(1)

    sim.save_final_state(save_early_path)


if __name__ == "__main__":
    import mujoco

    sim = Sim()
    run_script(sim)
    print("saved /work/final_state.npz")
