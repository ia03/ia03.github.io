import numpy as np
import mujoco

from sim import Sim, CUP_RADIUS, SUCCESS_CUP_Z


def _arm_dof_indices(model: mujoco.MjModel) -> np.ndarray:
    dofs = []
    for j in range(7):
        dof_adr = model.jnt_dofadr[j]
        dofs.append(dof_adr)
    return np.array(dofs, dtype=int)


def _ctrl_clamp(sim: Sim, ctrl: np.ndarray) -> np.ndarray:
    ctrl = ctrl.copy()
    for i in range(sim.model.nu):
        if sim.model.actuator_ctrllimited[i]:
            lo, hi = sim.model.actuator_ctrlrange[i]
            ctrl[i] = float(np.clip(ctrl[i], lo, hi))
    return ctrl


def _damped_ls(J: np.ndarray, err: np.ndarray, damping: float) -> np.ndarray:
    # dq = J^T (J J^T + λ^2 I)^-1 err
    JJt = J @ J.T
    A = JJt + (damping**2) * np.eye(JJt.shape[0])
    x = np.linalg.solve(A, err)
    return J.T @ x


def _body_quat(data: mujoco.MjData, body_id: int) -> np.ndarray:
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, data.xmat[body_id])
    return q


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    M = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(M, q)
    return M.reshape(3, 3)


def ik_step_hand_pose(
    sim: Sim,
    body_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    *,
    damping: float = 0.10,
    step_size: float = 0.30,
    rot_weight: float = 0.35,
) -> tuple[float, float]:
    """One resolved-rate IK step for hand position + orientation.

    Returns (pos_err_norm_m, rot_err_norm_rad).
    """
    m, d = sim.model, sim.data
    arm_dofs = _arm_dof_indices(m)

    jacp = np.zeros((3, m.nv), dtype=float)
    jacr = np.zeros((3, m.nv), dtype=float)
    mujoco.mj_jacBody(m, d, jacp, jacr, body_id)
    jacp = jacp[:, arm_dofs]
    jacr = jacr[:, arm_dofs]

    p_cur = d.xpos[body_id].copy()
    q_cur = _body_quat(d, body_id)

    epos = (target_pos - p_cur).astype(float)
    erot = np.zeros(3, dtype=float)
    mujoco.mju_subQuat(erot, target_quat, q_cur)

    J = np.vstack([jacp, rot_weight * jacr])  # (6, 7)
    err = np.concatenate([epos, rot_weight * erot], axis=0)  # (6,)
    dq = _damped_ls(J, err, damping=damping)
    dq = np.clip(dq, -0.20, 0.20)

    q = d.qpos[:7].copy()
    q_cmd = q + step_size * dq
    ctrl = d.ctrl.copy()
    ctrl[:7] = q_cmd
    ctrl = _ctrl_clamp(sim, ctrl)
    d.ctrl[:] = ctrl

    return float(np.linalg.norm(epos)), float(np.linalg.norm(erot))


def move_hand_pose(
    sim: Sim,
    body_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    *,
    max_updates: int,
    steps_per_update: int = 5,
    pos_tol: float = 0.006,
):
    for _ in range(max_updates):
        pos_err, _ = ik_step_hand_pose(sim, body_id, target_pos, target_quat)
        sim.step(steps_per_update)
        if pos_err < pos_tol:
            break


def ik_step_tasks(
    sim: Sim,
    *,
    finger_targets: tuple[np.ndarray, np.ndarray] | None = None,
    hand_target_pos: np.ndarray | None = None,
    hand_target_quat: np.ndarray | None = None,
    damping: float = 0.10,
    step_size: float = 0.30,
    w_finger: float = 1.0,
    w_hand_pos: float = 0.6,
    w_hand_rot: float = 0.25,
) -> float:
    """One IK step with stacked tasks (fingers + optional hand pose). Returns mean position error (m)."""
    m, d = sim.model, sim.data
    arm_dofs = _arm_dof_indices(m)

    J_rows = []
    e_rows = []
    pos_errs = []

    if finger_targets is not None:
        lf_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        rf_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        for body_id, p_des in [(lf_id, finger_targets[0]), (rf_id, finger_targets[1])]:
            jacp = np.zeros((3, m.nv), dtype=float)
            jacr = np.zeros((3, m.nv), dtype=float)
            mujoco.mj_jacBody(m, d, jacp, jacr, body_id)
            jacp = jacp[:, arm_dofs]
            p_cur = d.xpos[body_id].copy()
            epos = (p_des - p_cur).astype(float)
            J_rows.append(w_finger * jacp)
            e_rows.append(w_finger * epos)
            pos_errs.append(float(np.linalg.norm(epos)))

    if hand_target_pos is not None or hand_target_quat is not None:
        hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        jacp = np.zeros((3, m.nv), dtype=float)
        jacr = np.zeros((3, m.nv), dtype=float)
        mujoco.mj_jacBody(m, d, jacp, jacr, hand_id)
        jacp = jacp[:, arm_dofs]
        jacr = jacr[:, arm_dofs]

        if hand_target_pos is not None:
            p_cur = d.xpos[hand_id].copy()
            epos = (hand_target_pos - p_cur).astype(float)
            J_rows.append(w_hand_pos * jacp)
            e_rows.append(w_hand_pos * epos)
            pos_errs.append(float(np.linalg.norm(epos)))

        if hand_target_quat is not None:
            q_cur = _body_quat(d, hand_id)
            erot = np.zeros(3, dtype=float)
            mujoco.mju_subQuat(erot, hand_target_quat, q_cur)
            J_rows.append(w_hand_rot * jacr)
            e_rows.append(w_hand_rot * erot)

    if not J_rows:
        return 0.0

    J = np.concatenate(J_rows, axis=0)
    err = np.concatenate(e_rows, axis=0)
    dq = _damped_ls(J, err, damping=damping)
    dq = np.clip(dq, -0.20, 0.20)

    q = d.qpos[:7].copy()
    q_cmd = q + step_size * dq
    ctrl = d.ctrl.copy()
    ctrl[:7] = q_cmd
    d.ctrl[:] = _ctrl_clamp(sim, ctrl)

    return float(np.mean(pos_errs)) if pos_errs else 0.0


def move_with_tasks(
    sim: Sim,
    *,
    finger_targets: tuple[np.ndarray, np.ndarray] | None = None,
    hand_target_pos: np.ndarray | None = None,
    hand_target_quat: np.ndarray | None = None,
    max_updates: int,
    steps_per_update: int = 5,
    tol: float = 0.006,
):
    for _ in range(max_updates):
        err = ik_step_tasks(
            sim,
            finger_targets=finger_targets,
            hand_target_pos=hand_target_pos,
            hand_target_quat=hand_target_quat,
        )
        sim.step(steps_per_update)
        if err < tol:
            break


def any_finger_cup_contact(sim: Sim) -> bool:
    m, d = sim.model, sim.data
    cup_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")
    lf_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    rf_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    finger_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] in (lf_body, rf_body)}
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if cup_geom in (g1, g2):
            other = g2 if g1 == cup_geom else g1
            if other in finger_geoms:
                return True
    return False


def evaluate_trace(ctrl_trace: np.ndarray) -> dict:
    sim = Sim()
    sim.reset()
    for u in ctrl_trace:
        sim.data.ctrl[:] = _ctrl_clamp(sim, np.asarray(u, dtype=float))
        sim.step(1)

    # settle: 500 steps with last ctrl held
    last = sim.data.ctrl.copy()
    contacts = 0
    for _ in range(500):
        sim.data.ctrl[:] = last
        sim.step(1)
        if any_finger_cup_contact(sim):
            contacts += 1

    cup_z = float(sim.cup_position()[2])
    contact_frac = contacts / 500.0
    return {"final_cup_z": cup_z, "contact_frac": contact_frac}


def solve_hand_ik(
    model: mujoco.MjModel,
    *,
    hand_body_id: int,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    q_seed: np.ndarray,
    n_restarts: int = 8,
    iters: int = 140,
    damping: float = 0.08,
    step_size: float = 0.55,
    rot_weight: float = 0.35,
) -> tuple[np.ndarray, float, float]:
    """Kinematic IK (no dynamics). Returns (q_best, pos_err, rot_err)."""
    data = mujoco.MjData(model)
    arm_dofs = _arm_dof_indices(model)
    ctrl_lo = model.actuator_ctrlrange[:7, 0]
    ctrl_hi = model.actuator_ctrlrange[:7, 1]

    best_q = q_seed.copy()
    best_cost = float("inf")
    best_pos = float("inf")
    best_rot = float("inf")

    rng = np.random.default_rng(0)
    for r in range(n_restarts):
        if r == 0:
            q = q_seed.copy()
        else:
            q = q_seed + rng.normal(scale=0.35, size=7)
            q = np.clip(q, ctrl_lo, ctrl_hi)

        for _ in range(iters):
            data.qpos[:] = 0.0
            data.qpos[:7] = q
            mujoco.mj_forward(model, data)

            jacp = np.zeros((3, model.nv), dtype=float)
            jacr = np.zeros((3, model.nv), dtype=float)
            mujoco.mj_jacBody(model, data, jacp, jacr, hand_body_id)
            jacp = jacp[:, arm_dofs]
            jacr = jacr[:, arm_dofs]

            p_cur = data.xpos[hand_body_id].copy()
            q_cur = _body_quat(data, hand_body_id)

            epos = (target_pos - p_cur).astype(float)
            erot = np.zeros(3, dtype=float)
            mujoco.mju_subQuat(erot, target_quat, q_cur)

            J = np.vstack([jacp, rot_weight * jacr])
            err = np.concatenate([epos, rot_weight * erot], axis=0)
            dq = _damped_ls(J, err, damping=damping)
            dq = np.clip(dq, -0.35, 0.35)

            q = q + step_size * dq
            q = np.clip(q, ctrl_lo, ctrl_hi)

            if float(np.linalg.norm(epos)) < 0.004 and float(np.linalg.norm(erot)) < 0.10:
                break

        data.qpos[:] = 0.0
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        p_cur = data.xpos[hand_body_id].copy()
        q_cur = _body_quat(data, hand_body_id)
        epos = target_pos - p_cur
        erot = np.zeros(3, dtype=float)
        mujoco.mju_subQuat(erot, target_quat, q_cur)
        pos_err = float(np.linalg.norm(epos))
        rot_err = float(np.linalg.norm(erot))
        cost = pos_err + 0.15 * rot_err
        if cost < best_cost:
            best_cost = cost
            best_q = q.copy()
            best_pos = pos_err
            best_rot = rot_err

    return best_q, best_pos, best_rot


def drive_joint_targets(sim: Sim, q_start: np.ndarray, q_goal: np.ndarray, n_steps: int, grip: float):
    q_start = np.asarray(q_start, dtype=float).reshape(7)
    q_goal = np.asarray(q_goal, dtype=float).reshape(7)
    for t in range(n_steps):
        s = (t + 1) / n_steps
        d = sim.data
        d.ctrl[:7] = (1 - s) * q_start + s * q_goal
        d.ctrl[7] = float(grip)
        sim.step(1)


def main():
    sim = Sim()
    m = sim.model
    d = sim.data

    hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    lf_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    rf_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    # Start: open gripper, let settle a bit.
    d.ctrl[:7] = d.qpos[:7]
    d.ctrl[7] = 255.0
    sim.step(200)

    cup = sim.cup_position()
    quat_des = _body_quat(d, hand_id)

    # Plan joint-space waypoints with kinematic IK (avoids local minima in dynamic IK).
    q0 = d.qpos[:7].copy()
    above_pos = cup + np.array([0.0, 0.0, 0.36], dtype=float)  # hand body
    grasp_pos = cup + np.array([0.0, 0.0, 0.07], dtype=float)  # hand body
    lift_pos = cup + np.array([0.0, 0.0, 0.42], dtype=float)  # hand body

    q_above, e_above, _ = solve_hand_ik(m, hand_body_id=hand_id, target_pos=above_pos, target_quat=quat_des, q_seed=q0)
    q_grasp, e_grasp, _ = solve_hand_ik(m, hand_body_id=hand_id, target_pos=grasp_pos, target_quat=quat_des, q_seed=q_above)
    q_lift, e_lift, _ = solve_hand_ik(m, hand_body_id=hand_id, target_pos=lift_pos, target_quat=quat_des, q_seed=q_grasp)

    print(f"IK errors: above={e_above:.3f} grasp={e_grasp:.3f} lift={e_lift:.3f}")

    # Re-run from canonical start with a smooth joint-space trajectory.
    sim.reset()
    d.ctrl[:7] = d.qpos[:7]
    d.ctrl[7] = 255.0
    sim.step(200)

    drive_joint_targets(sim, d.qpos[:7].copy(), q_above, n_steps=2600, grip=255.0)
    drive_joint_targets(sim, q_above, q_grasp, n_steps=2000, grip=255.0)

    # Close gripper smoothly while holding current arm target.
    d.ctrl[:7] = _ctrl_clamp(sim, d.ctrl)[:7]
    for val in np.linspace(255, 0, 600):
        d.ctrl[7] = float(val)
        sim.step(2)

    # Lift.
    drive_joint_targets(sim, q_grasp, q_lift, n_steps=2600, grip=0.0)

    # Hold for a bit for stability.
    d.ctrl[7] = 0.0
    sim.step(1000)

    metrics = evaluate_trace(np.array(sim._ctrl_trace))
    print(
        f"eval: cup_z={metrics['final_cup_z']:.3f} (target {SUCCESS_CUP_Z}), "
        f"contact_frac={metrics['contact_frac']:.3f} (target 0.5)"
    )

    sim.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")


if __name__ == "__main__":
    main()
