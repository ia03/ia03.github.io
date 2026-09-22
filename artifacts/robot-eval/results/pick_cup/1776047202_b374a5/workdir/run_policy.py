import numpy as np
import mujoco

from sim import Sim, SUCCESS_CUP_Z


def _finger_geom_ids(model):
    left_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    return {
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == left_bid or model.geom_bodyid[i] == right_bid
    }


def _cup_geom_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")


def fingertip_midpoint(sim: Sim):
    m = sim.model
    left_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    return 0.5 * (sim.data.xpos[left_bid] + sim.data.xpos[right_bid])


def jacobian_fingertip_midpoint(sim: Sim):
    m = sim.model
    d = sim.data
    left_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    jacp_l = np.zeros((3, m.nv))
    jacr_l = np.zeros((3, m.nv))
    jacp_r = np.zeros((3, m.nv))
    jacr_r = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp_l, jacr_l, left_bid)
    mujoco.mj_jacBody(m, d, jacp_r, jacr_r, right_bid)
    return 0.5 * (jacp_l + jacp_r)


def finger_positions(sim: Sim):
    m = sim.model
    left_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    return sim.data.xpos[left_bid].copy(), sim.data.xpos[right_bid].copy()


def jacobian_finger_positions(sim: Sim):
    m = sim.model
    d = sim.data
    left_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    jacp_l = np.zeros((3, m.nv))
    jacr_l = np.zeros((3, m.nv))
    jacp_r = np.zeros((3, m.nv))
    jacr_r = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp_l, jacr_l, left_bid)
    mujoco.mj_jacBody(m, d, jacp_r, jacr_r, right_bid)
    return jacp_l, jacp_r


def damped_ls_solve(J, err, damping=1e-2):
    JJt = J @ J.T
    JJt += (damping**2) * np.eye(JJt.shape[0])
    return J.T @ np.linalg.solve(JJt, err)


def solve_ik_midpoint(
    sim: Sim,
    target_pos,
    q_init=None,
    max_iters=200,
    tol=5e-4,
    step_scale=0.6,
):
    m = sim.model
    if q_init is None:
        q = sim.data.qpos[:7].copy()
    else:
        q = np.array(q_init, dtype=float).copy()

    # Start with gripper open so fingers are out of the way.
    sim.data.qpos[:7] = q
    sim.data.qpos[7] = 0.04
    sim.data.qpos[8] = 0.04
    mujoco.mj_forward(m, sim.data)

    ctrl_min = m.actuator_ctrlrange[:7, 0]
    ctrl_max = m.actuator_ctrlrange[:7, 1]

    for _ in range(max_iters):
        mid = fingertip_midpoint(sim)
        err = np.array(target_pos, dtype=float) - mid
        if float(np.linalg.norm(err)) < tol:
            break
        J = jacobian_fingertip_midpoint(sim)[:, :7]
        dq = damped_ls_solve(J, err, damping=2e-2)
        q = q + step_scale * dq
        q = np.clip(q, ctrl_min, ctrl_max)
        sim.data.qpos[:7] = q
        mujoco.mj_forward(m, sim.data)
    return q


def solve_ik_grasp_shape(
    sim: Sim,
    target_mid,
    target_width_y,
    q_init=None,
    max_iters=250,
    tol=8e-4,
    step_scale=0.5,
):
    m = sim.model
    if q_init is None:
        q = sim.data.qpos[:7].copy()
    else:
        q = np.array(q_init, dtype=float).copy()

    sim.data.qpos[:7] = q
    sim.data.qpos[7] = 0.04
    sim.data.qpos[8] = 0.04
    mujoco.mj_forward(m, sim.data)

    ctrl_min = m.actuator_ctrlrange[:7, 0]
    ctrl_max = m.actuator_ctrlrange[:7, 1]

    for _ in range(max_iters):
        p_l, p_r = finger_positions(sim)
        mid = 0.5 * (p_l + p_r)
        diff = p_r - p_l

        err_mid = np.array(target_mid, dtype=float) - mid
        # Encourage finger separation to be mostly along world Y with desired width.
        err_shape = np.array(
            [0.0 - diff[0], target_width_y - diff[1], 0.0 - diff[2]], dtype=float
        )
        err = np.concatenate([err_mid, err_shape], axis=0)  # (6,)
        if float(np.linalg.norm(err)) < tol:
            break

        Jl, Jr = jacobian_finger_positions(sim)
        J_mid = 0.5 * (Jl + Jr)
        J_diff = (Jr - Jl)
        J = np.vstack([J_mid, J_diff])[:, :7]  # (6,7)
        dq = damped_ls_solve(J, err, damping=3e-2)
        q = q + step_scale * dq
        q = np.clip(q, ctrl_min, ctrl_max)
        sim.data.qpos[:7] = q
        mujoco.mj_forward(m, sim.data)
    return q


def select_inner_pad_geoms(sim: Sim):
    m = sim.model
    # Slightly open gripper so the inner faces are defined.
    sim.data.qpos[7] = 0.04
    sim.data.qpos[8] = 0.04
    mujoco.mj_forward(m, sim.data)

    left_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    left_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == left_bid]
    right_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == right_bid]
    # Inner-most geoms are closest to y=0: for left that is max y, for right min y.
    left_gid = max(left_geoms, key=lambda i: float(sim.data.geom_xpos[i][1]))
    right_gid = min(right_geoms, key=lambda i: float(sim.data.geom_xpos[i][1]))
    return left_gid, right_gid


def solve_ik_geom_pair_pose(
    sim: Sim,
    geom_left,
    geom_right,
    target_mid,
    target_diff_y=None,
    q_init=None,
    max_iters=350,
    tol=8e-4,
    step_scale=0.45,
):
    m = sim.model
    if q_init is None:
        q = sim.data.qpos[:7].copy()
    else:
        q = np.array(q_init, dtype=float).copy()

    sim.data.qpos[:7] = q
    sim.data.qpos[7] = 0.04
    sim.data.qpos[8] = 0.04
    mujoco.mj_forward(m, sim.data)

    if target_diff_y is None:
        target_diff_y = float((sim.data.geom_xpos[geom_right] - sim.data.geom_xpos[geom_left])[1])

    ctrl_min = m.actuator_ctrlrange[:7, 0]
    ctrl_max = m.actuator_ctrlrange[:7, 1]

    jacp_l = np.zeros((3, m.nv))
    jacr_l = np.zeros((3, m.nv))
    jacp_r = np.zeros((3, m.nv))
    jacr_r = np.zeros((3, m.nv))

    for _ in range(max_iters):
        p_l = sim.data.geom_xpos[geom_left].copy()
        p_r = sim.data.geom_xpos[geom_right].copy()
        mid = 0.5 * (p_l + p_r)
        diff = p_r - p_l

        err_mid = np.array(target_mid, dtype=float) - mid
        err_shape = np.array([0.0 - diff[0], target_diff_y - diff[1], 0.0 - diff[2]], dtype=float)
        err = np.concatenate([err_mid, err_shape], axis=0)
        if float(np.linalg.norm(err)) < tol:
            break

        mujoco.mj_jacGeom(m, sim.data, jacp_l, jacr_l, geom_left)
        mujoco.mj_jacGeom(m, sim.data, jacp_r, jacr_r, geom_right)
        J_mid = 0.5 * (jacp_l + jacp_r)
        J_diff = (jacp_r - jacp_l)
        J = np.vstack([J_mid, J_diff])[:, :7]
        dq = damped_ls_solve(J, err, damping=4e-2)
        q = q + step_scale * dq
        q = np.clip(q, ctrl_min, ctrl_max)
        sim.data.qpos[:7] = q
        mujoco.mj_forward(m, sim.data)

    return q


def solve_ik_two_geoms(
    sim: Sim,
    geom_left,
    geom_right,
    target_left,
    target_right,
    q_init=None,
    max_iters=300,
    tol=8e-4,
    step_scale=0.45,
):
    m = sim.model
    if q_init is None:
        q = sim.data.qpos[:7].copy()
    else:
        q = np.array(q_init, dtype=float).copy()

    # Open gripper for clearance while solving.
    sim.data.qpos[:7] = q
    sim.data.qpos[7] = 0.04
    sim.data.qpos[8] = 0.04
    mujoco.mj_forward(m, sim.data)

    ctrl_min = m.actuator_ctrlrange[:7, 0]
    ctrl_max = m.actuator_ctrlrange[:7, 1]

    jacp_l = np.zeros((3, m.nv))
    jacr_l = np.zeros((3, m.nv))
    jacp_r = np.zeros((3, m.nv))
    jacr_r = np.zeros((3, m.nv))

    for _ in range(max_iters):
        p_l = sim.data.geom_xpos[geom_left].copy()
        p_r = sim.data.geom_xpos[geom_right].copy()
        err = np.concatenate(
            [np.array(target_left, dtype=float) - p_l, np.array(target_right, dtype=float) - p_r],
            axis=0,
        )  # (6,)
        if float(np.linalg.norm(err)) < tol:
            break

        jacp_l[:] = 0
        jacr_l[:] = 0
        jacp_r[:] = 0
        jacr_r[:] = 0
        mujoco.mj_jacGeom(m, sim.data, jacp_l, jacr_l, geom_left)
        mujoco.mj_jacGeom(m, sim.data, jacp_r, jacr_r, geom_right)
        J = np.vstack([jacp_l, jacp_r])[:, :7]  # (6,7)
        dq = damped_ls_solve(J, err, damping=4e-2)
        q = q + step_scale * dq
        q = np.clip(q, ctrl_min, ctrl_max)
        sim.data.qpos[:7] = q
        mujoco.mj_forward(m, sim.data)

    return q


def set_arm_and_gripper_ctrl(sim: Sim, q, gripper):
    sim.data.ctrl[:7] = q
    sim.data.ctrl[7] = gripper


def step_hold(sim: Sim, q, gripper, steps):
    set_arm_and_gripper_ctrl(sim, q, gripper)
    sim.step(steps)


def close_gripper_ramp(sim: Sim, q, steps=300, start=255, end=0):
    for t in range(steps):
        u = t / max(steps - 1, 1)
        g = (1 - u) * start + u * end
        set_arm_and_gripper_ctrl(sim, q, g)
        sim.step(1)


def settle_contact_fraction(sim: Sim, steps=500):
    m = sim.model
    d = sim.data
    cup_gid = _cup_geom_id(m)
    finger_gids = _finger_geom_ids(m)
    contact_steps = 0
    for _ in range(steps):
        had_contact = False
        for ci in range(d.ncon):
            c = d.contact[ci]
            if c.geom1 == cup_gid and c.geom2 in finger_gids:
                had_contact = True
                break
            if c.geom2 == cup_gid and c.geom1 in finger_gids:
                had_contact = True
                break
        contact_steps += int(had_contact)
        sim.step(1)
    return contact_steps / steps


def rollout_and_score(
    pregrasp_z=0.55,
    grasp_z=0.44,
    lift_z=0.70,
    approach_steps=800,
    descend_steps=500,
    squeeze_steps=400,
    lift_steps=700,
    hold_steps=400,
    save_path=None,
    render_debug=False,
):
    # Use a separate sim for IK so we can cleanly reset rollout.
    sim_ik = Sim()
    cup = sim_ik.cup_position()
    # Use a stable pair of finger pad geoms.
    left_gid = 69
    right_gid = 77

    q_pre = solve_ik_geom_pair_pose(
        sim_ik, left_gid, right_gid, target_mid=[cup[0], cup[1], pregrasp_z]
    )
    q_grasp = solve_ik_geom_pair_pose(
        sim_ik, left_gid, right_gid, target_mid=[cup[0], cup[1], grasp_z], q_init=q_pre
    )
    q_lift = solve_ik_geom_pair_pose(
        sim_ik, left_gid, right_gid, target_mid=[cup[0], cup[1], lift_z], q_init=q_grasp
    )

    sim = Sim()

    # Stage 0: open gripper, move above cup.
    step_hold(sim, q_pre, 255, approach_steps)

    # Stage 1: descend with gripper open.
    for t in range(descend_steps):
        u = t / max(descend_steps - 1, 1)
        q = (1 - u) * q_pre + u * q_grasp
        set_arm_and_gripper_ctrl(sim, q, 255)
        sim.step(1)

    # Stage 2: close gripper.
    close_gripper_ramp(sim, q_grasp, steps=squeeze_steps, start=255, end=0)

    # Stage 3: lift.
    for t in range(lift_steps):
        u = t / max(lift_steps - 1, 1)
        q = (1 - u) * q_grasp + u * q_lift
        set_arm_and_gripper_ctrl(sim, q, 0)
        sim.step(1)

    # Stage 4: hold at top (important so the saved control history settles while grasping).
    step_hold(sim, q_lift, 0, hold_steps)

    if render_debug:
        try:
            from PIL import Image

            Image.fromarray(sim.render(width=320, height=240)).save("debug_final.png")
        except Exception:
            pass

    # Approximate grading: extra settle with final control held constant.
    contact_frac = settle_contact_fraction(sim, steps=500)
    final_z = float(sim.cup_position()[2])
    height_progress = float(np.clip((final_z - 0.435) / (SUCCESS_CUP_Z - 0.435), 0, 1))
    contact_progress = float(np.clip(contact_frac / 0.5, 0, 1))
    progress_score = 0.5 * height_progress + 0.5 * contact_progress
    passed = final_z >= SUCCESS_CUP_Z and contact_frac >= 0.5

    if save_path is not None:
        sim.save_final_state(save_path)

    return {
        "passed": passed,
        "final_z_after_settle": final_z,
        "contact_fraction": contact_frac,
        "progress_score": progress_score,
        "steps": len(sim._ctrl_trace),
    }


if __name__ == "__main__":
    res = rollout_and_score(save_path=None, render_debug=True)
    print(res)
