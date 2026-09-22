import math
import numpy as np
import mujoco

from sim import Sim, BIN_CENTER


LEFT_PAD_GEOM = 69
RIGHT_PAD_GEOM = 77


def clamp_ctrl(sim, ctrl):
    out = np.array(ctrl, dtype=float, copy=True)
    rng = sim.model.actuator_ctrlrange
    out[:] = np.clip(out, rng[:, 0], rng[:, 1])
    return out


def hand_axes(sim):
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    mat = sim.data.xmat[hand_id].reshape(3, 3)
    return mat[:, 0], mat[:, 1], mat[:, 2]


def pinch_state(sim):
    lp = sim.data.geom_xpos[LEFT_PAD_GEOM].copy()
    rp = sim.data.geom_xpos[RIGHT_PAD_GEOM].copy()
    center = 0.5 * (lp + rp)
    open_vec = lp - rp
    norm = np.linalg.norm(open_vec)
    if norm < 1e-8:
        open_vec = np.array([0.0, 1.0, 0.0])
    else:
        open_vec /= norm
    return center, open_vec, lp, rp


def set_arm_qpos(sim, q):
    sim.data.qpos[:7] = q
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


def solve_ik(sim, target_pos, target_z=None, target_open=None, q_init=None, iters=100):
    if q_init is None:
        q = sim.data.qpos[:7].copy()
    else:
        q = np.array(q_init, dtype=float, copy=True)
    q_lo = sim.model.actuator_ctrlrange[:7, 0]
    q_hi = sim.model.actuator_ctrlrange[:7, 1]
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    jacp_l = np.zeros((3, sim.model.nv))
    jacr_l = np.zeros((3, sim.model.nv))
    jacp_r = np.zeros((3, sim.model.nv))
    jacr_r = np.zeros((3, sim.model.nv))
    jacp_hand = np.zeros((3, sim.model.nv))
    jacr_hand = np.zeros((3, sim.model.nv))

    best_q = q.copy()
    best_cost = float("inf")
    for _ in range(iters):
        set_arm_qpos(sim, q)
        center, open_vec, _, _ = pinch_state(sim)
        _, _, hand_z = hand_axes(sim)
        pos_err = target_pos - center
        err_parts = [pos_err]
        J_parts = []
        mujoco.mj_jacGeom(sim.model, sim.data, jacp_l, jacr_l, LEFT_PAD_GEOM)
        mujoco.mj_jacGeom(sim.model, sim.data, jacp_r, jacr_r, RIGHT_PAD_GEOM)
        jac_center = 0.5 * (jacp_l[:3, :7] + jacp_r[:3, :7])
        J_parts.append(jac_center)
        if target_z is not None:
            z_err = np.cross(hand_z, target_z)
            err_parts.append(0.25 * z_err)
            mujoco.mj_jacBody(sim.model, sim.data, jacp_hand, jacr_hand, hand_id)
            J_parts.append(0.25 * jacr_hand[:3, :7])
        if target_open is not None:
            open_err = np.cross(open_vec, target_open)
            err_parts.append(0.15 * open_err)
            J_parts.append(0.15 * (jacr_l[:3, :7] + jacr_r[:3, :7]) * 0.5)
        err = np.concatenate(err_parts)
        J = np.vstack(J_parts)
        cost = float(np.dot(err, err))
        if cost < best_cost:
            best_cost = cost
            best_q = q.copy()
        if np.linalg.norm(pos_err) < 3e-3 and cost < 1e-4:
            break
        damping = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(J.shape[0]), err)
        q = np.clip(q + dq, q_lo, q_hi)
    set_arm_qpos(sim, best_q)
    return best_q, best_cost


def move_ctrl(sim, arm_target, grip_target, steps, save_each=True):
    start_arm = sim.data.ctrl[:7].copy()
    start_grip = float(sim.data.ctrl[7])
    for i in range(steps):
        a = (i + 1) / steps
        ctrl = np.empty(8, dtype=float)
        ctrl[:7] = (1 - a) * start_arm + a * arm_target
        ctrl[7] = (1 - a) * start_grip + a * grip_target
        sim.data.ctrl[:] = clamp_ctrl(sim, ctrl)
        sim.step(1)
        if save_each and i == steps // 2:
            sim.save_final_state("/work/final_state.npz")


def hold(sim, steps, grip=None):
    for _ in range(steps):
        if grip is not None:
            sim.data.ctrl[7] = grip
        sim.step(1)


def eval_run(sim):
    trace = np.array(sim._trace, dtype=object)
    contact_hist = np.array([float(x["cup_contact"]) for x in sim._trace], dtype=float)
    cup_hist = np.array([np.array(x["cup_pos"]) for x in sim._trace], dtype=float)
    replay_contact_any = bool(np.any(contact_hist > 0.5))
    best_z = float(np.max(cup_hist[:, 2]))
    last10_frac = float(np.mean(contact_hist[-10:])) if len(contact_hist) >= 10 else float(np.mean(contact_hist))

    settle_contact = []
    for _ in range(500):
        sim.step(1)
        settle_contact.append(float(sim.has_gripper_cup_contact()))
    settle_contact_frac = float(np.mean(settle_contact))

    cup_body = sim.cup_body_id
    cup_pos = sim.cup_position()
    cup_up_z = abs(sim.data.xmat[cup_body].reshape(3, 3)[2, 2])
    inside = abs(cup_pos[0] - BIN_CENTER[0]) <= 0.046 and abs(cup_pos[1] - BIN_CENTER[1]) <= 0.046
    success = (
        len(sim._ctrl_trace) >= 20
        and replay_contact_any
        and best_z >= 0.50
        and last10_frac < 0.5
        and settle_contact_frac < 0.1
        and inside
        and cup_pos[2] >= 0.42
        and cup_up_z >= 0.85
    )
    return {
        "success": success,
        "contact_any": replay_contact_any,
        "best_z": best_z,
        "last10_frac": last10_frac,
        "settle_contact_frac": settle_contact_frac,
        "final_pos": cup_pos,
        "inside": inside,
        "final_up_z": cup_up_z,
        "steps": len(sim._ctrl_trace),
    }


def run_attempt():
    sim = Sim()
    sim.data.ctrl[:7] = sim.data.qpos[:7]
    sim.data.ctrl[7] = 255
    hold(sim, 30, grip=255)
    sim.save_final_state("/work/final_state.npz")

    cup0 = sim.cup_position().copy()
    q = sim.data.qpos[:7].copy()
    open_axis = np.array([0.0, 1.0, 0.0])
    down = np.array([0.0, 0.0, -1.0])

    waypoints = [
        (cup0 + np.array([0.0, 0.0, 0.16]), 255, 150),
        (cup0 + np.array([0.0, 0.0, 0.085]), 220, 140),
        (cup0 + np.array([0.0, 0.0, 0.055]), 180, 140),
        (cup0 + np.array([0.0, 0.0, 0.040]), 120, 140),
    ]
    for pos, grip, steps in waypoints:
        q, _ = solve_ik(sim, pos, target_z=down, target_open=open_axis, q_init=q)
        move_ctrl(sim, q, grip, steps)

    for grip in [90, 70, 50, 35, 20, 10]:
        current_target = sim.cup_position().copy() + np.array([0.0, 0.0, 0.040])
        q, _ = solve_ik(sim, current_target, target_z=down, target_open=open_axis, q_init=sim.data.qpos[:7])
        move_ctrl(sim, q, grip, 60)

    hold(sim, 60, grip=10)

    lift_targets = [
        sim.cup_position().copy() + np.array([0.0, 0.0, 0.10]),
        np.array([0.58, -0.02, 0.58]),
        np.array([0.70, 0.14, 0.58]),
        np.array([0.70, 0.14, 0.51]),
        np.array([0.70, 0.14, 0.475]),
    ]
    for pos in lift_targets:
        q, _ = solve_ik(sim, pos, target_z=down, target_open=open_axis, q_init=sim.data.qpos[:7])
        move_ctrl(sim, q, 10, 140)

    for grip in [30, 60, 100, 160, 220, 255]:
        q, _ = solve_ik(sim, np.array([0.70, 0.14, 0.50]), target_z=down, target_open=open_axis, q_init=sim.data.qpos[:7])
        move_ctrl(sim, q, grip, 40)

    q, _ = solve_ik(sim, np.array([0.70, 0.14, 0.62]), target_z=down, target_open=open_axis, q_init=sim.data.qpos[:7])
    move_ctrl(sim, q, 255, 120)
    hold(sim, 200, grip=255)
    sim.save_final_state("/work/final_state.npz")

    metrics = eval_run(sim)
    print(metrics)
    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    run_attempt()
