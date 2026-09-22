import numpy as np
import mujoco

from sim import Sim


def _body_id(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def ik_hand_position(sim: Sim, target_pos, q_init, iters=300, lam=1e-2):
    """Solve q for hand body position (no orientation)."""
    m, d = sim.model, sim.data
    hand_id = _body_id(m, "hand")

    q = np.array(q_init, dtype=float).copy()
    lo = m.actuator_ctrlrange[:7, 0]
    hi = m.actuator_ctrlrange[:7, 1]

    for _ in range(iters):
        d.qpos[:7] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        err = np.array(target_pos, dtype=float) - d.xpos[hand_id]
        if np.linalg.norm(err) < 1e-4:
            break

        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, None, hand_id)
        J = jacp[:, :7]
        H = J.T @ J + lam * np.eye(7)
        dq = np.linalg.solve(H, J.T @ (2.0 * err))
        dq = np.clip(dq, -0.07, 0.07)
        q = np.clip(q + dq, lo, hi)

    return q


def ik_finger_center_position(sim: Sim, target_pos, q_init, iters=400, lam=1e-2, finger_open_q=0.04):
    """Solve arm q for the average position of left/right finger bodies."""
    m, d = sim.model, sim.data
    left_id = _body_id(m, "left_finger")
    right_id = _body_id(m, "right_finger")

    q = np.array(q_init, dtype=float).copy()
    lo = m.actuator_ctrlrange[:7, 0]
    hi = m.actuator_ctrlrange[:7, 1]

    for _ in range(iters):
        d.qpos[:7] = q
        d.qpos[7] = finger_open_q
        d.qpos[8] = finger_open_q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        p = 0.5 * (d.xpos[left_id] + d.xpos[right_id])
        err = np.array(target_pos, dtype=float) - p
        if np.linalg.norm(err) < 1e-4:
            break

        jacp_l = np.zeros((3, m.nv))
        jacp_r = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp_l, None, left_id)
        mujoco.mj_jacBody(m, d, jacp_r, None, right_id)
        J = 0.5 * (jacp_l[:, :7] + jacp_r[:, :7])

        H = J.T @ J + lam * np.eye(7)
        dq = np.linalg.solve(H, J.T @ (2.0 * err))
        dq = np.clip(dq, -0.07, 0.07)
        q = np.clip(q + dq, lo, hi)

    return q


def lerp(a, b, s):
    return (1.0 - s) * a + s * b


def peg_alignment_x(sim: Sim) -> float:
    m, d = sim.model, sim.data
    peg_id = _body_id(m, "peg")
    R = d.xmat[peg_id].reshape(3, 3)
    # Dot(body_x_axis_world, world_x_axis) == first column x component.
    return float(abs(R[0, 0]))


def main():
    sim = Sim()
    m, d = sim.model, sim.data

    # Start with a stable, within-range posture.
    q0 = np.array([0.0, -0.6, 0.0, -1.8, 0.0, 1.3, 0.7], dtype=float)
    q0 = np.clip(q0, m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1])

    # Command the finger-center (average of left/right finger bodies).
    lane_y = -0.102
    peg_z = 0.52
    tab_x = 0.42

    contact_x = tab_x - 0.02
    contact_z = peg_z - 0.005
    p_high = np.array([contact_x, lane_y, peg_z + 0.10])
    p_contact = np.array([contact_x, lane_y, contact_z])

    push_xs = [contact_x, 0.50, 0.60, 0.66, 0.68]
    push_ps = [np.array([x, lane_y, contact_z]) for x in push_xs]

    q_high = ik_finger_center_position(sim, p_high, q0, iters=1200)
    q_low = ik_finger_center_position(sim, p_contact, q_high, iters=1200)
    push_qs = [ik_finger_center_position(sim, p, q_low, iters=1200) for p in push_ps]

    # Controller timeline (replay steps).
    # Keep it under ~4500 for full efficiency; start modest.
    steps_settle = 250
    steps_high = 500
    steps_down = 400
    steps_push_seg = 350
    steps_hold_push = 400
    steps_retract = 250
    steps_close = 200

    # Open gripper for pushing with the palm/fingers.
    gripper_open = 255.0
    gripper_pinch = 80.0

    def step_hold(q_cmd, n, grip_cmd):
        for _ in range(n):
            d.ctrl[:7] = q_cmd
            d.ctrl[7] = grip_cmd
            sim.step(1)

    # Initial settle into commanded posture.
    step_hold(q0, steps_settle, gripper_open)

    # Move to approach above the peg/tab lane.
    for i in range(steps_high):
        s = (i + 1) / steps_high
        step_hold(lerp(q0, q_high, s), 1, gripper_open)

    # Lower to contact height.
    for i in range(steps_down):
        s = (i + 1) / steps_down
        step_hold(lerp(q_high, q_low, s), 1, gripper_open)

    # Gently close to increase contact reliability (light pinch on the tab/body).
    for i in range(steps_close):
        s = (i + 1) / steps_close
        grip = lerp(gripper_open, gripper_pinch, s)
        step_hold(q_low, 1, grip)

    # Push forward in segments; stop early if we hit the insertion threshold.
    for seg in range(len(push_qs) - 1):
        qa, qb = push_qs[seg], push_qs[seg + 1]
        for i in range(steps_push_seg):
            s = (i + 1) / steps_push_seg
            step_hold(lerp(qa, qb, s), 1, gripper_pinch)
            if sim.peg_position()[0] >= 0.535:
                break

    # Hold pressure briefly to ensure insertion persists.
    step_hold(push_qs[-1], steps_hold_push, gripper_pinch)

    # Retract a bit to avoid dragging the peg back out.
    q_retract = ik_finger_center_position(sim, np.array([0.58, lane_y, peg_z + 0.05]), push_qs[-1], iters=1200)
    for i in range(steps_retract):
        s = (i + 1) / steps_retract
        step_hold(lerp(push_qs[-1], q_retract, s), 1, gripper_open)

    # Save mandatory output.
    sim.save_final_state("/work/final_state.npz")

    peg_p = sim.peg_position()
    print("final replay steps:", len(sim._ctrl_trace))
    print("final peg pos:", peg_p)
    print("final peg x-axis alignment:", peg_alignment_x(sim))


if __name__ == "__main__":
    main()
