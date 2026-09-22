import numpy as np
import mujoco
from sim import Sim


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def orientation_error(q_cur, q_des):
    q_rel = quat_mul(q_des, quat_conj(q_cur))
    if q_rel[0] < 0:
        q_rel = -q_rel
    return 2.0 * q_rel[1:]


def run_attempt(save_path="/work/final_state.npz", render_debug=False):
    sim = Sim()
    m, d = sim.model, sim.data

    hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    cup_id = sim.cup_body_id

    # Move to a known good pre-shape and use its hand orientation as desired grasp orientation.
    q_seed = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.8])
    d.ctrl[:7] = q_seed
    d.ctrl[7] = 255.0
    sim.step(1800)

    q_des = d.xquat[hand_id].copy()

    # Finger midpoint is about hand + [0.027, 0, -0.051] in this posture.
    hand_to_mid = np.array([0.027, 0.0, -0.051])

    def drive_hand(mid_target, grip_ctrl, steps, kp_pos=2.6, kp_rot=1.4, damp=5e-3):
        target = np.array(mid_target) - hand_to_mid
        for _ in range(steps):
            x_hand = d.xpos[hand_id].copy()
            q_hand = d.xquat[hand_id].copy()
            e_pos = target - x_hand
            e_rot = orientation_error(q_hand, q_des)
            task = np.concatenate([kp_pos * e_pos, kp_rot * e_rot])

            jacp = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jacp, jacr, hand_id)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            H = J.T @ J + damp * np.eye(7)
            dq = np.linalg.solve(H, J.T @ task)

            q_target = d.qpos[:7] + dq
            q_target = np.clip(q_target, m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1])
            d.ctrl[:7] = q_target
            d.ctrl[7] = grip_ctrl
            sim.step(1)

    cup = sim.cup_position().copy()

    # Stage 1: approach above cup
    drive_hand([cup[0], cup[1], cup[2] + 0.11], grip_ctrl=255.0, steps=1600)
    # Stage 2: align around cup center
    drive_hand([cup[0], cup[1], cup[2] + 0.01], grip_ctrl=255.0, steps=1400)
    # Stage 3: close while holding pose
    for g in np.linspace(255.0, 0.0, 1200):
        drive_hand([cup[0], cup[1], cup[2] + 0.005], grip_ctrl=float(g), steps=1, kp_pos=2.2, kp_rot=1.2)

    # Stage 4: lift
    drive_hand([cup[0], cup[1], cup[2] + 0.20], grip_ctrl=0.0, steps=1800)

    # Stage 5: hold to avoid transient success
    drive_hand([cup[0], cup[1], cup[2] + 0.20], grip_ctrl=0.0, steps=1000)

    # Early mandatory save (best-so-far; overwritten by later runs if better)
    sim.save_final_state(save_path)

    # Local estimate of grader metrics: 500-step settle with same final ctrl.
    finger_contact_steps = 0
    for _ in range(500):
        sim.step(1)
        touching = False
        for k in range(d.ncon):
            c = d.contact[k]
            b1 = m.geom_bodyid[c.geom1]
            b2 = m.geom_bodyid[c.geom2]
            pair = {b1, b2}
            if cup_id in pair and (left_id in pair or right_id in pair):
                touching = True
                break
        if touching:
            finger_contact_steps += 1

    final_cup = sim.cup_position().copy()
    return {
        "final_cup_z": float(final_cup[2]),
        "contact_fraction": float(finger_contact_steps / 500.0),
        "time": float(d.time),
    }


if __name__ == "__main__":
    out = run_attempt()
    print(out)
