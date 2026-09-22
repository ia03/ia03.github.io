import numpy as np
import mujoco
from sim import Sim


def run_attempt(save_path='/work/final_state.npz', render_debug=False):
    sim = Sim()
    m, d = sim.model, sim.data
    hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    ctrl_lo = m.actuator_ctrlrange[:7, 0].copy()
    ctrl_hi = m.actuator_ctrlrange[:7, 1].copy()

    def servo_to_pos(target, grip, steps=500, tol=0.006, gain=0.9, damping=1e-4):
        reached = 0
        for _ in range(steps):
            pos = d.xpos[hand_id].copy()
            err = target - pos
            if np.linalg.norm(err) < tol:
                reached += 1
            else:
                reached = 0

            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jacp, None, hand_id)
            J = jacp[:, :7]
            A = J @ J.T + damping * np.eye(3)
            v = np.linalg.solve(A, err * gain)
            dq = J.T @ v
            q_des = np.clip(d.qpos[:7] + dq, ctrl_lo, ctrl_hi)

            d.ctrl[:7] = q_des
            d.ctrl[7] = grip
            sim.step(1)

            if reached >= 25:
                break

    def hold(target, grip, steps=200):
        for _ in range(steps):
            d.ctrl[:7] = np.clip(d.ctrl[:7], ctrl_lo, ctrl_hi)
            d.ctrl[7] = grip
            # keep closed-loop corrections while holding
            pos = d.xpos[hand_id].copy()
            err = target - pos
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jacp, None, hand_id)
            J = jacp[:, :7]
            A = J @ J.T + 1e-4 * np.eye(3)
            v = np.linalg.solve(A, err * 0.8)
            dq = J.T @ v
            q_des = np.clip(d.qpos[:7] + dq, ctrl_lo, ctrl_hi)
            d.ctrl[:7] = q_des
            sim.step(1)

    # Open gripper and stabilize.
    for _ in range(120):
        d.ctrl[:7] = np.clip(d.qpos[:7], ctrl_lo, ctrl_hi)
        d.ctrl[7] = 255.0
        sim.step(1)

    # Approach from side of wall (y > wall extent), then grasp.
    servo_to_pos(np.array([0.52, 0.15, 0.62]), grip=255.0, steps=500)
    servo_to_pos(np.array([0.55, 0.15, 0.53]), grip=255.0, steps=450)
    servo_to_pos(np.array([0.55, 0.15, 0.50]), grip=255.0, steps=350)

    # Close fingers while maintaining pose.
    for g in np.linspace(255, 0, 220):
        pos = d.xpos[hand_id].copy()
        err = np.array([0.55, 0.15, 0.50]) - pos
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, None, hand_id)
        J = jacp[:, :7]
        A = J @ J.T + 1e-4 * np.eye(3)
        v = np.linalg.solve(A, err * 0.8)
        dq = J.T @ v
        d.ctrl[:7] = np.clip(d.qpos[:7] + dq, ctrl_lo, ctrl_hi)
        d.ctrl[7] = float(g)
        sim.step(1)

    # Lift and return to robot side.
    servo_to_pos(np.array([0.50, 0.15, 0.67]), grip=0.0, steps=700)
    servo_to_pos(np.array([0.42, 0.14, 0.67]), grip=0.0, steps=700)
    servo_to_pos(np.array([0.38, 0.10, 0.66]), grip=0.0, steps=700)

    # Let behavior play out and settle while held.
    hold(np.array([0.36, 0.10, 0.66]), grip=0.0, steps=900)

    cup = sim.cup_position()
    print('final cup', cup, 'contact', sim.has_gripper_cup_contact(), 'steps', len(sim._ctrl_trace))

    if render_debug:
        img = sim.render(640, 480)
        from imageio.v3 import imwrite
        imwrite('/work/last_frame.png', img)

    sim.save_final_state(save_path)
    return cup


if __name__ == '__main__':
    run_attempt()
