import numpy as np
import mujoco
from sim import Sim


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def geom_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def set_arm_ik_step(sim, q_target, target_pos, gain=2.0, damping=1e-3, max_step=0.05):
    m, d = sim.model, sim.data
    hand_bid = body_id(m, "hand")

    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp, jacr, hand_bid)
    J = jacp[:, :7]

    pos = d.xpos[hand_bid].copy()
    err = target_pos - pos

    # damped least squares
    A = J @ J.T + damping * np.eye(3)
    dq = J.T @ np.linalg.solve(A, gain * err)
    dq = np.clip(dq, -max_step, max_step)

    q_target = q_target + dq
    for i in range(7):
        lo, hi = m.jnt_range[i]
        q_target[i] = np.clip(q_target[i], lo, hi)

    d.ctrl[:7] = q_target
    return q_target, np.linalg.norm(err)


def finger_contact_fraction(sim, steps=500):
    m, d = sim.model, sim.data
    cup_gid = geom_id(m, "cup_geom")
    left_gid = geom_id(m, "left_finger")
    right_gid = geom_id(m, "right_finger")

    hit = 0
    for _ in range(steps):
        d.ctrl[:] = d.ctrl
        sim.step(1)
        touched = False
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (cup_gid in (g1, g2)) and ((left_gid in (g1, g2)) or (right_gid in (g1, g2))):
                touched = True
                break
        if touched:
            hit += 1
    return hit / steps


def run_attempt(save=True):
    sim = Sim()
    m, d = sim.model, sim.data
    q_target = np.zeros(7)

    # Open gripper first
    d.ctrl[:7] = q_target
    d.ctrl[7] = 255
    sim.step(300)

    phases = [
        # (target xyz, gripper ctrl, steps)
        (np.array([0.50, 0.00, 0.58]), 255, 500),
        (np.array([0.50, 0.00, 0.52]), 255, 500),
        (np.array([0.50, 0.00, 0.50]), 200, 150),
        (np.array([0.50, 0.00, 0.495]), 100, 150),
        (np.array([0.50, 0.00, 0.495]), 30, 200),
        (np.array([0.50, 0.00, 0.54]), 0, 350),
        (np.array([0.50, 0.00, 0.60]), 0, 450),
    ]

    for target, grip, n in phases:
        for _ in range(n):
            q_target, _ = set_arm_ik_step(sim, q_target, target)
            d.ctrl[7] = grip
            sim.step(1)

    # Hold to stabilize before saving
    for _ in range(250):
        q_target, _ = set_arm_ik_step(sim, q_target, np.array([0.50, 0.00, 0.62]), gain=1.5)
        d.ctrl[7] = 0
        sim.step(1)

    cup_z = sim.cup_position()[2]
    # approximate settle contact in current trajectory context
    contact_frac = finger_contact_fraction(sim, steps=200)

    if save:
        sim.save_final_state('/work/final_state.npz')

    return cup_z, contact_frac, d.qpos[:9].copy(), sim.cup_position().copy()


if __name__ == '__main__':
    cup_z, contact_frac, qpos, cup_pos = run_attempt(save=True)
    print('cup_pos', cup_pos)
    print('cup_z', cup_z)
    print('contact_frac_est', contact_frac)
    print('qpos[:9]', qpos)
