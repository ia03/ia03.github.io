import numpy as np
import mujoco
from sim import Sim


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def run_policy(save=True):
    sim = Sim()
    d = sim.data

    # Waypoints from kinematic IK for finger-midpoint targets around cup centerline
    q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_above = np.array([0.0, -0.4587, 0.0, -2.4359, 0.0, 2.8323, 0.8])
    q_pre = np.array([0.0, -0.4182, 0.0, -2.4907, 0.0, 2.8144, 0.8])
    q_grasp = np.array([0.0, -0.3776, 0.0, -2.5154, 0.0, 2.8069, 0.8])
    q_lift = np.array([0.0, -0.4939, 0.0, -2.3708, 0.0, 2.8541, 0.8])

    def hold(q, grip, n):
        d.ctrl[:7] = q
        d.ctrl[7] = grip
        sim.step(n)

    # Stabilize and open gripper
    hold(q_home, 255.0, 250)

    # Approach cup from above with open gripper
    hold(q_above, 255.0, 900)
    hold(q_pre, 255.0, 700)
    hold(q_grasp, 255.0, 700)

    # Close while maintaining grasp pose
    for g in np.linspace(255.0, 0.0, 450):
        hold(q_grasp, float(g), 1)
    hold(q_grasp, 0.0, 350)

    # Lift and hold
    hold(q_lift, 0.0, 900)
    hold(q_lift, 0.0, 1000)

    if save:
        sim.save_final_state('/work/final_state.npz')

    return sim


def replay_eval(npz_path='/work/final_state.npz'):
    pack = np.load(npz_path)
    trace = pack['ctrl_trace']

    sim = Sim()
    m, d = sim.model, sim.data
    left = body_id(m, 'left_finger')
    right = body_id(m, 'right_finger')
    cup = body_id(m, 'cup')

    for u in trace:
        d.ctrl[:] = u
        sim.step(1)

    good = 0
    u_last = trace[-1] if len(trace) else np.zeros(m.nu)
    for _ in range(500):
        d.ctrl[:] = u_last
        sim.step(1)
        touching = False
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = m.geom_bodyid[c.geom1]
            b2 = m.geom_bodyid[c.geom2]
            if ((b1 in (left, right) and b2 == cup) or (b2 in (left, right) and b1 == cup)):
                touching = True
                break
        if touching:
            good += 1

    z = sim.cup_position()[2]
    frac = good / 500.0
    height_progress = np.clip((z - 0.435) / (0.52 - 0.435), 0.0, 1.0)
    contact_progress = np.clip(frac / 0.5, 0.0, 1.0)
    score = 0.5 * height_progress + 0.5 * contact_progress

    return {
        'final_z_after_settle': float(z),
        'contact_fraction': float(frac),
        'progress_score': float(score),
        'pass_like': bool(z >= 0.52 and frac >= 0.5),
        'steps': int(len(trace)),
    }


if __name__ == '__main__':
    sim = run_policy(save=True)
    print('cup_now', sim.cup_position())
    print(replay_eval('/work/final_state.npz'))
