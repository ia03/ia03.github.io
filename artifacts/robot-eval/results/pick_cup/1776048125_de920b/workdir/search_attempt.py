import numpy as np
import mujoco
from sim import Sim


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def ik_q_for_midpoint(target, q_seed):
    sim = Sim()
    m, d = sim.model, sim.data
    left = body_id(m, 'left_finger')
    right = body_id(m, 'right_finger')
    q = q_seed.copy()
    qlow = m.actuator_ctrlrange[:7, 0]
    qhi = m.actuator_ctrlrange[:7, 1]
    for _ in range(400):
        d.qpos[:7] = q
        d.qpos[7] = 0.03
        d.qpos[8] = 0.03
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        pm = 0.5 * (d.xpos[left] + d.xpos[right])
        err = target - pm
        if np.linalg.norm(err) < 2e-4:
            break
        jl = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jl, None, left)
        mujoco.mj_jacBody(m, d, jr, None, right)
        J = 0.5 * (jl + jr)
        J7 = J[:, :7]
        dq = J7.T @ np.linalg.solve(J7 @ J7.T + 1e-4*np.eye(3), 1.0 * err)
        q = np.clip(q + np.clip(dq, -0.05, 0.05), qlow, qhi)
    return q


def eval_trace(trace):
    sim = Sim()
    m, d = sim.model, sim.data
    left = body_id(m, 'left_finger')
    right = body_id(m, 'right_finger')
    cup = body_id(m, 'cup')

    for u in trace:
        d.ctrl[:] = u
        sim.step(1)
    ulast = trace[-1]
    good = 0
    for _ in range(500):
        d.ctrl[:] = ulast
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
    hp = np.clip((z - 0.435) / (0.085), 0.0, 1.0)
    cp = np.clip(frac / 0.5, 0.0, 1.0)
    return z, frac, 0.5*hp + 0.5*cp


def run_trial(params, save=False):
    sim = Sim(); d = sim.data
    cup = sim.cup_position().copy()

    q0 = np.array([0.0, 0.32, 0.0, -1.88, 0.0, 3.06, 0.80])
    q_above = ik_q_for_midpoint(cup + np.array([0, 0, params['z_above']]), q0)
    q_pre = ik_q_for_midpoint(cup + np.array([0, 0, params['z_pre']]), q_above)
    q_grasp = ik_q_for_midpoint(cup + np.array([0, 0, params['z_grasp']]), q_pre)
    q_lift = ik_q_for_midpoint(cup + np.array([0, 0, params['z_lift']]), q_pre)

    def hold(q, grip, n):
        d.ctrl[:7] = q
        d.ctrl[7] = grip
        sim.step(n)

    hold(np.zeros(7), 255.0, 120)
    hold(q_above, params['g_open'], params['n_above'])
    hold(q_pre, params['g_pre'], params['n_pre'])
    hold(q_grasp, params['g_desc'], params['n_desc'])

    for g in np.linspace(params['g_desc'], params['g_close'], params['n_close']):
        hold(q_grasp, float(g), 1)
    hold(q_grasp, params['g_close'], params['n_squeeze'])
    hold(q_lift, params['g_close'], params['n_lift'])
    hold(q_lift, params['g_close'], params['n_hold'])

    trace = np.array(sim._ctrl_trace)
    z, frac, score = eval_trace(trace)
    if save:
        sim.save_final_state('/work/final_state.npz')
    return score, z, frac, trace


def main():
    best = (-1, None, None, None, None)
    rng = np.random.default_rng(0)

    base = {
        'z_above': 0.11, 'z_pre': 0.06, 'z_grasp': 0.028, 'z_lift': 0.20,
        'g_open': 210.0, 'g_pre': 160.0, 'g_desc': 140.0, 'g_close': 35.0,
        'n_above': 700, 'n_pre': 550, 'n_desc': 650, 'n_close': 250,
        'n_squeeze': 300, 'n_lift': 850, 'n_hold': 800,
    }

    candidates = [base]
    for _ in range(35):
        p = base.copy()
        p['z_above'] = float(rng.uniform(0.09, 0.14))
        p['z_pre'] = float(rng.uniform(0.045, 0.08))
        p['z_grasp'] = float(rng.uniform(0.01, 0.04))
        p['z_lift'] = float(rng.uniform(0.16, 0.24))
        p['g_open'] = float(rng.uniform(170, 255))
        p['g_pre'] = float(rng.uniform(120, 190))
        p['g_desc'] = float(rng.uniform(90, 170))
        p['g_close'] = float(rng.uniform(0, 70))
        p['n_above'] = int(rng.integers(450, 900))
        p['n_pre'] = int(rng.integers(350, 800))
        p['n_desc'] = int(rng.integers(450, 850))
        p['n_close'] = int(rng.integers(150, 420))
        p['n_squeeze'] = int(rng.integers(180, 500))
        p['n_lift'] = int(rng.integers(650, 1000))
        p['n_hold'] = int(rng.integers(500, 1000))
        candidates.append(p)

    for i, p in enumerate(candidates):
        score, z, frac, trace = run_trial(p, save=False)
        print(f"trial {i:02d} score={score:.3f} z={z:.3f} contact={frac:.3f}")
        if score > best[0]:
            best = (score, z, frac, p, trace)
            # Save best so far immediately
            sim = Sim(); d=sim.data
            for u in trace:
                d.ctrl[:] = u
                sim.step(1)
            sim.save_final_state('/work/final_state.npz')
            print('  saved new best')
            if z >= 0.52 and frac >= 0.5:
                break

    print('BEST', best[0], best[1], best[2], best[3])


if __name__ == '__main__':
    main()
