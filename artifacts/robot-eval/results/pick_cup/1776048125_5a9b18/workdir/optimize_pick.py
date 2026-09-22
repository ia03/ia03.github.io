import numpy as np
import mujoco
from sim import Sim

SUCCESS_Z = 0.52


def ids(m):
    return {
        'hand': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'hand'),
        'cup': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'cup'),
        'left_finger': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'left_finger'),
        'right_finger': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_finger'),
    }


def rot_error(R, Rd):
    # small-angle orientation error in world frame
    return 0.5 * (
        np.cross(R[:, 0], Rd[:, 0])
        + np.cross(R[:, 1], Rd[:, 1])
        + np.cross(R[:, 2], Rd[:, 2])
    )


def solve_ik(sim, q_init7, target_pos, target_R, niter=200, wp=1.0, wr=0.4):
    m, d = sim.model, sim.data
    I = ids(m)
    q = q_init7.copy()

    for _ in range(niter):
        d.qpos[:7] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

        pos = d.xpos[I['hand']].copy()
        R = d.xmat[I['hand']].reshape(3, 3).copy()
        ep = target_pos - pos
        er = rot_error(R, target_R)
        e = np.concatenate([wp * ep, wr * er])

        if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 2e-3:
            break

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, I['hand'])
        J = np.vstack([wp * jacp[:, :7], wr * jacr[:, :7]])

        lam = 5e-3
        A = J @ J.T + lam * np.eye(6)
        dq = J.T @ np.linalg.solve(A, e)
        dq = np.clip(dq, -0.04, 0.04)

        q += dq
        for j in range(7):
            lo, hi = m.jnt_range[j]
            q[j] = np.clip(q[j], lo, hi)

    d.qpos[:7] = q
    mujoco.mj_forward(m, d)
    return q


def run_traj(params):
    sim = Sim()
    m, d = sim.model, sim.data
    I = ids(m)

    # Keep reference hand orientation from start (fingers downward)
    R0 = d.xmat[I['hand']].reshape(3, 3).copy()

    q0 = np.zeros(7)
    q_above = solve_ik(sim, q0, np.array([0.50, 0.00, params['z_above']]), R0)
    q_pre = solve_ik(sim, q_above, np.array([0.50, 0.00, params['z_pre']]), R0)
    q_grasp = solve_ik(sim, q_pre, np.array([0.50, 0.00, params['z_grasp']]), R0)
    q_lift = solve_ik(sim, q_grasp, np.array([0.50, 0.00, params['z_lift']]), R0)

    d.ctrl[:7] = q0
    d.ctrl[7] = 255
    sim.step(150)

    def interp(q_from, q_to, steps, g_from, g_to):
        for t in range(steps):
            a = (t + 1) / steps
            d.ctrl[:7] = (1 - a) * q_from + a * q_to
            d.ctrl[7] = (1 - a) * g_from + a * g_to
            sim.step(1)

    interp(q0, q_above, params['n1'], 255, 255)
    interp(q_above, q_pre, params['n2'], 255, 255)
    interp(q_pre, q_grasp, params['n3'], 255, 220)
    interp(q_grasp, q_grasp, params['nclose'], 220, params['g_close'])
    interp(q_grasp, q_grasp, params['nhold'], params['g_close'], params['g_close'])
    interp(q_grasp, q_lift, params['nlift'], params['g_close'], params['g_close'])
    interp(q_lift, q_lift, params['nfinal'], params['g_close'], params['g_close'])

    ctrl_trace = np.array(sim._ctrl_trace, dtype=float)
    return sim, ctrl_trace


def replay_eval(ctrl_trace):
    sim = Sim()
    m, d = sim.model, sim.data
    I = ids(m)
    cup_bid = I['cup']
    finger_bodies = {I['left_finger'], I['right_finger']}

    for u in ctrl_trace:
        d.ctrl[:] = u
        sim.step(1)

    hit = 0
    for _ in range(500):
        # hold last command during settle
        sim.step(1)
        touched = False
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = m.geom_bodyid[c.geom1]
            b2 = m.geom_bodyid[c.geom2]
            if (cup_bid in (b1, b2)) and ((b1 in finger_bodies) or (b2 in finger_bodies)):
                touched = True
                break
        if touched:
            hit += 1

    cup_z = sim.cup_position()[2]
    contact_frac = hit / 500.0
    height_progress = np.clip((cup_z - 0.435) / (0.52 - 0.435), 0.0, 1.0)
    contact_progress = np.clip(contact_frac / 0.5, 0.0, 1.0)
    score = 0.5 * height_progress + 0.5 * contact_progress
    success = cup_z >= 0.52 and contact_frac >= 0.5
    return {
        'cup_z': float(cup_z),
        'contact_frac': float(contact_frac),
        'score': float(score),
        'success': bool(success),
    }


def main():
    trials = []
    for z_grasp in [0.505, 0.500, 0.495, 0.490, 0.485]:
        for g_close in [0, 5, 10, 15, 20]:
            trials.append({
                'z_above': 0.61,
                'z_pre': 0.545,
                'z_grasp': z_grasp,
                'z_lift': 0.66,
                'n1': 500,
                'n2': 350,
                'n3': 220,
                'nclose': 250,
                'nhold': 120,
                'nlift': 520,
                'nfinal': 120,
                'g_close': g_close,
            })

    best = None
    for i, p in enumerate(trials, 1):
        sim, ctrl_trace = run_traj(p)
        ev = replay_eval(ctrl_trace)
        if best is None or ev['score'] > best[0]['score']:
            sim.save_final_state('/work/final_state.npz')
            best = (ev, p)
            tag = 'BEST'
        else:
            tag = '----'
        print(f"trial {i:02d}/{len(trials)} {tag} score={ev['score']:.3f} z={ev['cup_z']:.3f} cf={ev['contact_frac']:.3f} success={ev['success']} p={p}")

        if ev['success']:
            print('SUCCESS FOUND')
            sim.save_final_state('/work/final_state.npz')
            break

    if best is not None:
        print('best', best[0], best[1])


if __name__ == '__main__':
    main()
