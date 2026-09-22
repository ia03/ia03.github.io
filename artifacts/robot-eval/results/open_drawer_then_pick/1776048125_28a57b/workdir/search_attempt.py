import time
import numpy as np
import mujoco
from sim import Sim

START_TS = time.time()
DEADLINE_TS = START_TS + 540  # keep margin


def solve_q_for_hand(sim, target, q_seed, iters=220):
    m, d = sim.model, sim.data
    hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    q = q_seed.copy()
    qmin = m.jnt_range[:7, 0]
    qmax = m.jnt_range[:7, 1]
    qpos_save = d.qpos.copy()
    qvel_save = d.qvel.copy()
    try:
        for _ in range(iters):
            d.qpos[:7] = q
            d.qvel[:] = 0
            mujoco.mj_fwdPosition(m, d)
            x = d.xpos[hand_id].copy()
            err = target - x
            if np.linalg.norm(err) < 1e-4:
                break
            jac = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jac, None, hand_id)
            J = jac[:, :7]
            dq = J.T @ np.linalg.solve(J @ J.T + 5e-5 * np.eye(3), 0.9 * err)
            q = np.clip(q + dq, qmin, qmax)
    finally:
        d.qpos[:] = qpos_save
        d.qvel[:] = qvel_save
        mujoco.mj_forward(m, d)
    return q


def run_trial(p, save_path=None):
    sim = Sim()
    m, d = sim.model, sim.data
    q_home = d.qpos[:7].copy()
    lo = m.actuator_ctrlrange[:7, 0]
    hi = m.actuator_ctrlrange[:7, 1]

    # Stage 1: guaranteed drawer opening first
    for _ in range(260):
        d.ctrl[:7] = np.clip(q_home, lo, hi)
        d.ctrl[7] = 255.0
        d.ctrl[8] = 1.0
        sim.step()

    b = sim.block_position().copy()

    pre = np.array([b[0] + p['pre_dx'], b[1] + p['pre_dy'], b[2] + p['pre_dz']])
    grasp = np.array([b[0] + p['grasp_dx'], b[1] + p['grasp_dy'], b[2] + p['grasp_dz']])
    lift = np.array([b[0] + p['lift_dx'], b[1] + p['lift_dy'], p['lift_z']])

    q_pre = solve_q_for_hand(sim, pre, q_home)
    q_grasp = solve_q_for_hand(sim, grasp, q_pre)
    q_lift = solve_q_for_hand(sim, lift, q_grasp)

    replay_drawer = []
    replay_bz = []

    def step_stage(q_a, q_b, n, grip_a, grip_b):
        for i in range(n):
            t = (i + 1) / n
            q = (1 - t) * q_a + t * q_b
            grip = (1 - t) * grip_a + t * grip_b
            d.ctrl[:7] = np.clip(q, lo, hi)
            d.ctrl[7] = float(np.clip(grip, 0, 255))
            d.ctrl[8] = 1.0
            sim.step()
            replay_drawer.append(sim.drawer_open_amount())
            replay_bz.append(sim.block_position()[2])

    step_stage(q_home, q_pre, 220, 255, 255)
    step_stage(q_pre, q_grasp, 220, 255, 255)
    step_stage(q_grasp, q_grasp, 220, 255, p['close_val'])
    step_stage(q_grasp, q_lift, 300, p['close_val'], p['close_val'])
    step_stage(q_lift, q_lift, 360, p['close_val'], p['close_val'])

    replay_drawer = np.asarray(replay_drawer)
    replay_bz = np.asarray(replay_bz)

    # settle
    contact_count = 0
    for _ in range(500):
        sim.step()
        contact_count += 1 if sim.has_gripper_block_contact() else 0

    final_drawer = sim.drawer_open_amount()
    final_bz = sim.block_position()[2]
    contact_frac = contact_count / 500.0

    # order check
    idx_retrieval = np.where(replay_bz >= 0.50)[0]
    if len(idx_retrieval) == 0:
        order_ok = False
    else:
        first_r = idx_retrieval[0]
        order_ok = np.max(replay_drawer[:first_r + 1]) >= 0.05

    pass_cond = (
        len(sim._ctrl_trace) >= 20 and
        order_ok and
        final_drawer >= 0.05 and
        final_bz >= 0.595 and
        contact_frac >= 0.4
    )

    drawer_progress = np.clip(np.max(replay_drawer) / 0.05, 0, 1)
    retrieval_progress = np.clip((np.max(replay_bz) - 0.435) / (0.50 - 0.435), 0, 1)
    order_progress = 1.0 if order_ok else 0.0
    ordered_retrieval_progress = order_progress * retrieval_progress
    final_height_progress = np.clip((final_bz - 0.43) / (0.595 - 0.43), 0, 1)
    hold_progress = np.clip(contact_frac / 0.4, 0, 1)
    final_drawer_progress = np.clip(final_drawer / 0.05, 0, 1)
    score = (0.22*drawer_progress + 0.28*ordered_retrieval_progress + 0.22*final_height_progress +
             0.18*hold_progress + 0.10*final_drawer_progress)

    result = {
        'pass': pass_cond,
        'score': float(score),
        'final_bz': float(final_bz),
        'final_drawer': float(final_drawer),
        'contact_frac': float(contact_frac),
        'max_bz': float(np.max(replay_bz)),
        'max_drawer': float(np.max(replay_drawer)),
        'params': p,
    }
    if save_path is not None:
        sim.save_final_state(save_path)
    return result


def sample_params(rng):
    return {
        'pre_dx': rng.uniform(-0.04, 0.03),
        'pre_dy': rng.uniform(-0.03, 0.03),
        'pre_dz': rng.uniform(0.09, 0.16),
        'grasp_dx': rng.uniform(-0.03, 0.025),
        'grasp_dy': rng.uniform(-0.02, 0.02),
        'grasp_dz': rng.uniform(0.04, 0.085),
        'lift_dx': rng.uniform(-0.01, 0.03),
        'lift_dy': rng.uniform(-0.02, 0.02),
        'lift_z': rng.uniform(0.66, 0.80),
        'close_val': rng.uniform(0.0, 25.0),
    }


if __name__ == '__main__':
    rng = np.random.default_rng(0)

    # save a plausible early run immediately
    seed_params = {
        'pre_dx': -0.01, 'pre_dy': 0.0, 'pre_dz': 0.14,
        'grasp_dx': -0.005, 'grasp_dy': 0.0, 'grasp_dz': 0.060,
        'lift_dx': 0.0, 'lift_dy': 0.0, 'lift_z': 0.72,
        'close_val': 0.0,
    }
    best = run_trial(seed_params, save_path='/work/final_state.npz')
    print('seed', best)

    for i in range(80):
        if time.time() > DEADLINE_TS:
            break
        p = sample_params(rng)
        res = run_trial(p)
        if res['score'] > best['score']:
            best = res
            run_trial(p, save_path='/work/final_state.npz')
            print('improved', i, best)
        elif i % 10 == 0:
            print('trial', i, res)
        if best['pass']:
            print('PASS found at', i, best)
            run_trial(best['params'], save_path='/work/final_state.npz')
            break

    print('best_final', best)
