import math
import time
import numpy as np
from sim import Sim, TARGET_XY, TOUCH_POINT

# Fixed PD gains for torque actuators
KP = np.array([260,220,280,340,150,260,220,280,340,150,260,120,90,70,70,120,90,70,70], dtype=float)
KD = np.array([12,10,14,16,8,12,10,14,16,8,12,6,5,4,4,6,5,4,4], dtype=float)

rng = np.random.default_rng(7)


def run_candidate(params, save_path=None):
    s = Sim()
    m = s.model
    home = s.home_ctrl().copy()
    qidx = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])
    vidx = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])

    # phase lengths
    n1 = int(params['n1'])
    n2 = int(params['n2'])
    n = n1 + n2

    # symmetric leg offsets + torso
    q1 = home.copy()
    q1[2] += params['hip1']
    q1[7] += params['hip1']
    q1[3] += params['knee1']
    q1[8] += params['knee1']
    q1[4] += params['ank1']
    q1[9] += params['ank1']
    q1[10] += params['torso1']

    q2 = q1.copy()
    q2[2] += params['hip2_delta']
    q2[7] += params['hip2_delta']
    q2[3] += params['knee2_delta']
    q2[8] += params['knee2_delta']
    q2[4] += params['ank2_delta']
    q2[9] += params['ank2_delta']
    q2[10] += params['torso2_delta']

    # right arm reach
    q2[15] += params['rsp']
    q2[16] += params['rsr']
    q2[17] += params['rsy']
    q2[18] += params['re']
    # left arm balancing / optional reach assist
    q2[11] += params['lsp']
    q2[12] += params['lsr']
    q2[13] += params['lsy']
    q2[14] += params['le']

    pelvis_z_ok = 0
    torso_ok = 0
    checkpoints = [0.25, 0.55, 0.80]
    hit = [False, False, False]
    best_touch = 1e9

    for t in range(n):
        alpha = 0.0 if t < n1 else min(1.0, (t - n1) / max(1, int(params['blend'])))
        qdes = (1.0 - alpha) * q1 + alpha * q2

        # tiny stepping oscillation to avoid static stalling
        if t < n1:
            ph = 2.0 * math.pi * t / max(1.0, params['period'])
            qdes = qdes.copy()
            qdes[2] += params['gait_hip'] * math.sin(ph)
            qdes[7] -= params['gait_hip'] * math.sin(ph)
            qdes[3] += params['gait_knee'] * max(0.0, math.sin(ph))
            qdes[8] += params['gait_knee'] * max(0.0, -math.sin(ph))
            qdes[4] += params['gait_ank'] * math.sin(ph)
            qdes[9] -= params['gait_ank'] * math.sin(ph)

        q = s.data.qpos[qidx]
        v = s.data.qvel[vidx]
        tau = KP * (qdes - q) - KD * v
        s.data.ctrl[:] = tau
        s.step()

        p = s.pelvis_position()
        tu = s.torso_up()
        d_touch = s.distance_to_touch()
        best_touch = min(best_touch, d_touch)
        for i, c in enumerate(checkpoints):
            if p[0] >= c:
                hit[i] = True
        if p[2] >= 0.23 and tu >= 0.55:
            pelvis_z_ok += 1

    stable_frac = pelvis_z_ok / n

    # Settle check with final control held
    min_settle_z = 1e9
    min_settle_up = 1e9
    settle_best_touch = 1e9
    for _ in range(500):
        s.step()
        p = s.pelvis_position()
        min_settle_z = min(min_settle_z, p[2])
        tu = s.torso_up()
        min_settle_up = min(min_settle_up, tu)
        settle_best_touch = min(settle_best_touch, s.distance_to_touch())

    final_dist = float(np.linalg.norm(s.pelvis_position()[:2] - TARGET_XY))

    passed = (
        n >= 30
        and all(hit)
        and stable_frac >= 0.35
        and best_touch <= 0.12
        and final_dist <= 0.45
        and min_settle_z >= 0.23
        and min_settle_up >= 0.55
        and settle_best_touch <= 0.12
    )

    # Smooth objective similar to progress_score, with hard penalty on stability/settle
    checkpoint_progress = sum(hit) / 3.0
    touch_progress = 1.0 if best_touch <= 0.12 else max(0.0, 1.0 - (best_touch - 0.12) / 1.2)
    stability_progress = min(1.0, stable_frac / 0.35)
    distance_progress = 1.0 if final_dist <= 0.45 else max(0.0, 1.0 - (final_dist - 0.45) / 1.2)
    settle_posture = 0.5 * min(1.0, min_settle_z / 0.23) + 0.5 * min(1.0, min_settle_up / 0.55)
    settle_touch_progress = 1.0 if settle_best_touch <= 0.12 else max(0.0, 1.0 - (settle_best_touch - 0.12) / 1.2)
    score = (
        0.2 * checkpoint_progress
        + 0.35 * touch_progress
        + 0.15 * settle_touch_progress
        + 0.15 * stability_progress
        + 0.10 * distance_progress
        + 0.05 * settle_posture
    )

    stats = {
        'passed': passed,
        'score': score,
        'hit': hit,
        'stable_frac': stable_frac,
        'best_touch': best_touch,
        'settle_best_touch': settle_best_touch,
        'final_dist': final_dist,
        'min_settle_z': min_settle_z,
        'min_settle_up': min_settle_up,
        'final_pelvis': s.pelvis_position().copy(),
    }

    if save_path is not None:
        s.save_final_state(save_path)

    return stats


def sample_params(best=None, scale=1.0):
    if best is None:
        p = {
            'n1': int(rng.integers(1100, 1900)),
            'n2': int(rng.integers(350, 950)),
            'hip1': rng.uniform(0.15, 0.75),
            'knee1': rng.uniform(-0.35, 0.25),
            'ank1': rng.uniform(-0.35, 0.1),
            'torso1': rng.uniform(-0.2, 0.35),
            'hip2_delta': rng.uniform(-0.25, 0.25),
            'knee2_delta': rng.uniform(-0.35, 0.35),
            'ank2_delta': rng.uniform(-0.25, 0.25),
            'torso2_delta': rng.uniform(-0.35, 0.35),
            'rsp': rng.uniform(-1.8, 1.2),
            'rsr': rng.uniform(-1.4, 1.8),
            'rsy': rng.uniform(-1.3, 1.3),
            're': rng.uniform(-1.6, 1.6),
            'lsp': rng.uniform(-0.8, 0.8),
            'lsr': rng.uniform(-0.8, 0.8),
            'lsy': rng.uniform(-0.8, 0.8),
            'le': rng.uniform(-0.8, 0.8),
            'blend': int(rng.integers(80, 360)),
            'period': float(rng.uniform(140, 420)),
            'gait_hip': rng.uniform(0.0, 0.35),
            'gait_knee': rng.uniform(0.0, 0.35),
            'gait_ank': rng.uniform(-0.20, 0.20),
        }
        return p

    p = dict(best)
    for k in ['hip1','knee1','ank1','torso1','hip2_delta','knee2_delta','ank2_delta','torso2_delta',
              'rsp','rsr','rsy','re','lsp','lsr','lsy','le','period','gait_hip','gait_knee','gait_ank']:
        sigma = {
            'hip1':0.10,'knee1':0.10,'ank1':0.08,'torso1':0.08,
            'hip2_delta':0.08,'knee2_delta':0.10,'ank2_delta':0.08,'torso2_delta':0.10,
            'rsp':0.35,'rsr':0.35,'rsy':0.30,'re':0.30,'lsp':0.25,'lsr':0.25,'lsy':0.25,'le':0.25,
            'period':45.0,'gait_hip':0.08,'gait_knee':0.08,'gait_ank':0.06,
        }[k] * scale
        p[k] = float(p[k] + rng.normal(0, sigma))

    p['n1'] = int(np.clip(int(best['n1'] + rng.normal(0, 120 * scale)), 1000, 2200))
    p['n2'] = int(np.clip(int(best['n2'] + rng.normal(0, 120 * scale)), 280, 1200))
    p['blend'] = int(np.clip(int(best['blend'] + rng.normal(0, 50 * scale)), 40, 420))

    # clamp ranges
    p['hip1'] = float(np.clip(p['hip1'], -0.2, 1.0))
    p['knee1'] = float(np.clip(p['knee1'], -0.6, 0.5))
    p['ank1'] = float(np.clip(p['ank1'], -0.6, 0.35))
    p['torso1'] = float(np.clip(p['torso1'], -0.5, 0.6))
    p['hip2_delta'] = float(np.clip(p['hip2_delta'], -0.45, 0.45))
    p['knee2_delta'] = float(np.clip(p['knee2_delta'], -0.6, 0.6))
    p['ank2_delta'] = float(np.clip(p['ank2_delta'], -0.45, 0.45))
    p['torso2_delta'] = float(np.clip(p['torso2_delta'], -0.6, 0.6))
    for k in ['rsp','rsr']:
        p[k] = float(np.clip(p[k], -2.4, 2.4))
    for k in ['rsy','re']:
        p[k] = float(np.clip(p[k], -2.0, 2.0))
    for k in ['lsp','lsr','lsy','le']:
        p[k] = float(np.clip(p[k], -1.6, 1.6))
    p['period'] = float(np.clip(p['period'], 100, 520))
    p['gait_hip'] = float(np.clip(p['gait_hip'], 0.0, 0.5))
    p['gait_knee'] = float(np.clip(p['gait_knee'], 0.0, 0.5))
    p['gait_ank'] = float(np.clip(p['gait_ank'], -0.35, 0.35))
    return p


def main():
    start = time.time()
    budget_s = 640.0
    best_p = None
    best_stats = None

    # Start from known good forward posture seed
    seed = {
        'n1': 1500,
        'n2': 700,
        'hip1': 0.50,
        'knee1': -0.20,
        'ank1': -0.15,
        'torso1': 0.0,
        'hip2_delta': 0.0,
        'knee2_delta': 0.0,
        'ank2_delta': 0.0,
        'torso2_delta': 0.0,
        'rsp': 0.0,
        'rsr': 0.8,
        'rsy': -0.4,
        're': 0.8,
        'lsp': 0.0,
        'lsr': 0.0,
        'lsy': 0.0,
        'le': 0.0,
        'blend': 140,
        'period': 260.0,
        'gait_hip': 0.08,
        'gait_knee': 0.12,
        'gait_ank': -0.05,
    }

    stats = run_candidate(seed, save_path='/work/final_state.npz')
    best_p, best_stats = seed, stats
    print('seed',stats)

    i = 0
    while time.time() - start < budget_s:
        i += 1
        if i < 20:
            p = sample_params(None)
        else:
            scale = 1.0 if i < 80 else 0.6 if i < 150 else 0.35
            p = sample_params(best_p, scale=scale)

        st = run_candidate(p)
        if (st['score'] > best_stats['score']) or (st['passed'] and not best_stats['passed']):
            best_p, best_stats = p, st
            run_candidate(best_p, save_path='/work/final_state.npz')
            print('iter', i, 'NEW BEST', best_stats)
            print('params', best_p)

        if i % 10 == 0:
            elapsed = time.time() - start
            print(f'iter {i} elapsed {elapsed:.1f}s best_score {best_stats["score"]:.3f} pass {best_stats["passed"]} touch {best_stats["best_touch"]:.3f}/{best_stats["settle_best_touch"]:.3f} final_dist {best_stats["final_dist"]:.3f}')

        if best_stats['passed']:
            print('Found passing candidate; saved and stopping early.')
            break

    print('final best stats', best_stats)
    print('final best params', best_p)


if __name__ == '__main__':
    main()
