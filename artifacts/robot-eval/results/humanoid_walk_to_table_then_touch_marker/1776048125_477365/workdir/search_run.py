import time
import math
import numpy as np
from sim import Sim, TOUCH_POINT, TARGET_XY

DT = 0.002


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def make_policy(p, home):
    def policy(step, pelvis_x):
        t = step * DT
        phase = 2.0 * math.pi * p['freq'] * t + p['phase0']
        s = math.sin(phase)
        c = home.copy()

        c[1] = p['roll_bias'] + p['roll_amp'] * s
        c[6] = -p['roll_bias'] - p['roll_amp'] * s

        c[2] = home[2] + p['hip_bias'] + p['hip_amp'] * s
        c[7] = home[7] + p['hip_bias'] - p['hip_amp'] * s

        sp = max(0.0, s)
        sn = max(0.0, -s)
        c[3] = home[3] + p['knee_bias'] + p['knee_amp'] * sp
        c[8] = home[8] + p['knee_bias'] + p['knee_amp'] * sn

        c[4] = home[4] + p['ank_bias'] + p['ank_amp'] * s
        c[9] = home[9] + p['ank_bias'] - p['ank_amp'] * s

        c[10] = p['torso_bias'] + p['torso_amp'] * s

        trigger = (pelvis_x > p['reach_x']) or (step > p['reach_step'])
        if trigger:
            a = clamp((pelvis_x - p['reach_x']) / 0.25, 0.0, 1.0)
            c[15] = (1 - a) * home[15] + a * p['rsp']
            c[16] = (1 - a) * home[16] + a * p['rsr']
            c[17] = (1 - a) * home[17] + a * p['rsy']
            c[18] = (1 - a) * home[18] + a * p['re']
            c[11] = (1 - a) * home[11] + a * p['lsp']
            c[12] = (1 - a) * home[12] + a * p['lsr']
            c[13] = (1 - a) * home[13] + a * p['lsy']
            c[14] = (1 - a) * home[14] + a * p['le']

        return c

    return policy


def sample_params(rng):
    return {
        'freq': rng.uniform(0.6, 2.8),
        'phase0': rng.uniform(-math.pi, math.pi),
        'roll_bias': rng.uniform(-0.18, 0.18),
        'roll_amp': rng.uniform(0.0, 0.25),
        'hip_bias': rng.uniform(-0.3, 0.2),
        'hip_amp': rng.uniform(0.02, 0.65),
        'knee_bias': rng.uniform(-0.2, 0.6),
        'knee_amp': rng.uniform(0.0, 1.2),
        'ank_bias': rng.uniform(-0.45, 0.45),
        'ank_amp': rng.uniform(0.0, 0.5),
        'torso_bias': rng.uniform(-0.6, 0.8),
        'torso_amp': rng.uniform(0.0, 0.5),
        'reach_x': rng.uniform(0.35, 0.95),
        'reach_step': int(rng.integers(120, 1100)),
        'rsp': rng.uniform(-1.8, 1.8),
        'rsr': rng.uniform(-1.8, 1.8),
        'rsy': rng.uniform(-1.8, 1.8),
        're': rng.uniform(-1.8, 1.8),
        'lsp': rng.uniform(-1.2, 1.2),
        'lsr': rng.uniform(-1.2, 1.2),
        'lsy': rng.uniform(-1.2, 1.2),
        'le': rng.uniform(-1.2, 1.2),
        'replay_steps': int(rng.integers(320, 1800)),
    }


def mutate_params(rng, p, scale=0.2):
    q = dict(p)
    for k in ['freq','phase0','roll_bias','roll_amp','hip_bias','hip_amp','knee_bias','knee_amp','ank_bias','ank_amp','torso_bias','torso_amp',
              'reach_x','rsp','rsr','rsy','re','lsp','lsr','lsy','le']:
        span = {
            'freq': 2.2, 'phase0': math.pi, 'roll_bias': 0.18, 'roll_amp': 0.25, 'hip_bias': 0.4, 'hip_amp': 0.6,
            'knee_bias': 0.7, 'knee_amp': 1.0, 'ank_bias': 0.4, 'ank_amp': 0.45, 'torso_bias': 0.8, 'torso_amp': 0.45,
            'reach_x': 0.6, 'rsp': 1.8, 'rsr': 1.8, 'rsy': 1.8, 're': 1.8, 'lsp': 1.2, 'lsr': 1.2, 'lsy': 1.2, 'le': 1.2,
        }[k]
        q[k] = p[k] + rng.normal(0.0, span * scale)
    q['reach_step'] = int(p['reach_step'] + rng.normal(0.0, 180 * scale))
    q['replay_steps'] = int(p['replay_steps'] + rng.normal(0.0, 450 * scale))

    q['freq'] = float(clamp(q['freq'], 0.4, 3.2))
    q['phase0'] = float(clamp(q['phase0'], -math.pi, math.pi))
    q['roll_bias'] = float(clamp(q['roll_bias'], -0.22, 0.22))
    q['roll_amp'] = float(clamp(q['roll_amp'], 0.0, 0.35))
    q['hip_bias'] = float(clamp(q['hip_bias'], -0.55, 0.35))
    q['hip_amp'] = float(clamp(q['hip_amp'], 0.0, 0.9))
    q['knee_bias'] = float(clamp(q['knee_bias'], -0.5, 1.0))
    q['knee_amp'] = float(clamp(q['knee_amp'], 0.0, 1.8))
    q['ank_bias'] = float(clamp(q['ank_bias'], -0.7, 0.7))
    q['ank_amp'] = float(clamp(q['ank_amp'], 0.0, 0.7))
    q['torso_bias'] = float(clamp(q['torso_bias'], -1.0, 1.4))
    q['torso_amp'] = float(clamp(q['torso_amp'], 0.0, 0.7))
    q['reach_x'] = float(clamp(q['reach_x'], 0.2, 1.05))
    q['reach_step'] = int(clamp(q['reach_step'], 80, 1700))
    q['replay_steps'] = int(clamp(q['replay_steps'], 320, 2200))
    for k in ['rsp','rsr','rsy','re']:
        q[k] = float(clamp(q[k], -2.2, 2.2))
    for k in ['lsp','lsr','lsy','le']:
        q[k] = float(clamp(q[k], -1.6, 1.6))
    return q


def evaluate(sim, p, save_path=None):
    sim.reset()
    home = sim.home_ctrl()
    policy = make_policy(p, home)

    stable_count = 0
    max_x = -1e9
    best_touch = 1e9

    last_c = home.copy()
    for i in range(p['replay_steps']):
        px = sim.pelvis_position()[0]
        c = policy(i, px)
        sim.data.ctrl[:] = c
        sim.step(1)
        last_c = c

        pelvis = sim.pelvis_position()
        up = sim.torso_up()
        d_touch = sim.distance_to_touch()
        if pelvis[0] > max_x:
            max_x = pelvis[0]
        if d_touch < best_touch:
            best_touch = d_touch
        if pelvis[2] >= 0.23 and up >= 0.55:
            stable_count += 1

    stable_frac = stable_count / max(1, p['replay_steps'])

    settle_min_z = 1e9
    settle_min_up = 1e9
    settle_best_touch = 1e9
    for _ in range(500):
        sim.data.ctrl[:] = last_c
        sim.step(1)
        pelvis = sim.pelvis_position()
        up = sim.torso_up()
        d_touch = sim.distance_to_touch()
        if pelvis[2] < settle_min_z:
            settle_min_z = pelvis[2]
        if up < settle_min_up:
            settle_min_up = up
        if d_touch < settle_best_touch:
            settle_best_touch = d_touch

    final_target_d = sim.distance_to_target()

    checkpoints = [0.25, 0.55, 0.80]
    ck_hits = sum(1 for c in checkpoints if max_x >= c)
    ck_prog = ck_hits / 3.0
    touch_prog = 1.0 if best_touch <= 0.12 else max(0.0, 1.0 - (best_touch - 0.12) / 1.2)
    stab_prog = min(1.0, stable_frac / 0.35)
    dist_prog = 1.0 if final_target_d <= 0.45 else max(0.0, 1.0 - (final_target_d - 0.45) / 1.5)
    settle_post = 0.5 * min(1.0, settle_min_z / 0.23) + 0.5 * min(1.0, settle_min_up / 0.55)
    eff_prog = min(1.0, 2200 / p['replay_steps'])
    progress = 0.15*ck_prog + 0.30*touch_prog + 0.20*stab_prog + 0.15*dist_prog + 0.10*settle_post + 0.10*eff_prog

    passed = (
        p['replay_steps'] >= 30 and
        max_x >= 0.80 and
        stable_frac >= 0.35 and
        best_touch <= 0.12 and
        final_target_d <= 0.45 and
        settle_min_z >= 0.23 and
        settle_min_up >= 0.55 and
        settle_best_touch <= 0.12
    )

    metrics = {
        'progress': progress,
        'pass': passed,
        'max_x': max_x,
        'stable_frac': stable_frac,
        'best_touch': best_touch,
        'final_target_d': final_target_d,
        'settle_min_z': settle_min_z,
        'settle_min_up': settle_min_up,
        'settle_best_touch': settle_best_touch,
        'replay_steps': p['replay_steps'],
        'pelvis': sim.pelvis_position().copy(),
    }

    if save_path is not None:
        sim.save_final_state(save_path)
    return metrics


def main():
    start = time.time()
    budget_s = 430
    deadline = start + budget_s

    rng = np.random.default_rng(0)
    sim = Sim()

    best_p = sample_params(rng)
    best_m = evaluate(sim, best_p, save_path='/work/final_state.npz')
    print('initial', best_m)

    it = 0
    while time.time() < deadline:
        it += 1
        # exploit around best most of the time
        if rng.random() < 0.78:
            cand = mutate_params(rng, best_p, scale=0.45 if it < 120 else 0.25)
        else:
            cand = sample_params(rng)

        m = evaluate(sim, cand)
        better = (m['pass'] and not best_m['pass']) or (m['pass'] == best_m['pass'] and m['progress'] > best_m['progress'])
        if better:
            best_p, best_m = cand, m
            evaluate(sim, best_p, save_path='/work/final_state.npz')
            print(f"it={it} NEW progress={best_m['progress']:.4f} pass={best_m['pass']} max_x={best_m['max_x']:.3f} touch={best_m['best_touch']:.3f} stable={best_m['stable_frac']:.3f} target={best_m['final_target_d']:.3f} settle_z={best_m['settle_min_z']:.3f} settle_up={best_m['settle_min_up']:.3f} steps={best_m['replay_steps']}")

        if it % 25 == 0:
            rem = deadline - time.time()
            print(f"it={it} best progress={best_m['progress']:.4f} pass={best_m['pass']} max_x={best_m['max_x']:.3f} touch={best_m['best_touch']:.3f} stable={best_m['stable_frac']:.3f} rem_s={rem:.1f}")

        if best_m['pass']:
            # keep improving slightly for robustness, but stop early if very good
            if best_m['progress'] > 0.98:
                break

    print('FINAL BEST', best_m)
    print('PARAMS', best_p)
    # re-save best trajectory one last time
    evaluate(sim, best_p, save_path='/work/final_state.npz')

if __name__ == '__main__':
    main()
