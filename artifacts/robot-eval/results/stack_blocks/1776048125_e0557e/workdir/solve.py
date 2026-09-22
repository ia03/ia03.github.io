import numpy as np
import mujoco
from sim import Sim


CTRL_MIN = None
CTRL_MAX = None


def run_episode(params, save_path=None, verbose=False):
    sim = Sim()
    m, d = sim.model, sim.data
    hand_id = m.body('hand').id
    q_home = d.qpos[:7].copy()
    cmin = m.actuator_ctrlrange[:, 0].copy()
    cmax = m.actuator_ctrlrange[:, 1].copy()

    def ik_step(target_xyz, grip, gain=1.4, damp=2e-4, null_gain=0.02):
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, None, hand_id)
        J = jacp[:, :7]
        err = target_xyz - d.xpos[hand_id]
        A = J @ J.T + damp * np.eye(3)
        dq = J.T @ np.linalg.solve(A, gain * err)
        q = d.qpos[:7].copy()
        dq += null_gain * (q_home - q)
        q_tgt = q + dq
        d.ctrl[:7] = np.clip(q_tgt, cmin[:7], cmax[:7])
        d.ctrl[7] = float(np.clip(grip, cmin[7], cmax[7]))

    def go(target_xyz, steps, grip, gain=1.4):
        target_xyz = np.array(target_xyz, dtype=float)
        for _ in range(steps):
            ik_step(target_xyz, grip, gain=gain)
            sim.step(1)

    def hold(steps, grip):
        q = d.qpos[:7].copy()
        d.ctrl[:7] = np.clip(q, cmin[:7], cmax[:7])
        d.ctrl[7] = float(np.clip(grip, cmin[7], cmax[7]))
        sim.step(steps)

    red0 = sim.block_positions()['red']
    green0 = sim.block_positions()['green']

    lift_z = params.get('lift_z', 0.62)
    pregrasp_z = params.get('pregrasp_z', 0.56)
    grasp_z = params.get('grasp_z', 0.488)
    place_z = params.get('place_z', 0.538)
    open_u = params.get('open_u', 220.0)
    close_u = params.get('close_u', 0.0)
    place_dx = params.get('place_dx', 0.0)
    place_dy = params.get('place_dy', 0.0)

    # Open and move above red block
    go([red0[0], red0[1], 0.64], 280, open_u, gain=1.2)
    go([red0[0], red0[1], pregrasp_z], 220, open_u, gain=1.2)
    go([red0[0], red0[1], grasp_z], 220, open_u, gain=1.0)

    # Close to grasp, then lift
    go([red0[0], red0[1], grasp_z], 240, close_u, gain=0.9)
    go([red0[0], red0[1], lift_z], 260, close_u, gain=1.2)

    # Transfer above green and place
    tx = green0[0] + place_dx
    ty = green0[1] + place_dy
    go([tx, ty, lift_z], 340, close_u, gain=1.2)
    go([tx, ty, place_z], 240, close_u, gain=1.0)

    # Release cleanly and retreat
    go([tx, ty, place_z], 180, open_u, gain=0.8)
    go([tx, ty, lift_z], 220, open_u, gain=1.0)

    # Let stack settle
    hold(700, open_u)

    p = sim.block_positions()
    dx, dy, dz = p['red'] - p['green']
    max_xy = max(abs(dx), abs(dy))
    success = (abs(dx) <= 0.018) and (abs(dy) <= 0.018) and (0.040 <= dz <= 0.070)

    # Public progress proxy
    xy_progress = 1.0 if max_xy <= 0.008 else max(0.0, (0.018 - max_xy) / (0.018 - 0.008))
    z_err = abs(dz - 0.055)
    z_progress = 1.0 if z_err <= 0.006 else max(0.0, (0.015 - z_err) / (0.015 - 0.006))
    replay_steps = len(sim._ctrl_trace)
    eff = min(1.0, 7000.0 / max(1, replay_steps))
    score = 0.45 * (1.0 if success else 0.0) + 0.25 * xy_progress + 0.18 * z_progress + 0.02 * eff

    if verbose:
        print('params', params)
        print('final red', p['red'], 'green', p['green'])
        print(f'dx={dx:.4f} dy={dy:.4f} dz={dz:.4f} steps={replay_steps} success={success} score={score:.3f}')

    if save_path is not None:
        sim.save_final_state(save_path)

    return {
        'success': success,
        'score': score,
        'dx': float(dx),
        'dy': float(dy),
        'dz': float(dz),
        'steps': replay_steps,
        'params': dict(params),
    }


def main():
    save_path = '/work/final_state.npz'

    # Mandatory early save: first executable end-to-end attempt.
    first_params = dict(grasp_z=0.488, place_z=0.538, place_dx=0.0, place_dy=0.0, open_u=220.0)
    best = run_episode(first_params, save_path=save_path, verbose=True)
    print('EARLY_SAVE', best)

    # Then refine with small sweep.
    candidates = []
    for grasp_z in [0.482, 0.486, 0.490, 0.494]:
        for place_z in [0.534, 0.538, 0.542, 0.546]:
            for dx in [-0.004, -0.002, 0.0, 0.002, 0.004]:
                for dy in [-0.004, -0.002, 0.0, 0.002, 0.004]:
                    candidates.append(dict(grasp_z=grasp_z, place_z=place_z, place_dx=dx, place_dy=dy, open_u=220.0))

    for i, params in enumerate(candidates, 1):
        out = run_episode(params, save_path=None, verbose=False)
        if (out['success'] and not best['success']) or (out['score'] > best['score'] + 1e-9):
            best = out
            run_episode(params, save_path=save_path, verbose=True)
            print('NEW_BEST', i, best)
        if i % 50 == 0:
            print('checked', i, 'best', best)
        if best['success']:
            # Keep searching a bit for robustness not needed; exit early.
            break

    print('FINAL_BEST', best)


if __name__ == '__main__':
    main()
