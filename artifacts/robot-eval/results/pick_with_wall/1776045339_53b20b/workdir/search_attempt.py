import math
import time
import numpy as np
import mujoco
from sim import Sim

HAND = 'hand'
WALL_DEADLINE = 1776046239


def hand_body_id(sim):
    return mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND)


def target_rot(yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def orientation_error(R, Rd):
    return 0.5 * (
        np.cross(R[:, 0], Rd[:, 0]) +
        np.cross(R[:, 1], Rd[:, 1]) +
        np.cross(R[:, 2], Rd[:, 2])
    )


def step_pose(sim, pos, rot, grip, pos_gain=5.0, rot_gain=2.0, max_joint_step=0.08):
    bid = hand_body_id(sim)
    curp = sim.data.xpos[bid].copy()
    curr = sim.data.xmat[bid].reshape(3, 3).copy()
    ep = np.array(pos) - curp
    er = orientation_error(curr, rot)
    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, bid)
    J = np.vstack([jacp[:, :7], jacr[:, :7]])
    e = np.concatenate([pos_gain * ep, rot_gain * er])
    lam = 3e-3
    dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
    dq = np.clip(dq, -max_joint_step, max_joint_step)
    q_target = sim.data.qpos[:7] + dq
    for j in range(7):
        lo, hi = sim.model.jnt_range[j]
        q_target[j] = np.clip(q_target[j], lo, hi)
    sim.data.ctrl[:7] = q_target
    sim.data.ctrl[7] = grip
    sim.step()


def run_stage(sim, pos, rot, grip, steps):
    for _ in range(steps):
        step_pose(sim, pos, rot, grip)


def progress(stats):
    grasp = 1.0 if stats['ever_contact'] else 0.0
    return_prog = min(max((0.55 - stats['best_x']) / (0.55 - 0.48), 0.0), 1.0)
    lift_prog = min(max((stats['best_z'] - 0.435) / (0.56 - 0.435), 0.0), 1.0)
    final_return = min(max((0.55 - stats['final_x']) / (0.55 - 0.42), 0.0), 1.0)
    final_height = min(max((stats['final_z'] - 0.435) / (0.56 - 0.435), 0.0), 1.0)
    settle_contact = min(max(stats['settle_contact_fraction'] / 0.5, 0.0), 1.0)
    eff = min(max(2600.0 / max(stats['replay_steps'], 1), 0.0), 1.0)
    return 0.05*grasp + 0.15*return_prog + 0.10*lift_prog + 0.10*final_return + 0.25*final_height + 0.25*settle_contact + 0.10*eff


def evaluate(ctrl_trace):
    sim = Sim()
    best_x = 1e9
    best_z = -1e9
    ever_contact = False
    for u in ctrl_trace:
        sim.data.ctrl[:] = u
        sim.step()
        cup = sim.cup_position()
        best_x = min(best_x, float(cup[0]))
        best_z = max(best_z, float(cup[2]))
        ever_contact = ever_contact or sim.has_gripper_cup_contact()
    settle_contact = 0
    for _ in range(500):
        sim.step()
        settle_contact += int(sim.has_gripper_cup_contact())
    cup = sim.cup_position()
    return {
        'replay_steps': len(ctrl_trace),
        'best_x': best_x,
        'best_z': best_z,
        'ever_contact': ever_contact,
        'final_x': float(cup[0]),
        'final_z': float(cup[2]),
        'settle_contact_fraction': settle_contact / 500.0,
    }


def execute_plan(yaw, grasp_x, grasp_y, grasp_z, close_profile=(180,80,20,0)):
    sim = Sim()
    R = target_rot(yaw)
    plan = [
        ((0.28, 0.18, 0.80), 255, 200),
        ((0.42, 0.20, 0.76), 255, 160),
        ((grasp_x - 0.02, grasp_y + 0.03, 0.68), 255, 160),
        ((grasp_x, grasp_y, grasp_z + 0.06), 255, 180),
        ((grasp_x, grasp_y, grasp_z + 0.02), 255, 140),
        ((grasp_x, grasp_y, grasp_z), 220, 120),
        ((grasp_x, grasp_y, grasp_z), close_profile[0], 80),
        ((grasp_x, grasp_y, grasp_z), close_profile[1], 80),
        ((grasp_x, grasp_y, grasp_z), close_profile[2], 100),
        ((grasp_x, grasp_y, grasp_z), close_profile[3], 140),
        ((grasp_x, grasp_y, 0.60), close_profile[3], 220),
        ((0.49, grasp_y, 0.64), close_profile[3], 160),
        ((0.42, grasp_y, 0.66), close_profile[3], 220),
        ((0.39, grasp_y, 0.66), close_profile[3], 260),
    ]
    for idx, (pos, grip, steps) in enumerate(plan):
        run_stage(sim, pos, R, grip, steps)
        if idx in {0, 5, 9, 12, 13}:
            sim.save_final_state('/work/final_state.npz')
    stats = evaluate(np.array(sim._ctrl_trace))
    return sim, stats


def main():
    t0 = time.time()
    best = None
    candidates = []
    for yaw_deg in [0, 45, 90, -45]:
        yaw = math.radians(yaw_deg)
        for gx in [0.548, 0.555, 0.562]:
            for gy in [0.145, 0.155, 0.165]:
                for gz in [0.500, 0.506, 0.512]:
                    candidates.append((yaw_deg, yaw, gx, gy, gz))
    for i, (yaw_deg, yaw, gx, gy, gz) in enumerate(candidates, 1):
        if time.time() > WALL_DEADLINE - 90:
            break
        sim, stats = execute_plan(yaw, gx, gy, gz)
        score = progress(stats)
        print(f'{i}/{len(candidates)} yaw={yaw_deg} gx={gx:.3f} gy={gy:.3f} gz={gz:.3f} score={score:.3f} stats={stats}')
        if best is None or score > best[0]:
            best = (score, (yaw_deg, gx, gy, gz), stats, sim)
            sim.save_final_state('/work/final_state.npz')
            print('NEW_BEST', best[0], best[1], best[2])
        if stats['final_x'] <= 0.42 and stats['final_z'] >= 0.56 and stats['settle_contact_fraction'] >= 0.5 and stats['ever_contact'] and stats['best_x'] <= 0.48 and stats['best_z'] >= 0.56:
            print('FOUND_SUCCESS', yaw_deg, gx, gy, gz)
            sim.save_final_state('/work/final_state.npz')
            return
    print('BEST', None if best is None else (best[0], best[1], best[2]), 'elapsed', time.time()-t0)

if __name__ == '__main__':
    main()
