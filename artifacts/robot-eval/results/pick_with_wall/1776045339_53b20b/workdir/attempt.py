import numpy as np
import mujoco
from sim import Sim

HAND = 'hand'


def hand_body_id(sim):
    return mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND)


def step_toward(sim, target_pos, gripper_ctrl, pos_gain=4.0, steps=1, max_joint_step=0.05):
    bid = hand_body_id(sim)
    for _ in range(steps):
        cur = sim.data.xpos[bid].copy()
        err = np.array(target_pos) - cur
        jacp = np.zeros((3, sim.model.nv))
        jacr = np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, bid)
        J = jacp[:, :7]
        lam = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), pos_gain * err)
        dq = np.clip(dq, -max_joint_step, max_joint_step)
        q_target = sim.data.qpos[:7] + dq
        for j in range(7):
            lo, hi = sim.model.jnt_range[j]
            q_target[j] = np.clip(q_target[j], lo, hi)
        sim.data.ctrl[:7] = q_target
        sim.data.ctrl[7] = gripper_ctrl
        sim.step()


def run_segment(sim, target, grip, n, label=''):
    bid = hand_body_id(sim)
    for i in range(n):
        step_toward(sim, target, grip)
        if i in {0, n//2, n-1}:
            print(label, i, 'hand', sim.data.xpos[bid].copy(), 'cup', sim.cup_position(), 'contact', sim.has_gripper_cup_contact())
        if sim.has_gripper_cup_contact():
            sim.save_final_state('/work/final_state.npz')


def evaluate_replay(ctrl_trace):
    sim = Sim()
    best_x = 1e9
    best_z = -1e9
    ever_contact = False
    for u in ctrl_trace:
        sim.data.ctrl[:] = u
        sim.step()
        cup = sim.cup_position()
        best_x = min(best_x, cup[0])
        best_z = max(best_z, cup[2])
        ever_contact = ever_contact or sim.has_gripper_cup_contact()
    settle_contact = 0
    for _ in range(500):
        sim.step()
        settle_contact += int(sim.has_gripper_cup_contact())
    cup = sim.cup_position()
    return {
        'replay_steps': len(ctrl_trace),
        'best_x': float(best_x),
        'best_z': float(best_z),
        'ever_contact': ever_contact,
        'final_x': float(cup[0]),
        'final_z': float(cup[2]),
        'settle_contact_fraction': settle_contact / 500.0,
    }


def main():
    sim = Sim()
    path = [
        ((0.30, 0.20, 0.78), 255, 220, 'pre1'),
        ((0.46, 0.22, 0.74), 255, 180, 'pre2'),
        ((0.55, 0.185, 0.72), 255, 180, 'pre3'),
        ((0.555, 0.165, 0.60), 255, 240, 'down1'),
        ((0.555, 0.155, 0.535), 255, 220, 'down2'),
        ((0.555, 0.152, 0.520), 180, 80, 'close1'),
        ((0.555, 0.150, 0.512), 120, 90, 'close2'),
        ((0.555, 0.150, 0.508), 60, 100, 'close3'),
        ((0.555, 0.150, 0.508), 0, 140, 'close4'),
        ((0.555, 0.150, 0.60), 0, 220, 'lift'),
        ((0.50, 0.15, 0.63), 0, 180, 'ret1'),
        ((0.42, 0.15, 0.64), 0, 260, 'ret2'),
        ((0.38, 0.15, 0.64), 0, 320, 'hold'),
    ]
    for idx, (target, grip, n, label) in enumerate(path):
        run_segment(sim, target, grip, n, label)
        sim.save_final_state('/work/final_state.npz')
        print('checkpoint', idx, 'time', sim.data.time, 'cup', sim.cup_position(), 'contact', sim.has_gripper_cup_contact())
    stats = evaluate_replay(np.array(sim._ctrl_trace))
    print('final cup', sim.cup_position(), 'contact', sim.has_gripper_cup_contact(), 'time', sim.data.time)
    print('eval', stats)

if __name__ == '__main__':
    main()
