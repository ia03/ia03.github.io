import time
import numpy as np
import mujoco
from sim import Sim


def peg_alignment_x(sim):
    bid = sim.peg_body_id
    xmat = sim.data.xmat[bid].reshape(3, 3)
    return float(abs(xmat[0, 0]))


def metrics(sim):
    p = sim.peg_position()
    return {
        'x': float(p[0]),
        'y': float(p[1]),
        'z': float(p[2]),
        'align': peg_alignment_x(sim),
    }


def print_metrics(tag, sim):
    m = metrics(sim)
    print(f"{tag}: peg=({m['x']:.4f}, {m['y']:.4f}, {m['z']:.4f}) align={m['align']:.3f} t={sim.data.time:.3f} steps={len(sim._ctrl_trace)}")


def solved(m):
    return (
        m['x'] >= 0.515
        and abs(m['y'] - (-0.10)) <= 0.025
        and abs(m['z'] - 0.52) <= 0.020
        and abs(m['align']) >= 0.45
    )


def move_ee(sim, target, steps=800, kp=6.0, damping=1e-3, max_step=0.03):
    model, data = sim.model, sim.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    jnt_qposadr = model.jnt_qposadr
    # 7 arm joints start at qpos 0..6 in this model
    q_idx = np.arange(7)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(steps):
        hand_pos = data.xpos[hand_id].copy()
        err = target - hand_pos
        v = kp * err
        n = np.linalg.norm(v)
        if n > max_step:
            v *= max_step / n

        mujoco.mj_jacBodyCom(model, data, jacp, jacr, hand_id)
        J = jacp[:, :7]
        # damped least squares
        A = J @ J.T + damping * np.eye(3)
        qdot = J.T @ np.linalg.solve(A, v)

        q = data.qpos[q_idx].copy()
        q_des = q + qdot

        # stay in actuator range
        lo = model.actuator_ctrlrange[:7, 0]
        hi = model.actuator_ctrlrange[:7, 1]
        q_des = np.clip(q_des, lo, hi)
        data.ctrl[:7] = q_des
        data.ctrl[7] = 0.0
        sim.step(1)


def hold(sim, steps=500):
    # hold current joint posture
    sim.data.ctrl[:7] = sim.data.qpos[:7]
    sim.data.ctrl[7] = 0.0
    sim.step(steps)


def run_once(save_path='/work/final_state.npz'):
    start = time.time()
    sim = Sim()
    print_metrics('start', sim)

    # Phase 1: descend near peg tab lane
    move_ee(sim, np.array([0.40, -0.102, 0.55]), steps=1200)
    # Phase 2: contact tab from behind
    move_ee(sim, np.array([0.422, -0.102, 0.525]), steps=1200)
    # Phase 3: push into slot along +x
    move_ee(sim, np.array([0.57, -0.102, 0.522]), steps=2200)
    # Phase 4: slight overpush / settle
    move_ee(sim, np.array([0.585, -0.102, 0.522]), steps=800)
    hold(sim, steps=900)

    print_metrics('final', sim)
    m = metrics(sim)
    print('solved?', solved(m))
    sim.save_final_state(save_path)
    print(f'saved {save_path} in {time.time()-start:.2f}s, replay_steps={len(sim._ctrl_trace)}')


if __name__ == '__main__':
    run_once()
