import numpy as np
import mujoco
from sim import Sim

OPEN = 255.0
HALF_OPEN = 170.0
CLOSE = 0.0


def body_jacp(sim, body_id):
    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, body_id)
    return jacp[:, :7]


def pinch_center_and_jac(sim, left_id, right_id):
    xl = sim.data.xpos[left_id].copy()
    xr = sim.data.xpos[right_id].copy()
    Jl = body_jacp(sim, left_id)
    Jr = body_jacp(sim, right_id)
    return 0.5 * (xl + xr), 0.5 * (Jl + Jr)


def drive_pinch_to(sim, left_id, right_id, target, steps, grip, kp=2.0, damping=1e-3):
    low = sim.model.actuator_ctrlrange[:7, 0]
    high = sim.model.actuator_ctrlrange[:7, 1]
    for _ in range(steps):
        q = sim.data.qpos[:7].copy()
        x, J = pinch_center_and_jac(sim, left_id, right_id)
        err = target - x
        A = J @ J.T + damping * np.eye(3)
        v = kp * err
        dq = J.T @ np.linalg.solve(A, v)
        q_des = q + dq
        sim.data.ctrl[:7] = np.clip(q_des, low, high)
        sim.data.ctrl[7] = grip
        sim.step(1)


def hold(sim, steps, grip):
    for _ in range(steps):
        sim.data.ctrl[:7] = sim.data.qpos[:7]
        sim.data.ctrl[7] = grip
        sim.step(1)


def close_gripper(sim, steps=800):
    for i in range(steps):
        a = i / max(1, steps - 1)
        sim.data.ctrl[:7] = sim.data.qpos[:7]
        sim.data.ctrl[7] = (1.0 - a) * HALF_OPEN + a * CLOSE
        sim.step(1)


def replay_metrics(ctrl_trace):
    sim = Sim()
    for u in ctrl_trace:
        sim.data.ctrl[:] = u
        sim.step(1)

    best_x = 1e9
    best_z = -1e9
    ever_contact = False
    for t in sim._trace:
        x, _, z = t['cup_pos']
        best_x = min(best_x, x)
        best_z = max(best_z, z)
        if t['cup_contact'] > 0.5:
            ever_contact = True

    contact = 0
    for _ in range(500):
        sim.step(1)
        if sim.has_gripper_cup_contact():
            contact += 1
    final = sim.cup_position().copy()
    return {
        'best_x': float(best_x),
        'best_z': float(best_z),
        'ever_contact': bool(ever_contact),
        'final_x': float(final[0]),
        'final_z': float(final[2]),
        'settle_contact_fraction': float(contact / 500.0),
    }


def run_once(save_path='/work/final_state.npz'):
    sim = Sim()
    left_id = sim.model.body('left_finger').id
    right_id = sim.model.body('right_finger').id

    cup0 = sim.cup_position().copy()
    above = np.array([cup0[0], cup0[1], cup0[2] + 0.18])
    near = np.array([cup0[0], cup0[1], cup0[2] + 0.04])
    lift = np.array([0.50, 0.15, 0.70])
    retrieve = np.array([0.36, 0.14, 0.68])

    hold(sim, 200, OPEN)
    drive_pinch_to(sim, left_id, right_id, above, 1200, OPEN)
    drive_pinch_to(sim, left_id, right_id, near, 900, HALF_OPEN)
    close_gripper(sim, 800)
    hold(sim, 300, CLOSE)
    drive_pinch_to(sim, left_id, right_id, lift, 1200, CLOSE)
    drive_pinch_to(sim, left_id, right_id, retrieve, 1600, CLOSE)
    hold(sim, 1500, CLOSE)

    sim.save_final_state(save_path)
    metrics = replay_metrics(np.array(sim._ctrl_trace))
    metrics['steps'] = len(sim._ctrl_trace)
    return metrics


if __name__ == '__main__':
    print(run_once())
