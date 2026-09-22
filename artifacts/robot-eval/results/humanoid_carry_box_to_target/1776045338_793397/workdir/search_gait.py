import itertools
import math
import time

import numpy as np

from sim import Sim, TARGET_XY, MIN_PELVIS_Z, MIN_TORSO_UP, MIN_BOX_Z, MAX_BOX_TO_PELVIS


def evaluate(sim, settle_steps=500):
    trace = sim._trace
    pelvis = np.array([t["pelvis_pos"] for t in trace])
    box = np.array([t["box_pos"] for t in trace])
    torso_up = np.array([t["torso_up"] for t in trace])
    box_d = np.array([t["box_distance_to_pelvis"] for t in trace])
    stable = (
        (pelvis[:, 2] >= MIN_PELVIS_Z)
        & (torso_up >= MIN_TORSO_UP)
        & (box[:, 2] >= MIN_BOX_Z)
        & (box_d <= MAX_BOX_TO_PELVIS)
    )
    qpos = sim.data.qpos.copy()
    qvel = sim.data.qvel.copy()
    ctrl = sim.data.ctrl.copy()
    settle_pelvis = []
    settle_torso = []
    settle_box = []
    settle_box_d = []
    for _ in range(settle_steps):
        sim.step(1)
        settle_pelvis.append(sim.pelvis_position().copy())
        settle_torso.append(sim.torso_up())
        settle_box.append(sim.box_position().copy())
        settle_box_d.append(sim.box_distance_to_pelvis())
    settle_pelvis = np.array(settle_pelvis)
    settle_torso = np.array(settle_torso)
    settle_box = np.array(settle_box)
    settle_box_d = np.array(settle_box_d)
    result = {
        "replay_steps": len(sim._ctrl_trace),
        "max_pelvis_x": float(pelvis[:, 0].max()),
        "final_pelvis_x": float(pelvis[-1, 0]),
        "max_box_x": float(box[:, 0].max()),
        "final_box_x": float(box[-1, 0]),
        "stable_frac": float(stable.mean()),
        "final_pelvis_dist": float(np.linalg.norm(settle_pelvis[-1, :2] - TARGET_XY)),
        "final_box_dist": float(np.linalg.norm(settle_box[-1, :2] - TARGET_XY)),
        "min_settle_pelvis_z": float(settle_pelvis[:, 2].min()),
        "min_settle_torso_up": float(settle_torso.min()),
        "min_settle_box_z": float(settle_box[:, 2].min()),
        "max_settle_box_d": float(settle_box_d.max()),
    }
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel
    sim.data.ctrl[:] = ctrl
    return result


def gait_pose(home, phase, amp_hip, amp_knee, amp_ankle, lean, stance_knee, arm_pitch, arm_roll, elbow):
    s = math.sin(phase)
    c = math.cos(phase)
    q = home.copy()
    q[2] = home[2] - lean - amp_hip * s
    q[3] = stance_knee + amp_knee * max(0.0, s)
    q[4] = home[4] - amp_ankle * s
    q[7] = home[7] - lean + amp_hip * s
    q[8] = stance_knee + amp_knee * max(0.0, -s)
    q[9] = home[9] + amp_ankle * s
    q[1] = 0.04 * c
    q[6] = -0.04 * c
    q[10] = 0.08 + 0.04 * s
    q[11:15] = [arm_pitch, arm_roll, -0.1, elbow]
    q[15:19] = [arm_pitch, -arm_roll, 0.1, elbow]
    return q


def run_candidate(params, total_steps=900, warmup=150):
    sim = Sim()
    home = sim.home_ctrl()
    dt = sim.model.opt.timestep
    for _ in range(40):
        sim.data.ctrl[:] = home
        sim.step(1)
    amp_hip, amp_knee, amp_ankle, lean, stance_knee, freq, arm_pitch, arm_roll, elbow = params
    for t in range(total_steps):
        if t < warmup:
            blend = (t + 1) / warmup
            ctrl = gait_pose(home, 0.0, amp_hip * blend, amp_knee * blend, amp_ankle * blend,
                             lean * blend, home[3] * (1 - blend) + stance_knee * blend,
                             arm_pitch * blend, arm_roll * blend, elbow * blend)
        else:
            phase = 2 * math.pi * freq * dt * (t - warmup)
            ctrl = gait_pose(home, phase, amp_hip, amp_knee, amp_ankle, lean, stance_knee,
                             arm_pitch, arm_roll, elbow)
        sim.data.ctrl[:] = ctrl
        sim.step(1)
    return sim, evaluate(sim)


def score(m):
    return (
        4.0 * min(m["max_pelvis_x"], 0.25)
        + 2.0 * min(m["stable_frac"], 1.0)
        - 3.0 * max(0.0, m["final_pelvis_dist"] - 0.2)
        - 4.0 * max(0.0, m["max_settle_box_d"] - 0.42)
        - 4.0 * max(0.0, 0.74 - m["min_settle_pelvis_z"])
        - 4.0 * max(0.0, 0.50 - m["min_settle_torso_up"])
        - 4.0 * max(0.0, 0.72 - m["min_settle_box_z"])
    )


def main():
    start = time.time()
    candidates = list(itertools.product(
        [0.18, 0.24, 0.30],
        [0.18, 0.28, 0.38],
        [0.06, 0.12],
        [0.03, 0.08, 0.13],
        [0.82, 0.90, 1.00],
        [0.8, 1.1, 1.4],
        [0.0, 0.4],
        [0.0, 0.25],
        [0.0, 0.8],
    ))
    best = None
    for i, params in enumerate(candidates, 1):
        sim, m = run_candidate(params)
        s = score(m)
        print(i, params, m, "score", s, flush=True)
        if best is None or s > best[0]:
            best = (s, params, m)
            sim.save_final_state("/work/final_state.npz")
            print("BEST", best, flush=True)
        if time.time() - start > 420:
            break
    print("FINAL_BEST", best)


if __name__ == "__main__":
    main()
