import math
import os
import numpy as np

from sim import (
    MAX_BOX_TO_PELVIS,
    MIN_BOX_Z,
    MIN_PELVIS_Z,
    MIN_TORSO_UP,
    Sim,
    TARGET_RADIUS,
)


def metrics(sim, settle_steps=500):
    trace = sim._trace
    pelvis_x = np.array([t["pelvis_pos"][0] for t in trace], dtype=float)
    box_x = np.array([t["box_pos"][0] for t in trace], dtype=float)
    stable = np.array(
        [
            (
                t["pelvis_pos"][2] >= MIN_PELVIS_Z
                and t["torso_up"] >= MIN_TORSO_UP
                and t["box_pos"][2] >= MIN_BOX_Z
                and t["box_distance_to_pelvis"] <= MAX_BOX_TO_PELVIS
            )
            for t in trace
        ],
        dtype=bool,
    )
    replay_ok = (
        len(sim._ctrl_trace) >= 30
        and np.all(pelvis_x.max() >= np.array([0.05, 0.09, 0.13]))
        and np.all(box_x.max() >= np.array([0.00, 0.04, 0.08]))
        and stable.mean() >= 0.90
    )
    qpos = sim.data.qpos.copy()
    qvel = sim.data.qvel.copy()
    ctrl = sim.data.ctrl.copy()
    settle_min_pelvis_z = 1e9
    settle_min_torso_up = 1e9
    settle_min_box_z = 1e9
    settle_max_box_pelvis = -1e9
    for _ in range(settle_steps):
        sim.step(1)
        settle_min_pelvis_z = min(settle_min_pelvis_z, sim.pelvis_position()[2])
        settle_min_torso_up = min(settle_min_torso_up, sim.torso_up())
        settle_min_box_z = min(settle_min_box_z, sim.box_position()[2])
        settle_max_box_pelvis = max(settle_max_box_pelvis, sim.box_distance_to_pelvis())
    pelvis_target = sim.distance_to_target()
    box_target = sim.box_distance_to_target()
    settle_ok = (
        pelvis_target <= TARGET_RADIUS
        and box_target <= TARGET_RADIUS
        and settle_min_pelvis_z >= MIN_PELVIS_Z
        and settle_min_torso_up >= MIN_TORSO_UP
        and settle_min_box_z >= MIN_BOX_Z
        and settle_max_box_pelvis <= MAX_BOX_TO_PELVIS
    )
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel
    sim.data.ctrl[:] = ctrl
    return {
        "replay_ok": replay_ok,
        "settle_ok": settle_ok,
        "pass": replay_ok and settle_ok,
        "pelvis_x_max": float(pelvis_x.max()),
        "box_x_max": float(box_x.max()),
        "stable_frac": float(stable.mean()),
        "pelvis_target": float(pelvis_target),
        "box_target": float(box_target),
        "settle_min_pelvis_z": float(settle_min_pelvis_z),
        "settle_min_torso_up": float(settle_min_torso_up),
        "settle_min_box_z": float(settle_min_box_z),
        "settle_max_box_pelvis": float(settle_max_box_pelvis),
        "steps": len(sim._ctrl_trace),
    }


def save_baseline():
    sim = Sim()
    sim.data.ctrl[:] = sim.home_ctrl()
    sim.step(40)
    sim.save_final_state("/work/final_state.npz")
    print("saved baseline", metrics(sim))


def gait_rollout(
    sim,
    steps=800,
    step_len=0.22,
    lift=0.06,
    knee_bend=0.10,
    torso_pitch=0.18,
    arm_pitch=-1.05,
    arm_roll_left=-0.20,
    arm_roll_right=0.20,
    elbow_left=1.45,
    elbow_right=-1.45,
    settle_steps=80,
):
    home = sim.home_ctrl()
    ctrl = home.copy()
    ctrl[10] = torso_pitch
    ctrl[11] = arm_pitch
    ctrl[12] = arm_roll_left
    ctrl[13] = 0.0
    ctrl[14] = elbow_left
    ctrl[15] = arm_pitch
    ctrl[16] = arm_roll_right
    ctrl[17] = 0.0
    ctrl[18] = elbow_right
    sim.data.ctrl[:] = ctrl
    sim.step(60)
    for i in range(steps):
        phase = 2.0 * math.pi * i / 70.0
        s = math.sin(phase)
        left = home[:5].copy()
        right = home[5:10].copy()
        left[2] += step_len * s
        right[2] -= step_len * s
        left[3] += knee_bend + lift * max(0.0, -s)
        right[3] += knee_bend + lift * max(0.0, s)
        left[4] -= 0.5 * lift * max(0.0, -s)
        right[4] -= 0.5 * lift * max(0.0, s)
        left[1] += 0.03 * s
        right[1] -= 0.03 * s
        ctrl[:5] = left
        ctrl[5:10] = right
        ctrl[10] = torso_pitch + 0.04 * math.sin(phase)
        sim.data.ctrl[:] = ctrl
        sim.step(1)
    sim.step(settle_steps)


def main():
    save_baseline()
    sim = Sim()
    gait_rollout(sim)
    result = metrics(sim)
    print("gait", result)
    if result["pelvis_x_max"] > 0.02:
        sim.save_final_state("/work/final_state.npz")
        print("saved gait")


if __name__ == "__main__":
    main()
