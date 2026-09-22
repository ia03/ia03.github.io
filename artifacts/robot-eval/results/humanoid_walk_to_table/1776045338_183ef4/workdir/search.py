import math
import os
import time
from dataclasses import dataclass

import numpy as np

from sim import MIN_PELVIS_Z, MIN_TORSO_UP, Sim, TARGET_XY


FINAL_PATH = "/work/final_state.npz"
START_TS = time.time()
DEADLINE_TS = START_TS + 12.5 * 60


@dataclass
class Metrics:
    score: float
    checkpoint_progress: float
    stability_progress: float
    distance_progress: float
    settle_posture_progress: float
    efficiency_progress: float
    replay_steps: int
    final_distance: float
    min_settle_pelvis_z: float
    min_settle_torso_up: float
    stable_trace_fraction: float
    max_x: float
    final_x: float


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def evaluate_trace(trace_pelvis, trace_torso_up, ctrl_steps):
    pelvis_x = trace_pelvis[:, 0]
    checkpoints = [0.18, 0.34, 0.45]
    checkpoint_hits = sum(float(np.any(pelvis_x >= c)) for c in checkpoints)
    checkpoint_progress = checkpoint_hits / len(checkpoints)

    stable = (trace_pelvis[:, 2] >= MIN_PELVIS_Z) & (trace_torso_up >= MIN_TORSO_UP)
    stable_trace_fraction = float(np.mean(stable))
    stability_progress = clamp(stable_trace_fraction / 0.35, 0.0, 1.0)

    final_xy = trace_pelvis[-1, :2]
    final_distance = float(np.linalg.norm(final_xy - TARGET_XY))
    distance_progress = 1.0 if final_distance <= 0.20 else clamp(1 - (final_distance - 0.20) / 0.20, 0.0, 1.0)

    settle_pelvis = trace_pelvis[-500:, 2]
    settle_torso = trace_torso_up[-500:]
    min_settle_pelvis_z = float(np.min(settle_pelvis))
    min_settle_torso_up = float(np.min(settle_torso))
    settle_posture_progress = 0.5 * clamp(min_settle_pelvis_z / 0.23, 0.0, 1.0) + 0.5 * clamp(min_settle_torso_up / 0.55, 0.0, 1.0)

    efficiency_progress = clamp(1600 / max(ctrl_steps, 1), 0.0, 1.0)
    score = (
        0.25 * checkpoint_progress
        + 0.25 * stability_progress
        + 0.30 * distance_progress
        + 0.15 * settle_posture_progress
        + 0.05 * efficiency_progress
    )
    return Metrics(
        score=score,
        checkpoint_progress=checkpoint_progress,
        stability_progress=stability_progress,
        distance_progress=distance_progress,
        settle_posture_progress=settle_posture_progress,
        efficiency_progress=efficiency_progress,
        replay_steps=ctrl_steps,
        final_distance=final_distance,
        min_settle_pelvis_z=min_settle_pelvis_z,
        min_settle_torso_up=min_settle_torso_up,
        stable_trace_fraction=stable_trace_fraction,
        max_x=float(np.max(pelvis_x)),
        final_x=float(pelvis_x[-1]),
    )


def controller(sim, p, step_idx):
    t = sim.data.time
    x = sim.pelvis_position()[0]
    dist = sim.distance_to_target()
    home = sim.home_ctrl()
    ctrl = home.copy()

    progress = clamp(x / max(p["target_x"], 1e-3), 0.0, 1.2)
    phase = 2.0 * math.pi * p["freq"] * t + p["phase"]
    s = math.sin(phase)
    c = math.cos(phase)
    left = s
    right = -s

    swing_gate_l = max(0.0, left)
    swing_gate_r = max(0.0, right)

    ctrl[10] = home[10] + p["torso_lean"] + p["torso_bob"] * c

    ctrl[2] = home[2] + p["hip_bias"] + p["hip_amp"] * left
    ctrl[7] = home[7] + p["hip_bias"] + p["hip_amp"] * right

    ctrl[3] = home[3] + p["knee_bias"] + p["knee_amp"] * swing_gate_l
    ctrl[8] = home[8] + p["knee_bias"] + p["knee_amp"] * swing_gate_r

    ctrl[4] = home[4] + p["ankle_bias"] - p["ankle_amp"] * left
    ctrl[9] = home[9] + p["ankle_bias"] - p["ankle_amp"] * right

    # Shift weight toward stance leg.
    ctrl[1] = home[1] + p["roll_bias"] - p["roll_amp"] * left
    ctrl[6] = home[6] - p["roll_bias"] - p["roll_amp"] * right

    ctrl[0] = home[0] + p["yaw_amp"] * left
    ctrl[5] = home[5] - p["yaw_amp"] * left

    ctrl[11] = home[11] - p["arm_amp"] * left
    ctrl[15] = home[15] + p["arm_amp"] * left
    ctrl[12] = home[12] + p["arm_roll"]
    ctrl[16] = home[16] - p["arm_roll"]

    # Transition into a lower stop near the target.
    if x >= p["stop_x"] or dist <= p["stop_dist"]:
        alpha = clamp((max(x - p["stop_x"], 0.0) + max(p["stop_dist"] - dist, 0.0)) / max(p["stop_span"], 1e-3), 0.0, 1.0)
        ctrl[2] += alpha * p["stop_hip"]
        ctrl[7] += alpha * p["stop_hip"]
        ctrl[3] += alpha * p["stop_knee"]
        ctrl[8] += alpha * p["stop_knee"]
        ctrl[4] += alpha * p["stop_ankle"]
        ctrl[9] += alpha * p["stop_ankle"]
        ctrl[10] += alpha * p["stop_torso"]
        decay = 1.0 - alpha
        ctrl[1] = home[1] + decay * (ctrl[1] - home[1])
        ctrl[6] = home[6] + decay * (ctrl[6] - home[6])

    # If it starts tipping, crouch and reduce oscillation instead of fighting late.
    if sim.torso_up() < 0.7:
        ctrl[3] += p["recover_knee"]
        ctrl[8] += p["recover_knee"]
        ctrl[10] += p["recover_torso"]

    return ctrl


def run_trial(params, steps=1200, settle=500, save=False):
    sim = Sim()
    trace_pelvis = []
    trace_torso = []
    for i in range(steps):
        sim.data.ctrl[:] = controller(sim, params, i)
        sim.step()
        trace_pelvis.append(sim.pelvis_position().copy())
        trace_torso.append(sim.torso_up())
    # Hold final pose through settle since the grader does this after replay.
    final_ctrl = sim.data.ctrl.copy()
    for _ in range(settle):
        sim.data.ctrl[:] = final_ctrl
        sim.step()
        trace_pelvis.append(sim.pelvis_position().copy())
        trace_torso.append(sim.torso_up())
    trace_pelvis = np.asarray(trace_pelvis)
    trace_torso = np.asarray(trace_torso)
    metrics = evaluate_trace(trace_pelvis, trace_torso, steps)
    if save:
        sim.save_final_state(FINAL_PATH)
    return metrics, sim


def sample_params(rng, base=None):
    p = dict(base or {})
    def pick(name, lo, hi):
        if name not in p:
            p[name] = rng.uniform(lo, hi)
    pick("freq", 0.45, 1.25)
    pick("phase", -math.pi, math.pi)
    pick("target_x", 0.55, 0.70)
    pick("torso_lean", -0.35, 0.45)
    pick("torso_bob", -0.08, 0.08)
    pick("hip_bias", -0.45, 0.2)
    pick("hip_amp", 0.1, 0.85)
    pick("knee_bias", -0.1, 0.55)
    pick("knee_amp", -0.1, 0.95)
    pick("ankle_bias", -0.35, 0.3)
    pick("ankle_amp", -0.45, 0.45)
    pick("roll_bias", -0.08, 0.08)
    pick("roll_amp", -0.22, 0.22)
    pick("yaw_amp", -0.12, 0.12)
    pick("arm_amp", -0.5, 0.8)
    pick("arm_roll", -0.3, 0.3)
    pick("stop_x", 0.36, 0.62)
    pick("stop_dist", 0.08, 0.22)
    pick("stop_span", 0.05, 0.20)
    pick("stop_hip", -0.55, 0.10)
    pick("stop_knee", 0.15, 0.95)
    pick("stop_ankle", -0.35, 0.25)
    pick("stop_torso", -0.45, 0.25)
    pick("recover_knee", 0.0, 0.65)
    pick("recover_torso", -0.2, 0.2)
    return p


def mutate_params(rng, base):
    out = dict(base)
    for k, v in list(out.items()):
        scale = 0.15 * (abs(v) + 0.15)
        out[k] = v + rng.normal(0.0, scale)
    return out


def print_metrics(tag, m):
    print(
        f"{tag} score={m.score:.3f} x={m.final_x:.3f} max_x={m.max_x:.3f} "
        f"dist={m.final_distance:.3f} stable={m.stable_trace_fraction:.3f} "
        f"settle_z={m.min_settle_pelvis_z:.3f} settle_up={m.min_settle_torso_up:.3f}"
    )


def ensure_saved():
    if not os.path.exists(FINAL_PATH):
        sim = Sim()
        for _ in range(60):
            sim.data.ctrl[:] = sim.home_ctrl()
            sim.step()
        sim.save_final_state(FINAL_PATH)
        print("saved baseline to", FINAL_PATH)


def main():
    rng = np.random.default_rng(0)
    ensure_saved()

    seeds = [
        dict(freq=0.72, phase=0.0, torso_lean=0.12, torso_bob=0.02, hip_bias=-0.12, hip_amp=0.38, knee_bias=0.05, knee_amp=0.38, ankle_bias=-0.06, ankle_amp=0.18, roll_bias=0.0, roll_amp=0.08, yaw_amp=0.04, arm_amp=0.35, arm_roll=0.05, stop_x=0.50, stop_dist=0.14, stop_span=0.10, stop_hip=-0.18, stop_knee=0.35, stop_ankle=-0.04, stop_torso=0.05, recover_knee=0.2, recover_torso=0.0, target_x=0.58),
        dict(freq=0.88, phase=0.4, torso_lean=0.20, torso_bob=0.0, hip_bias=-0.18, hip_amp=0.56, knee_bias=0.08, knee_amp=0.42, ankle_bias=-0.10, ankle_amp=0.18, roll_bias=0.0, roll_amp=0.12, yaw_amp=0.02, arm_amp=0.45, arm_roll=0.08, stop_x=0.48, stop_dist=0.12, stop_span=0.08, stop_hip=-0.24, stop_knee=0.52, stop_ankle=-0.08, stop_torso=0.08, recover_knee=0.28, recover_torso=-0.05, target_x=0.58),
        dict(freq=0.64, phase=-0.3, torso_lean=0.05, torso_bob=-0.01, hip_bias=-0.05, hip_amp=0.30, knee_bias=0.12, knee_amp=0.22, ankle_bias=-0.02, ankle_amp=0.10, roll_bias=0.0, roll_amp=0.05, yaw_amp=0.0, arm_amp=0.25, arm_roll=0.04, stop_x=0.46, stop_dist=0.16, stop_span=0.12, stop_hip=-0.15, stop_knee=0.30, stop_ankle=-0.06, stop_torso=0.10, recover_knee=0.18, recover_torso=0.0, target_x=0.58),
    ]

    best_m = None
    best_p = None

    for idx, p in enumerate(seeds):
        m, sim = run_trial(p, save=(idx == 0))
        print_metrics(f"seed{idx}", m)
        if best_m is None or m.score > best_m.score:
            best_m, best_p = m, p
            sim.save_final_state(FINAL_PATH)
            print("saved new best seed")

    trial = 0
    while time.time() < DEADLINE_TS - 45:
        if trial % 20 == 0:
            print("elapsed", round(time.time() - START_TS, 1), "seconds")
        if trial < 80:
            p = sample_params(rng)
        else:
            p = mutate_params(rng, best_p if rng.random() < 0.8 else sample_params(rng))
        m, sim = run_trial(p)
        if (trial < 20) or (m.score >= best_m.score * 0.95) or (m.max_x > best_m.max_x + 0.05):
            print_metrics(f"trial{trial}", m)
        if (
            best_m is None
            or m.score > best_m.score
            or (m.max_x > best_m.max_x and m.stable_trace_fraction >= best_m.stable_trace_fraction * 0.8)
        ):
            best_m, best_p = m, p
            sim.save_final_state(FINAL_PATH)
            print("saved improved best")
            print(best_p)
        trial += 1

    print("best params", best_p)
    print_metrics("best", best_m)


if __name__ == "__main__":
    main()
