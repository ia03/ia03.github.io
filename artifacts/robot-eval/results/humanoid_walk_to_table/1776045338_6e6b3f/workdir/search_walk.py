import math
import os
from dataclasses import dataclass, asdict

import numpy as np

from sim import MIN_PELVIS_Z, MIN_TORSO_UP, Sim, TARGET_XY


CHECKPOINTS = [0.18, 0.34, 0.45]


@dataclass
class Params:
    warmup_steps: int = 80
    walk_steps: int = 900
    stop_steps: int = 220
    settle_steps: int = 80
    freq: float = 0.085
    hip_amp: float = 0.24
    knee_amp: float = 0.18
    ankle_amp: float = 0.08
    roll_amp: float = 0.035
    torso_bias: float = -0.08
    torso_swing: float = 0.025
    arm_swing: float = 0.45
    knee_base: float = 0.92
    hip_base: float = -0.34
    ankle_base: float = -0.46
    crouch_knee: float = 1.08
    crouch_hip: float = -0.68
    crouch_ankle: float = -0.38
    target_stop_x: float = 0.56
    ramp: int = 90
    phase: float = 0.0


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def compute_progress(trace_pelvis, trace_torso_up):
    xs = trace_pelvis[:, 0]
    checkpoint_progress = sum(xs.max() >= c for c in CHECKPOINTS) / len(CHECKPOINTS)
    stable = ((trace_pelvis[:, 2] >= MIN_PELVIS_Z) & (trace_torso_up >= MIN_TORSO_UP)).mean()
    return checkpoint_progress, stable


def replay_metrics(ctrl_trace):
    sim = Sim()
    for ctrl in ctrl_trace:
        sim.data.ctrl[:] = ctrl
        sim.step()
    replay_trace_pelvis = np.array([entry["pelvis_pos"] for entry in sim._trace], dtype=float)
    replay_trace_torso = np.array([entry["torso_up"] for entry in sim._trace], dtype=float)
    for _ in range(500):
        sim.step()
    settle_trace_pelvis = np.array([entry["pelvis_pos"] for entry in sim._trace], dtype=float)
    settle_trace_torso = np.array([entry["torso_up"] for entry in sim._trace], dtype=float)
    checkpoint_progress, stable_frac = compute_progress(replay_trace_pelvis, replay_trace_torso)
    min_settle_z = settle_trace_pelvis[-51:, 2].min()
    min_settle_torso = settle_trace_torso[-51:].min()
    final_xy_dist = np.linalg.norm(settle_trace_pelvis[-1, :2] - TARGET_XY)
    passes = (
        len(ctrl_trace) >= 30
        and replay_trace_pelvis[0, :2].dot(replay_trace_pelvis[0, :2]) ** 0.5 <= 0.15
        and checkpoint_progress >= 1.0
        and stable_frac >= 0.35
        and final_xy_dist <= 0.20
        and min_settle_z >= MIN_PELVIS_Z
        and min_settle_torso >= MIN_TORSO_UP
    )
    distance_progress = 1.0 if final_xy_dist <= 0.20 else clamp(1 - (final_xy_dist - 0.20) / 0.20, 0, 1)
    settle_progress = 0.5 * clamp(min_settle_z / MIN_PELVIS_Z, 0, 1) + 0.5 * clamp(min_settle_torso / MIN_TORSO_UP, 0, 1)
    efficiency = clamp(1600 / max(len(ctrl_trace), 1), 0, 1)
    progress_score = (
        0.25 * checkpoint_progress
        + 0.25 * clamp(stable_frac / 0.35, 0, 1)
        + 0.30 * distance_progress
        + 0.15 * settle_progress
        + 0.05 * efficiency
    )
    return {
        "pass": bool(passes),
        "progress_score": float(progress_score),
        "checkpoint_progress": float(checkpoint_progress),
        "stable_frac": float(stable_frac),
        "final_xy_dist": float(final_xy_dist),
        "min_settle_z": float(min_settle_z),
        "min_settle_torso": float(min_settle_torso),
        "final_pelvis": settle_trace_pelvis[-1].tolist(),
    }


def control_step(home, i, pelvis_x, p: Params):
    ctrl = home.copy()
    if i < p.warmup_steps:
        return ctrl

    walk_i = i - p.warmup_steps
    if pelvis_x < p.target_stop_x and walk_i < p.walk_steps:
        ramp = min(1.0, walk_i / max(p.ramp, 1))
        theta = p.phase + walk_i * p.freq * 2 * math.pi
        s = math.sin(theta)
        c = math.cos(theta)
        left = s
        right = -s
        ctrl[2] = p.hip_base + ramp * p.hip_amp * left
        ctrl[7] = p.hip_base + ramp * p.hip_amp * right
        ctrl[3] = p.knee_base - ramp * p.knee_amp * left
        ctrl[8] = p.knee_base - ramp * p.knee_amp * right
        ctrl[4] = p.ankle_base + ramp * p.ankle_amp * left
        ctrl[9] = p.ankle_base + ramp * p.ankle_amp * right
        ctrl[1] = ramp * p.roll_amp * c
        ctrl[6] = -ramp * p.roll_amp * c
        ctrl[10] = p.torso_bias + ramp * p.torso_swing * c
        ctrl[11] = -ramp * p.arm_swing * right
        ctrl[15] = -ramp * p.arm_swing * left
        return ctrl

    stop_alpha = clamp((walk_i - p.walk_steps) / max(p.stop_steps, 1), 0.0, 1.0)
    ctrl[2] = (1 - stop_alpha) * home[2] + stop_alpha * p.crouch_hip
    ctrl[7] = (1 - stop_alpha) * home[7] + stop_alpha * p.crouch_hip
    ctrl[3] = (1 - stop_alpha) * home[3] + stop_alpha * p.crouch_knee
    ctrl[8] = (1 - stop_alpha) * home[8] + stop_alpha * p.crouch_knee
    ctrl[4] = (1 - stop_alpha) * home[4] + stop_alpha * p.crouch_ankle
    ctrl[9] = (1 - stop_alpha) * home[9] + stop_alpha * p.crouch_ankle
    ctrl[10] = (1 - stop_alpha) * p.torso_bias + stop_alpha * (-0.18)
    return ctrl


def rollout(p: Params, save_path=None, render_dir=None):
    sim = Sim()
    home = sim.home_ctrl()
    total_steps = p.warmup_steps + p.walk_steps + p.stop_steps + p.settle_steps
    frame_paths = []
    for i in range(total_steps):
        pelvis_x = sim.pelvis_position()[0]
        sim.data.ctrl[:] = control_step(home, i, pelvis_x, p)
        sim.step()
        if render_dir and i % 80 == 0:
            frame = sim.render(width=480, height=360)
            fp = os.path.join(render_dir, f"frame_{i:04d}.npy")
            np.save(fp, frame)
            frame_paths.append(fp)
    if save_path:
        sim.save_final_state(save_path)
    metrics = replay_metrics(np.array(sim._ctrl_trace, dtype=float))
    metrics["runtime_final_pelvis"] = sim.pelvis_position().tolist()
    metrics["runtime_torso_up"] = float(sim.torso_up())
    metrics["runtime_dist"] = float(sim.distance_to_target())
    return sim, metrics, frame_paths


def main():
    base = Params()
    sim, metrics, _ = rollout(base, save_path="/work/final_state.npz")
    print("BASE", asdict(base), metrics)

    candidates = []
    for freq in [0.07, 0.08, 0.09, 0.10]:
        for hip_amp in [0.18, 0.24, 0.30]:
            for knee_amp in [0.12, 0.18, 0.24]:
                for torso_bias in [-0.04, -0.08, -0.12]:
                    for target_stop_x in [0.50, 0.56, 0.60]:
                        p = Params(
                            freq=freq,
                            hip_amp=hip_amp,
                            knee_amp=knee_amp,
                            torso_bias=torso_bias,
                            target_stop_x=target_stop_x,
                            ankle_amp=0.08 + 0.05 * (hip_amp - 0.18) / 0.12,
                            crouch_hip=-0.60 - 0.10 * (target_stop_x - 0.50) / 0.10,
                            crouch_knee=1.00 + 0.08 * (knee_amp - 0.12) / 0.12,
                        )
                        candidates.append(p)

    best_params = base
    best_metrics = metrics
    for idx, p in enumerate(candidates, 1):
        _, m, _ = rollout(p)
        if m["progress_score"] > best_metrics["progress_score"]:
            best_params = p
            best_metrics = m
            rollout(p, save_path="/work/final_state.npz")
            print("NEW_BEST", idx, asdict(p), m, flush=True)
        if idx % 20 == 0:
            print("PROGRESS", idx, best_metrics["progress_score"], best_metrics, flush=True)

    print("BEST_PARAMS", asdict(best_params))
    print("BEST_METRICS", best_metrics)


if __name__ == "__main__":
    main()
