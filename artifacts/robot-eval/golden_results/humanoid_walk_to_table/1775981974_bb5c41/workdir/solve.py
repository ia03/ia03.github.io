from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim import MIN_PELVIS_Z, MIN_TORSO_UP, Sim, TARGET_XY


CHECKPOINTS = (0.25, 0.55, 0.85)


@dataclass
class Params:
    freq: float
    hip_amp: float
    knee_amp: float
    ankle_amp: float
    roll_amp: float
    yaw_amp: float
    torso_lean: float
    torso_bob: float
    arm_amp: float
    knee_bias: float
    ankle_bias: float
    settle_time: float
    duration: float
    ramp_time: float


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def stable_fraction(trace_pelvis, trace_up):
    good = (trace_pelvis[:, 2] >= MIN_PELVIS_Z) & (trace_up >= MIN_TORSO_UP)
    return float(np.mean(good)) if len(good) else 0.0


def evaluate_trace(trace_pelvis, trace_up, final_dist, settle_pelvis, settle_up):
    checkpoint_hits = sum(float(np.max(trace_pelvis[:, 0]) >= cp) for cp in CHECKPOINTS)
    checkpoint_progress = checkpoint_hits / len(CHECKPOINTS)
    stability_progress = clamp(stable_fraction(trace_pelvis, trace_up) / 0.95, 0.0, 1.0)
    distance_progress = clamp(1.0 - final_dist / 1.05, 0.0, 1.0)
    settle_posture_progress = (
        0.5 * clamp(np.min(settle_pelvis[:, 2]) / 0.75, 0.0, 1.0)
        + 0.5 * clamp(np.min(settle_up) / 0.8, 0.0, 1.0)
    )
    s1 = checkpoint_progress
    s2 = checkpoint_progress * stability_progress
    s3 = checkpoint_progress * stability_progress * distance_progress
    s4 = checkpoint_progress * stability_progress * distance_progress * settle_posture_progress
    return (s1 + s2 + s3 + s4) / 4.0


def controller(home, t, p: Params):
    ramp = clamp(t / max(p.ramp_time, 1e-3), 0.0, 1.0)
    phi = 2.0 * math.pi * p.freq * t
    s = math.sin(phi)
    c = math.cos(phi)
    ctrl = home.copy()

    ctrl[0] += ramp * p.yaw_amp * c
    ctrl[5] -= ramp * p.yaw_amp * c

    ctrl[1] += ramp * p.roll_amp * c
    ctrl[6] -= ramp * p.roll_amp * c

    ctrl[2] += ramp * (p.torso_lean + p.hip_amp * s)
    ctrl[7] += ramp * (p.torso_lean - p.hip_amp * s)

    knee_wave_l = max(0.0, s)
    knee_wave_r = max(0.0, -s)
    ctrl[3] += ramp * (p.knee_bias + p.knee_amp * knee_wave_l)
    ctrl[8] += ramp * (p.knee_bias + p.knee_amp * knee_wave_r)

    ctrl[4] += ramp * (p.ankle_bias - p.ankle_amp * knee_wave_l)
    ctrl[9] += ramp * (p.ankle_bias - p.ankle_amp * knee_wave_r)

    ctrl[10] += ramp * (p.torso_bob * s)

    ctrl[11] += ramp * (-p.arm_amp * s)
    ctrl[15] += ramp * (p.arm_amp * s)
    ctrl[13] += ramp * (-0.35 * p.arm_amp * c)
    ctrl[16] += ramp * (0.35 * p.arm_amp * c)
    return ctrl


def rollout(p: Params, save_path=None):
    sim = Sim()
    home = sim.home_ctrl()
    steps_per_ctrl = max(1, round(0.02 / sim.model.opt.timestep))
    ctrl_steps = int(p.duration / 0.02)
    for i in range(ctrl_steps):
        t = i * 0.02
        sim.data.ctrl[:] = controller(home, t, p)
        sim.step(steps_per_ctrl)

    trace_pelvis = np.array([e["pelvis_pos"] for e in sim._trace], dtype=float)
    trace_up = np.array([e["torso_up"] for e in sim._trace], dtype=float)
    settle_steps = int(p.settle_time / sim.model.opt.timestep)
    settle_pelvis = []
    settle_up = []
    for _ in range(settle_steps):
        sim.data.ctrl[:] = controller(home, p.duration, p)
        sim.step(1)
        settle_pelvis.append(sim.pelvis_position().copy())
        settle_up.append(sim.torso_up())
    settle_pelvis = np.array(settle_pelvis, dtype=float)
    settle_up = np.array(settle_up, dtype=float)
    score = evaluate_trace(trace_pelvis, trace_up, sim.distance_to_target(), settle_pelvis, settle_up)
    metrics = {
        "score": score,
        "max_x": float(np.max(trace_pelvis[:, 0])) if len(trace_pelvis) else 0.0,
        "stable_frac": stable_fraction(trace_pelvis, trace_up),
        "final_dist": float(sim.distance_to_target()),
        "final_pelvis": sim.pelvis_position().copy(),
        "final_up": float(sim.torso_up()),
        "min_settle_z": float(np.min(settle_pelvis[:, 2])) if len(settle_pelvis) else 0.0,
        "min_settle_up": float(np.min(settle_up)) if len(settle_up) else 0.0,
        "ctrl_steps": len(sim._ctrl_trace),
    }
    if save_path:
        sim.save_final_state(save_path)
    return metrics, sim


def sample_params(rng, base=None, scale=1.0):
    if base is None:
        return Params(
            freq=rng.uniform(0.7, 1.8),
            hip_amp=rng.uniform(0.15, 0.55),
            knee_amp=rng.uniform(0.1, 0.8),
            ankle_amp=rng.uniform(0.05, 0.45),
            roll_amp=rng.uniform(0.02, 0.14),
            yaw_amp=rng.uniform(0.0, 0.08),
            torso_lean=rng.uniform(-0.18, 0.12),
            torso_bob=rng.uniform(-0.08, 0.08),
            arm_amp=rng.uniform(0.05, 0.7),
            knee_bias=rng.uniform(-0.1, 0.45),
            ankle_bias=rng.uniform(-0.25, 0.2),
            settle_time=rng.uniform(0.25, 0.6),
            duration=rng.uniform(2.2, 5.0),
            ramp_time=rng.uniform(0.15, 0.6),
        )
    d = base.__dict__.copy()
    widths = {
        "freq": 0.15,
        "hip_amp": 0.08,
        "knee_amp": 0.12,
        "ankle_amp": 0.08,
        "roll_amp": 0.02,
        "yaw_amp": 0.015,
        "torso_lean": 0.05,
        "torso_bob": 0.03,
        "arm_amp": 0.12,
        "knee_bias": 0.08,
        "ankle_bias": 0.05,
        "settle_time": 0.08,
        "duration": 0.6,
        "ramp_time": 0.08,
    }
    bounds = {
        "freq": (0.6, 2.2),
        "hip_amp": (0.0, 0.8),
        "knee_amp": (0.0, 1.0),
        "ankle_amp": (0.0, 0.6),
        "roll_amp": (0.0, 0.2),
        "yaw_amp": (0.0, 0.15),
        "torso_lean": (-0.3, 0.2),
        "torso_bob": (-0.15, 0.15),
        "arm_amp": (0.0, 1.0),
        "knee_bias": (-0.2, 0.6),
        "ankle_bias": (-0.35, 0.3),
        "settle_time": (0.2, 0.8),
        "duration": (2.0, 6.0),
        "ramp_time": (0.05, 0.8),
    }
    for k in d:
        lo, hi = bounds[k]
        d[k] = clamp(d[k] + rng.normal(0.0, widths[k] * scale), lo, hi)
    return Params(**d)


def main():
    rng = np.random.default_rng(0)
    best_params = None
    best_metrics = None
    start = 0
    evals = 0
    max_evals = 250
    seed_bank = [
        {
            "freq": 1.6,
            "hip_amp": 0.30,
            "knee_amp": 0.55,
            "ankle_amp": 0.28,
            "roll_amp": 0.12,
            "yaw_amp": 0.04,
            "torso_lean": 0.05,
            "torso_bob": 0.00,
            "arm_amp": 0.80,
            "knee_bias": 0.00,
            "ankle_bias": -0.10,
            "settle_time": 0.35,
            "duration": 3.0,
            "ramp_time": 0.20,
        },
        {
            "freq": 2.0,
            "hip_amp": 0.42,
            "knee_amp": 0.70,
            "ankle_amp": 0.32,
            "roll_amp": 0.15,
            "yaw_amp": 0.03,
            "torso_lean": 0.02,
            "torso_bob": 0.00,
            "arm_amp": 1.00,
            "knee_bias": 0.00,
            "ankle_bias": -0.05,
            "settle_time": 0.30,
            "duration": 2.6,
            "ramp_time": 0.16,
        },
    ]

    while evals < max_evals:
        if evals < len(seed_bank):
            candidate = Params(**seed_bank[evals])
        elif best_params is None:
            candidate = sample_params(rng)
        else:
            scale = max(0.25, 1.0 - 0.015 * evals)
            candidate = sample_params(rng, best_params, scale) if rng.random() < 0.7 else sample_params(rng)
        metrics, sim = rollout(candidate)
        evals += 1
        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_metrics = metrics
            best_params = candidate
            sim.save_final_state("/work/final_state.npz")
            print(
                f"best eval={evals} score={metrics['score']:.4f} max_x={metrics['max_x']:.3f} "
                f"dist={metrics['final_dist']:.3f} up={metrics['final_up']:.3f} "
                f"stable={metrics['stable_frac']:.3f} pelvis={np.round(metrics['final_pelvis'], 3)}",
                flush=True,
            )
            print(best_params, flush=True)
        if evals % 25 == 0:
            print(f"progress eval={evals} best={best_metrics['score']:.4f}", flush=True)

    if best_metrics is None or best_params is None:
        raise RuntimeError("no rollouts completed")
    rollout(best_params, save_path="/work/final_state.npz")
    print("FINAL", best_metrics, flush=True)
    print(best_params, flush=True)


if __name__ == "__main__":
    main()
