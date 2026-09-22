"""Deterministic golden/reference solution for humanoid_carry_box_to_target.

This script is intended to be copied into the task workdir and executed by
``harness/run_golden.py``. It runs a small, deterministic set of fixed
reference controllers, scores them with the same objective progress components
used by the current replay-first grader, and saves the best final state to
``/work/final_state.npz``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from sim import Sim, TARGET_XY
except ImportError:  # pragma: no cover - local convenience fallback
    import sys

    task_workdir = Path(__file__).resolve().parents[2] / "tasks" / "humanoid_carry_box_to_target" / "workdir"
    sys.path.insert(0, str(task_workdir))
    from sim import Sim, TARGET_XY


BOX_INIT_POS = np.array([0.18, 0.0, 0.90], dtype=float)
TARGET_RADIUS = 0.22
MIN_PELVIS_Z = 0.74
MIN_TORSO_UP = 0.78
MIN_BOX_Z = 0.72
MAX_BOX_TO_PELVIS = 0.42
FORWARD_SIM_STEPS = 500
START_RADIUS = 0.15
PROGRESS_CHECKPOINTS = (0.35, 0.65, 0.95)
MIN_TRACE_SAMPLES = 30
MIN_STABLE_TRACE_FRACTION = 0.90
IDEAL_REPLAY_STEPS = 2600

KP = np.array(
    [220, 160, 260, 420, 120, 220, 160, 260, 420, 120, 160, 120, 80, 40, 30, 120, 80, 40, 30],
    dtype=float,
)
KD = np.array(
    [10, 8, 12, 20, 6, 10, 8, 12, 20, 6, 8, 4, 4, 2, 2, 4, 4, 2, 2],
    dtype=float,
)


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def weighted_subscore(*components) -> float:
    total_weight = sum(weight for _, _, weight in components)
    if total_weight <= 0:
        return 0.0
    return float(sum(clamp01(value) * weight for _, value, weight in components) / total_weight)


@dataclass(frozen=True)
class CarryProfile:
    name: str
    hip_pitch: float
    knee: float
    ankle: float
    torso: float
    shoulder_pitch: float
    elbow: float
    arm_pitch: float
    arm_roll: float
    arm_yaw: float
    arm_elbow: float
    phase_start: float
    phase_duration: float
    x_gain: float
    x_vel_gain: float
    z_gain: float
    y_gain: float
    y_vel_gain: float
    horizon: float
    settle_steps: int


class CarryController:
    def __init__(self, sim: Sim, profile: CarryProfile):
        self.sim = sim
        self.profile = profile
        self.base = sim.data.qpos[7:26].copy()
        self.left_ankle = sim.model.body("left_ankle_link").id
        self.right_ankle = sim.model.body("right_ankle_link").id
        self.stance = self._make_pose(
            hip_pitch=profile.hip_pitch,
            knee=profile.knee,
            ankle=profile.ankle,
            torso=profile.torso,
            shoulder_pitch=profile.shoulder_pitch,
            elbow=profile.elbow,
        )

    def _make_pose(self, hip_pitch, knee, ankle, torso, shoulder_pitch, elbow):
        q = self.base.copy()
        q[2] += hip_pitch
        q[7] += hip_pitch
        q[3] += knee
        q[8] += knee
        q[4] += ankle
        q[9] += ankle
        q[10] += torso
        q[11] += shoulder_pitch
        q[15] += shoulder_pitch
        q[14] += elbow
        q[18] += elbow
        return q

    def target(self, t: float):
        q = self.stance.copy()
        phase = clamp01((t - self.profile.phase_start) / max(self.profile.phase_duration, 1e-6))
        q[11] += self.profile.arm_pitch * phase
        q[15] += self.profile.arm_pitch * phase
        q[12] += self.profile.arm_roll * phase
        q[16] -= self.profile.arm_roll * phase
        q[13] += self.profile.arm_yaw * phase
        q[17] -= self.profile.arm_yaw * phase
        q[14] += self.profile.arm_elbow * phase
        q[18] += self.profile.arm_elbow * phase

        foot_x = 0.5 * (self.sim.data.xpos[self.left_ankle, 0] + self.sim.data.xpos[self.right_ankle, 0])
        pelvis = self.sim.pelvis_position()
        xerr = pelvis[0] - foot_x + 0.02
        xvel = self.sim.data.qvel[0]
        yerr = pelvis[1]
        yvel = self.sim.data.qvel[1]

        delta = self.profile.x_gain * xerr - self.profile.x_vel_gain * xvel - self.profile.z_gain * (pelvis[2] - 0.98)
        q[2] += -0.76 * delta
        q[7] += -0.76 * delta
        q[3] += 0.60 * delta
        q[8] += 0.60 * delta
        q[4] += -0.67 * delta
        q[9] += -0.67 * delta
        q[10] += -0.47 * delta

        lat = self.profile.y_gain * yerr + self.profile.y_vel_gain * yvel
        q[1] += lat
        q[6] += lat
        q[12] += 0.12 * lat
        q[16] += 0.12 * lat
        return q

    def torque(self, q_target):
        q = self.sim.data.qpos[7:26]
        qd = self.sim.data.qvel[6:25]
        bias = self.sim.data.qfrc_bias[6:25]
        ctrl = bias + KP * (q_target - q) - KD * qd
        lo = self.sim.model.actuator_ctrlrange[:, 0]
        hi = self.sim.model.actuator_ctrlrange[:, 1]
        return np.clip(ctrl, lo, hi)


class TraceRecorder:
    def __init__(self):
        self.pelvis = []
        self.torso_up = []
        self.box = []
        self.box_to_pelvis = []
        self.ctrl = []

    def record(self, sim: Sim):
        self.pelvis.append(sim.pelvis_position().copy())
        self.torso_up.append(float(sim.torso_up()))
        self.box.append(sim.box_position().copy())
        self.box_to_pelvis.append(float(sim.box_distance_to_pelvis()))
        self.ctrl.append(sim.data.ctrl.copy())


def score_from_trace(recorder: TraceRecorder, sim: Sim):
    trace_pelvis = np.asarray(recorder.pelvis, dtype=float).reshape(-1, 3)
    trace_torso_up = np.asarray(recorder.torso_up, dtype=float)
    trace_box = np.asarray(recorder.box, dtype=float).reshape(-1, 3)
    trace_box_distance_to_pelvis = np.asarray(recorder.box_to_pelvis, dtype=float)
    replay_steps = len(recorder.ctrl)

    if trace_pelvis.size == 0:
        return 0.0, {
            "error": "empty trace",
        }

    start_distance = float(np.linalg.norm(trace_pelvis[0, :2]))
    checkpoint_hits = [bool(np.any(trace_pelvis[:, 0] >= checkpoint)) for checkpoint in PROGRESS_CHECKPOINTS]
    box_checkpoint_hits = [bool(np.any(trace_box[:, 0] >= checkpoint - 0.05)) for checkpoint in PROGRESS_CHECKPOINTS]
    stable_mask = (
        (trace_pelvis[:, 2] >= MIN_PELVIS_Z)
        & (trace_torso_up >= MIN_TORSO_UP)
        & (trace_box[:, 2] >= MIN_BOX_Z)
        & (trace_box_distance_to_pelvis <= MAX_BOX_TO_PELVIS)
    )
    stable_fraction = float(np.mean(stable_mask)) if stable_mask.size else 0.0

    pelvis_pos = sim.pelvis_position().copy()
    box_pos = sim.box_position().copy()
    pelvis_dist = float(np.linalg.norm(pelvis_pos[:2] - TARGET_XY))
    box_dist = float(np.linalg.norm(box_pos[:2] - TARGET_XY))
    min_pelvis_z = float(np.min(trace_pelvis[:, 2]))
    min_torso_up = float(np.min(trace_torso_up))
    min_box_z = float(np.min(trace_box[:, 2]))
    max_box_to_pelvis = float(np.max(trace_box_distance_to_pelvis))

    success = (
        trace_pelvis.shape[0] >= MIN_TRACE_SAMPLES
        and start_distance <= START_RADIUS
        and all(checkpoint_hits)
        and all(box_checkpoint_hits)
        and stable_fraction >= MIN_STABLE_TRACE_FRACTION
        and pelvis_dist <= TARGET_RADIUS
        and box_dist <= TARGET_RADIUS
        and min_pelvis_z >= MIN_PELVIS_Z
        and min_torso_up >= MIN_TORSO_UP
        and min_box_z >= MIN_BOX_Z
        and max_box_to_pelvis <= MAX_BOX_TO_PELVIS
    )

    pelvis_checkpoint_progress = sum(checkpoint_hits) / len(PROGRESS_CHECKPOINTS)
    box_checkpoint_progress = sum(box_checkpoint_hits) / len(PROGRESS_CHECKPOINTS)
    stability_progress = clamp01(stable_fraction / MIN_STABLE_TRACE_FRACTION)
    pelvis_distance_progress = clamp01(1.0 - pelvis_dist / float(np.linalg.norm(TARGET_XY)))
    box_distance_progress = clamp01(
        1.0 - box_dist / float(np.linalg.norm(TARGET_XY - np.array([BOX_INIT_POS[0], BOX_INIT_POS[1]])))
    )
    carry_proximity_progress = clamp01(1.0 - max(0.0, max_box_to_pelvis - MAX_BOX_TO_PELVIS) / MAX_BOX_TO_PELVIS)
    efficiency_progress = clamp01(IDEAL_REPLAY_STEPS / max(replay_steps, 1))
    box_height_progress = clamp01(min_box_z / MIN_BOX_Z)

    score_components = [
        {"name": "box_distance_progress", "value": box_distance_progress, "weight": 0.15},
        {"name": "carry_proximity_progress", "value": carry_proximity_progress, "weight": 0.30},
        {"name": "box_height_progress", "value": box_height_progress, "weight": 0.25},
        {"name": "stability_progress", "value": stability_progress, "weight": 0.10},
        {"name": "box_checkpoint_progress", "value": box_checkpoint_progress, "weight": 0.03},
        {"name": "pelvis_distance_progress", "value": pelvis_distance_progress, "weight": 0.02},
        {"name": "pelvis_checkpoint_progress", "value": pelvis_checkpoint_progress, "weight": 0.00},
        {"name": "efficiency_progress", "value": efficiency_progress, "weight": 0.15},
    ]
    progress_score = weighted_subscore(*[(item["name"], item["value"], item["weight"]) for item in score_components])
    return progress_score, {
        "success": success,
        "pelvis_xyz_after": [float(x) for x in pelvis_pos],
        "box_xyz_after": [float(x) for x in box_pos],
        "start_distance_from_origin": start_distance,
        "checkpoint_hits": checkpoint_hits,
        "box_checkpoint_hits": box_checkpoint_hits,
        "trace_samples": int(trace_pelvis.shape[0]),
        "replay_steps": int(replay_steps),
        "stable_trace_fraction": stable_fraction,
        "pelvis_distance_to_target": pelvis_dist,
        "box_distance_to_target": box_dist,
        "target_radius": TARGET_RADIUS,
        "min_pelvis_z_during_sim": min_pelvis_z,
        "min_torso_up_during_sim": min_torso_up,
        "min_box_z_during_sim": min_box_z,
        "max_box_to_pelvis_during_sim": max_box_to_pelvis,
        "pelvis_checkpoint_progress": pelvis_checkpoint_progress,
        "box_checkpoint_progress": box_checkpoint_progress,
        "stability_progress": stability_progress,
        "pelvis_distance_progress": pelvis_distance_progress,
        "box_distance_progress": box_distance_progress,
        "carry_proximity_progress": carry_proximity_progress,
        "box_height_progress": box_height_progress,
        "efficiency_progress": efficiency_progress,
        "progress_score": progress_score,
        "score_method": "weighted_subscores",
        "score_components": score_components,
    }


def run_stage(sim: Sim, controller: CarryController, recorder: TraceRecorder, seconds: float):
    steps = max(1, int(seconds / sim.model.opt.timestep))
    for _ in range(steps):
        q_target = controller.target(sim.data.time)
        sim.data.ctrl[:] = controller.torque(q_target)
        sim.step()
        recorder.record(sim)


def run_hold(sim: Sim, recorder: TraceRecorder, seconds: float):
    steps = max(1, int(seconds / sim.model.opt.timestep))
    for _ in range(steps):
        sim.step()
        recorder.record(sim)


def run_profile(profile: CarryProfile):
    sim = Sim()
    controller = CarryController(sim, profile)
    recorder = TraceRecorder()
    run_stage(sim, controller, recorder, profile.horizon)
    run_hold(sim, recorder, profile.settle_steps * sim.model.opt.timestep)
    progress_score, info = score_from_trace(recorder, sim)
    info["profile"] = profile.name
    return sim, progress_score, info


def run_open_loop_cradle_walk():
    sim = Sim()
    base = sim.home_ctrl()

    cradle = base.copy()
    cradle[10] = 0.15
    cradle[11] = 1.45
    cradle[12] = -0.65
    cradle[13] = 0.15
    cradle[14] = -1.45
    cradle[15] = 1.45
    cradle[16] = 0.65
    cradle[17] = -0.15
    cradle[18] = 1.45
    cradle[2] = -0.55
    cradle[3] = 1.05
    cradle[4] = -0.50
    cradle[7] = -0.55
    cradle[8] = 1.05
    cradle[9] = -0.50

    recorder = TraceRecorder()

    for alpha in np.linspace(0.0, 1.0, 160):
        ctrl = base * (1.0 - alpha) + cradle * alpha
        sim.data.ctrl[:] = ctrl
        sim.step()
        recorder.record(sim)

    for _ in range(120):
        sim.data.ctrl[:] = cradle
        sim.step()
        recorder.record(sim)

    for k in range(420):
        t = k * 0.18
        s = math.sin(t)
        c = math.cos(t)
        ctrl = cradle.copy()
        ctrl[2] = -0.48 + 0.05 - 0.08 * s
        ctrl[3] = 0.92 + 0.20 * s
        ctrl[4] = -0.44 - 0.10 * s
        ctrl[7] = -0.48 + 0.05 + 0.08 * s
        ctrl[8] = 0.92 - 0.20 * s
        ctrl[9] = -0.44 + 0.10 * s
        ctrl[0] = 0.05 * c
        ctrl[5] = -0.05 * c
        ctrl[1] = 0.03 + 0.05 * s
        ctrl[6] = -0.03 - 0.05 * s
        ctrl[10] = 0.18
        ctrl[11] = 1.55 + 0.15 * c + 0.05 * s
        ctrl[12] = -0.65
        ctrl[13] = 0.18
        ctrl[14] = -1.45
        ctrl[15] = 1.55 - 0.15 * c - 0.05 * s
        ctrl[16] = 0.65
        ctrl[17] = -0.18
        ctrl[18] = 1.45
        sim.data.ctrl[:] = ctrl
        sim.step()
        recorder.record(sim)

    for _ in range(120):
        sim.data.ctrl[:] = cradle
        sim.step()
        recorder.record(sim)

    progress_score, info = score_from_trace(recorder, sim)
    info["profile"] = "open_loop_cradle_walk"
    return sim, progress_score, info


PROFILES = [
    CarryProfile(
        name="balanced",
        hip_pitch=-0.25,
        knee=-0.42,
        ankle=-0.40,
        torso=-0.24,
        shoulder_pitch=-0.07,
        elbow=-0.05,
        arm_pitch=0.25,
        arm_roll=0.80,
        arm_yaw=0.35,
        arm_elbow=0.80,
        phase_start=0.50,
        phase_duration=2.00,
        x_gain=0.70,
        x_vel_gain=0.09,
        z_gain=0.65,
        y_gain=2.00,
        y_vel_gain=0.68,
        horizon=4.00,
        settle_steps=120,
    ),
    CarryProfile(
        name="forward_bias",
        hip_pitch=-0.23,
        knee=-0.40,
        ankle=-0.38,
        torso=-0.22,
        shoulder_pitch=-0.07,
        elbow=-0.05,
        arm_pitch=0.22,
        arm_roll=0.82,
        arm_yaw=0.38,
        arm_elbow=0.84,
        phase_start=0.45,
        phase_duration=2.20,
        x_gain=0.82,
        x_vel_gain=0.08,
        z_gain=0.58,
        y_gain=2.10,
        y_vel_gain=0.70,
        horizon=4.20,
        settle_steps=120,
    ),
    CarryProfile(
        name="stable_bias",
        hip_pitch=-0.27,
        knee=-0.45,
        ankle=-0.42,
        torso=-0.26,
        shoulder_pitch=-0.06,
        elbow=-0.05,
        arm_pitch=0.30,
        arm_roll=0.78,
        arm_yaw=0.32,
        arm_elbow=0.76,
        phase_start=0.55,
        phase_duration=1.80,
        x_gain=0.60,
        x_vel_gain=0.10,
        z_gain=0.72,
        y_gain=1.90,
        y_vel_gain=0.66,
        horizon=4.00,
        settle_steps=150,
    ),
    CarryProfile(
        name="aggressive_forward",
        hip_pitch=-0.20,
        knee=-0.38,
        ankle=-0.36,
        torso=-0.20,
        shoulder_pitch=-0.06,
        elbow=-0.04,
        arm_pitch=0.34,
        arm_roll=0.86,
        arm_yaw=0.42,
        arm_elbow=0.90,
        phase_start=0.40,
        phase_duration=2.40,
        x_gain=1.02,
        x_vel_gain=0.11,
        z_gain=0.50,
        y_gain=2.25,
        y_vel_gain=0.72,
        horizon=4.40,
        settle_steps=120,
    ),
    CarryProfile(
        name="aggressive_stride",
        hip_pitch=-0.14,
        knee=-0.33,
        ankle=-0.30,
        torso=-0.15,
        shoulder_pitch=-0.05,
        elbow=-0.04,
        arm_pitch=0.20,
        arm_roll=0.88,
        arm_yaw=0.46,
        arm_elbow=0.92,
        phase_start=0.35,
        phase_duration=2.50,
        x_gain=1.25,
        x_vel_gain=0.15,
        z_gain=0.44,
        y_gain=2.35,
        y_vel_gain=0.78,
        horizon=5.00,
        settle_steps=100,
    ),
]


def main():
    candidates = []
    for profile in PROFILES:
        sim, score, info = run_profile(profile)
        candidates.append((score, info, sim))
        print(json.dumps({"candidate": profile.name, "score": score, "info": info}, indent=2))

    sim, score, info = run_open_loop_cradle_walk()
    candidates.append((score, info, sim))
    print(json.dumps({"candidate": info["profile"], "score": score, "info": info}, indent=2))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_info, best_sim = candidates[0]
    best_sim.save_final_state("/work/final_state.npz")
    print(json.dumps({"selected": best_info["profile"], "score": best_score, "info": best_info}, indent=2))
    print("saved /work/final_state.npz")


if __name__ == "__main__":
    main()
