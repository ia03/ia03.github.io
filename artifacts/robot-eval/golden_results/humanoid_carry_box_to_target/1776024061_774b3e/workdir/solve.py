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


def run_open_loop_carry_walk(profile: dict):
    sim = Sim()
    left_ankle = sim.model.body("left_ankle_link").id
    right_ankle = sim.model.body("right_ankle_link").id
    base = sim.home_ctrl().copy()

    # A stable, box-holding crouch that keeps the box close to the torso.
    base[2] = profile["stance_hip"]
    base[3] = profile["stance_knee"]
    base[4] = profile["stance_ankle"]
    base[7] = profile["stance_hip"]
    base[8] = profile["stance_knee"]
    base[9] = profile["stance_ankle"]
    base[10] = profile["torso"]
    base[11] = profile["arm_pitch"]
    base[15] = profile["arm_pitch"]
    base[12] = profile["arm_roll"]
    base[16] = -profile["arm_roll"]
    base[13] = profile["arm_yaw"]
    base[17] = -profile["arm_yaw"]
    base[14] = profile["arm_elbow"]
    base[18] = profile["arm_elbow"]

    recorder = TraceRecorder()

    ramp_steps = 120
    for alpha in np.linspace(0.0, 1.0, ramp_steps):
        ctrl = sim.home_ctrl().copy() * (1.0 - alpha) + base * alpha
        sim.data.ctrl[:] = ctrl
        sim.step()
        recorder.record(sim)

    walk_steps = int(profile["duration"] / 0.02)
    for i in range(walk_steps):
        t = i * 0.02
        phase = 2.0 * math.pi * profile["freq"] * t + profile["phase"]
        s = math.sin(phase)
        c = math.cos(phase)
        pos_s = max(0.0, s)
        neg_s = max(0.0, -s)

        ctrl = base.copy()
        pelvis = sim.pelvis_position()
        foot_x = 0.5 * (
            float(sim.data.xpos[left_ankle, 0]) + float(sim.data.xpos[right_ankle, 0])
        )
        xerr = float(pelvis[0] - foot_x + 0.02)
        xvel = float(sim.data.qvel[0])
        speed_err = profile["target_speed"] - float(sim.data.qvel[0])
        y_err = TARGET_XY[1] - float(sim.data.qpos[1])
        y_vel = float(sim.data.qvel[1])
        y_corr = profile["y_gain"] * y_err - profile["y_vel_gain"] * y_vel
        delta = profile["x_gain"] * xerr - profile["x_vel_gain"] * xvel - profile["z_gain"] * (float(pelvis[2]) - 0.98)

        ctrl[2] += profile["hip_amp"] * s + profile["speed_hip_gain"] * speed_err
        ctrl[7] -= profile["hip_amp"] * s - profile["speed_hip_gain"] * speed_err
        ctrl[3] += profile["knee_amp"] * pos_s
        ctrl[8] += profile["knee_amp"] * neg_s
        ctrl[4] -= profile["ankle_amp"] * s
        ctrl[9] += profile["ankle_amp"] * s
        ctrl[2] += -0.72 * delta
        ctrl[7] += -0.72 * delta
        ctrl[3] += 0.54 * delta
        ctrl[8] += 0.54 * delta
        ctrl[4] += -0.60 * delta
        ctrl[9] += -0.60 * delta
        ctrl[10] += -0.42 * delta
        ctrl[1] += profile["roll_amp"] * c + y_corr
        ctrl[6] -= profile["roll_amp"] * c + y_corr
        ctrl[10] += (
            profile["torso_bob"] * s
            + profile["speed_torso_gain"] * speed_err
            + profile["progress_torso_gain"] * max(0.0, TARGET_XY[0] - float(pelvis[0]))
        )

        ctrl[11] += 0.05 * s
        ctrl[15] -= 0.05 * s
        ctrl[12] += 0.04 * c + 0.20 * y_corr
        ctrl[16] -= 0.04 * c + 0.20 * y_corr
        ctrl[13] += 0.03 * c
        ctrl[17] -= 0.03 * c

        sim.data.ctrl[:] = ctrl
        sim.step()
        recorder.record(sim)

    for _ in range(profile["settle_steps"]):
        sim.data.ctrl[:] = base
        sim.step()
        recorder.record(sim)

    progress_score, info = score_from_trace(recorder, sim)
    info["profile"] = profile["name"]
    return sim, progress_score, info


def search_score(info: dict) -> float:
    if info.get("success"):
        return 1000.0 + float(info.get("progress_score", 0.0))
    return float(
        3.0 * float(info.get("pelvis_distance_progress", 0.0))
        + 3.0 * float(info.get("box_distance_progress", 0.0))
        + 2.0 * float(info.get("carry_proximity_progress", 0.0))
        + 2.0 * float(info.get("stability_progress", 0.0))
        + 1.0 * float(info.get("box_height_progress", 0.0))
        + 0.25 * float(info.get("efficiency_progress", 0.0))
    )


def transport_score(info: dict) -> float:
    return float(
        2.5 * float(info.get("box_distance_progress", 0.0))
        + 2.5 * float(info.get("pelvis_distance_progress", 0.0))
        + 1.5 * float(info.get("box_checkpoint_progress", 0.0))
        + 1.0 * float(info.get("pelvis_checkpoint_progress", 0.0))
        + 1.0 * float(info.get("carry_proximity_progress", 0.0))
        + 0.5 * float(info.get("stability_progress", 0.0))
    )


def mutate_profile(base: CarryProfile, rng: np.random.Generator, name: str, scale: float = 1.0) -> CarryProfile:
    def jitter(value, sigma, lo, hi):
        return float(np.clip(value + rng.normal(0.0, sigma * scale), lo, hi))

    return CarryProfile(
        name=name,
        hip_pitch=jitter(base.hip_pitch, 0.03, -0.38, -0.05),
        knee=jitter(base.knee, 0.04, -0.55, -0.20),
        ankle=jitter(base.ankle, 0.03, -0.50, -0.15),
        torso=jitter(base.torso, 0.03, -0.35, -0.05),
        shoulder_pitch=jitter(base.shoulder_pitch, 0.03, -0.15, 0.10),
        elbow=jitter(base.elbow, 0.03, -0.15, 0.15),
        arm_pitch=jitter(base.arm_pitch, 0.04, 0.10, 0.50),
        arm_roll=jitter(base.arm_roll, 0.05, 0.50, 1.10),
        arm_yaw=jitter(base.arm_yaw, 0.04, 0.20, 0.60),
        arm_elbow=jitter(base.arm_elbow, 0.05, 0.40, 1.10),
        phase_start=jitter(base.phase_start, 0.08, 0.20, 0.85),
        phase_duration=jitter(base.phase_duration, 0.15, 1.20, 2.80),
        x_gain=jitter(base.x_gain, 0.10, 0.55, 1.35),
        x_vel_gain=jitter(base.x_vel_gain, 0.02, 0.04, 0.18),
        z_gain=jitter(base.z_gain, 0.08, 0.35, 0.95),
        y_gain=jitter(base.y_gain, 0.12, 1.0, 3.0),
        y_vel_gain=jitter(base.y_vel_gain, 0.08, 0.25, 1.0),
        horizon=jitter(base.horizon, 0.30, 3.50, 7.50),
        settle_steps=int(round(jitter(float(base.settle_steps), 15.0, 80.0, 220.0))),
    )


CARRY_WALK_PROFILES = [
    {
        "name": "carry_walk_stable",
        "stance_hip": -0.27,
        "stance_knee": -0.45,
        "stance_ankle": -0.42,
        "torso": -0.26,
        "arm_pitch": -0.06,
        "arm_roll": 0.78,
        "arm_yaw": 0.32,
        "arm_elbow": 0.76,
        "freq": 1.10,
        "phase": 0.00,
        "hip_amp": 0.16,
        "knee_amp": 0.30,
        "ankle_amp": 0.18,
        "roll_amp": 0.06,
        "torso_bob": 0.02,
        "target_speed": 0.34,
        "speed_hip_gain": 0.18,
        "speed_torso_gain": 0.14,
        "progress_torso_gain": 0.18,
        "x_gain": 0.72,
        "x_vel_gain": 0.10,
        "z_gain": 0.62,
        "y_gain": 1.60,
        "y_vel_gain": 0.56,
        "duration": 4.0,
        "settle_steps": 140,
    },
    {
        "name": "carry_walk_forward",
        "stance_hip": -0.25,
        "stance_knee": -0.43,
        "stance_ankle": -0.40,
        "torso": -0.24,
        "arm_pitch": -0.04,
        "arm_roll": 0.80,
        "arm_yaw": 0.34,
        "arm_elbow": 0.78,
        "freq": 1.28,
        "phase": 0.35,
        "hip_amp": 0.24,
        "knee_amp": 0.42,
        "ankle_amp": 0.22,
        "roll_amp": 0.08,
        "torso_bob": 0.03,
        "target_speed": 0.48,
        "speed_hip_gain": 0.22,
        "speed_torso_gain": 0.18,
        "progress_torso_gain": 0.24,
        "x_gain": 0.86,
        "x_vel_gain": 0.11,
        "z_gain": 0.58,
        "y_gain": 1.80,
        "y_vel_gain": 0.62,
        "duration": 4.4,
        "settle_steps": 140,
    },
    {
        "name": "carry_walk_aggressive",
        "stance_hip": -0.23,
        "stance_knee": -0.40,
        "stance_ankle": -0.38,
        "torso": -0.22,
        "arm_pitch": -0.03,
        "arm_roll": 0.82,
        "arm_yaw": 0.36,
        "arm_elbow": 0.80,
        "freq": 1.46,
        "phase": -0.20,
        "hip_amp": 0.30,
        "knee_amp": 0.52,
        "ankle_amp": 0.26,
        "roll_amp": 0.10,
        "torso_bob": 0.04,
        "target_speed": 0.62,
        "speed_hip_gain": 0.26,
        "speed_torso_gain": 0.22,
        "progress_torso_gain": 0.28,
        "x_gain": 1.00,
        "x_vel_gain": 0.12,
        "z_gain": 0.52,
        "y_gain": 2.00,
        "y_vel_gain": 0.68,
        "duration": 4.8,
        "settle_steps": 120,
    },
]


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
        candidates.append((score, info, sim, profile))
        print(json.dumps({"candidate": profile.name, "score": score, "info": info}, indent=2))

    sim, score, info = run_open_loop_cradle_walk()
    candidates.append((score, info, sim, {"name": info["profile"]}))
    print(json.dumps({"candidate": info["profile"], "score": score, "info": info}, indent=2))

    for profile in CARRY_WALK_PROFILES:
        sim, score, info = run_open_loop_carry_walk(profile)
        candidates.append((score, info, sim, profile))
        print(json.dumps({"candidate": profile["name"], "score": score, "info": info}, indent=2))

    rng = np.random.default_rng(7)
    base_by_name = {profile.name: profile for profile in PROFILES}
    for base_name, count, scale in (
        ("stable_bias", 18, 1.0),
        ("forward_bias", 10, 0.85),
        ("balanced", 8, 0.75),
    ):
        base = base_by_name[base_name]
        for i in range(count):
            candidate_profile = mutate_profile(base, rng, f"{base.name}_rand{i:02d}", scale=scale)
            sim, score, info = run_profile(candidate_profile)
            candidates.append((score, info, sim, candidate_profile))
            print(json.dumps({"candidate": candidate_profile.name, "score": score, "info": info}, indent=2))

    transport_seed = max(candidates, key=lambda item: transport_score(item[1]))
    seed_profile = transport_seed[3]
    if isinstance(seed_profile, CarryProfile):
        for i in range(24):
            candidate_profile = mutate_profile(seed_profile, rng, f"{seed_profile.name}_lift{i:02d}", scale=0.55)
            # Nudge the strongest transport candidate a little more upright so
            # the carried box can stay above the threshold while moving.
            candidate_profile = CarryProfile(
                name=candidate_profile.name,
                hip_pitch=max(candidate_profile.hip_pitch, -0.24),
                knee=max(candidate_profile.knee, -0.38),
                ankle=max(candidate_profile.ankle, -0.36),
                torso=max(candidate_profile.torso, -0.22),
                shoulder_pitch=min(candidate_profile.shoulder_pitch, 0.02),
                elbow=candidate_profile.elbow,
                arm_pitch=min(candidate_profile.arm_pitch + 0.05, 0.55),
                arm_roll=min(candidate_profile.arm_roll + 0.06, 1.15),
                arm_yaw=candidate_profile.arm_yaw,
                arm_elbow=min(candidate_profile.arm_elbow + 0.08, 1.20),
                phase_start=candidate_profile.phase_start,
                phase_duration=candidate_profile.phase_duration,
                x_gain=max(candidate_profile.x_gain, 0.80),
                x_vel_gain=candidate_profile.x_vel_gain,
                z_gain=max(candidate_profile.z_gain, 0.55),
                y_gain=max(candidate_profile.y_gain, 1.50),
                y_vel_gain=max(candidate_profile.y_vel_gain, 0.45),
                horizon=min(candidate_profile.horizon + 0.40, 8.0),
                settle_steps=int(min(candidate_profile.settle_steps + 20, 240)),
            )
            sim, score, info = run_profile(candidate_profile)
            candidates.append((score, info, sim, candidate_profile))
            print(json.dumps({"candidate": candidate_profile.name, "score": score, "info": info}, indent=2))

    candidates.sort(key=lambda item: (1 if item[1].get("success") else 0, search_score(item[1])), reverse=True)
    best_score, best_info, best_sim, best_profile = candidates[0]
    best_sim.save_final_state("/work/final_state.npz")
    print(json.dumps({"selected": best_info["profile"], "score": best_score, "info": best_info}, indent=2))
    print("saved /work/final_state.npz")


if __name__ == "__main__":
    main()
