from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sim import MIN_PELVIS_Z, MIN_TORSO_UP, Sim, TARGET_XY, TOUCH_POINT


QPOS_IDX = np.arange(7, 26)
QVEL_IDX = np.arange(6, 25)
CTRL_MIN = None
CTRL_MAX = None
CHECKPOINTS = (0.25, 0.55, 0.80)


def quat_to_euler_xyz(quat):
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


@dataclass
class EvalResult:
    passed: bool
    progress_score: float
    checkpoint_progress: float
    stability_progress: float
    touch_progress: float
    distance_progress: float
    settle_posture_progress: float
    efficiency_progress: float
    replay_steps: int
    final_target_distance: float
    best_touch_distance: float
    stable_trace_fraction: float
    checkpoints_hit: int
    settle_min_pelvis_z: float
    settle_min_torso_up: float
    settle_touch_distance: float
    final_pelvis_x: float
    final_pelvis_z: float
    final_torso_up: float


class GaitController:
    def __init__(self, sim: Sim, params: dict):
        self.sim = sim
        self.params = params
        self.home = sim.home_ctrl().copy()
        self.kp = np.array(
            [80, 120, 220, 260, 120, 80, 120, 220, 260, 120, 220, 70, 50, 35, 20, 70, 50, 35, 20],
            dtype=float,
        )
        self.kd = 0.8 * 2.0 * np.sqrt(self.kp)
        self.base_walk = self.home.copy()
        self.base_walk[2] = params["stance_hip"]
        self.base_walk[3] = params["stance_knee"]
        self.base_walk[4] = params["stance_ankle"]
        self.base_walk[7] = params["stance_hip"]
        self.base_walk[8] = params["stance_knee"]
        self.base_walk[9] = params["stance_ankle"]
        self.base_walk[10] = params["torso_lean"]
        self.base_walk[11] = params["arm_pitch"]
        self.base_walk[15] = params["arm_pitch"]

        self.base_final = self.base_walk.copy()
        self.base_final[2] = params["final_hip"]
        self.base_final[3] = params["final_knee"]
        self.base_final[4] = params["final_ankle"]
        self.base_final[7] = params["final_hip"]
        self.base_final[8] = params["final_knee"]
        self.base_final[9] = params["final_ankle"]
        self.base_final[10] = params["final_torso"]
        self.base_final[11] = params["left_shoulder_pitch"]
        self.base_final[12] = params["left_shoulder_roll"]
        self.base_final[13] = params["left_shoulder_yaw"]
        self.base_final[14] = params["left_elbow"]
        self.base_final[15] = params["right_shoulder_pitch"]
        self.base_final[16] = params["right_shoulder_roll"]
        self.base_final[17] = params["right_shoulder_yaw"]
        self.base_final[18] = params["right_elbow"]

    def desired_qpos(self):
        data = self.sim.data
        t = data.time
        pelvis_x = self.sim.pelvis_position()[0]
        roll, pitch, _ = quat_to_euler_xyz(data.qpos[3:7])
        ang_roll = data.qvel[3]
        ang_pitch = data.qvel[4]
        yaw_amp = self.params.get("yaw_amp", 0.0)

        if t < self.params["walk_time"]:
            qdes = self.base_walk.copy()
            phase = 2.0 * math.pi * self.params["freq"] * t + self.params["phase"]
            s = math.sin(phase)
            c = math.cos(phase)
            pos_s = max(0.0, s)
            neg_s = max(0.0, -s)

            qdes[0] += yaw_amp * c
            qdes[5] -= yaw_amp * c
            qdes[2] += self.params["hip_amp"] * s
            qdes[7] -= self.params["hip_amp"] * s
            qdes[3] += self.params["knee_amp"] * pos_s
            qdes[8] += self.params["knee_amp"] * neg_s
            qdes[4] -= self.params["ankle_amp"] * s
            qdes[9] += self.params["ankle_amp"] * s
            qdes[1] += self.params["hip_roll_amp"] * c
            qdes[6] -= self.params["hip_roll_amp"] * c
            qdes[12] = self.params["arm_roll_amp"] * c
            qdes[16] = -self.params["arm_roll_amp"] * c
            qdes[11] = self.params["arm_pitch"] + self.params["arm_swing_amp"] * s
            qdes[15] = self.params["arm_pitch"] - self.params["arm_swing_amp"] * s

            speed_err = self.params["target_speed"] - data.qvel[0]
            qdes[2] += self.params["speed_hip_gain"] * speed_err
            qdes[7] += self.params["speed_hip_gain"] * speed_err
            qdes[10] += self.params["speed_torso_gain"] * speed_err
            qdes[10] += self.params["progress_torso_gain"] * max(0.0, TARGET_XY[0] - pelvis_x)
        else:
            qdes = self.base_final.copy()
            blend = min(1.0, (t - self.params["walk_time"]) / self.params["blend_time"])
            qdes = (1.0 - blend) * self.base_walk + blend * qdes
            reach = clamp((t - self.params["walk_time"]) / max(0.6, self.params["blend_time"]), 0.0, 1.0)
            qdes[0] += 0.35 * yaw_amp * reach
            qdes[5] -= 0.35 * yaw_amp * reach
            qdes[15] += 0.35 * reach
            qdes[16] -= 0.10 * reach
            qdes[17] += 0.20 * reach
            qdes[18] += 0.45 * reach

        qdes[1] += -self.params["roll_gain"] * roll - self.params["roll_rate_gain"] * ang_roll
        qdes[6] += -self.params["roll_gain"] * roll - self.params["roll_rate_gain"] * ang_roll
        qdes[4] += -self.params["roll_ankle_gain"] * roll - self.params["roll_ankle_rate_gain"] * ang_roll
        qdes[9] += -self.params["roll_ankle_gain"] * roll - self.params["roll_ankle_rate_gain"] * ang_roll
        qdes[2] += -self.params["pitch_gain"] * pitch - self.params["pitch_rate_gain"] * ang_pitch
        qdes[7] += -self.params["pitch_gain"] * pitch - self.params["pitch_rate_gain"] * ang_pitch
        qdes[4] += -self.params["pitch_ankle_gain"] * pitch - self.params["pitch_ankle_rate_gain"] * ang_pitch
        qdes[9] += -self.params["pitch_ankle_gain"] * pitch - self.params["pitch_ankle_rate_gain"] * ang_pitch
        qdes[10] += -self.params["torso_pitch_gain"] * pitch - self.params["torso_pitch_rate_gain"] * ang_pitch

        y_err = TARGET_XY[1] - data.qpos[1]
        y_vel = data.qvel[1]
        y_correction = self.params.get("y_gain", 0.0) * y_err - self.params.get("y_vel_gain", 0.0) * y_vel
        qdes[1] += y_correction
        qdes[6] -= y_correction
        qdes[12] += 0.20 * y_correction
        qdes[16] -= 0.20 * y_correction
        return qdes

    def torque(self):
        q = self.sim.data.qpos[QPOS_IDX]
        qvel = self.sim.data.qvel[QVEL_IDX]
        qdes = self.desired_qpos()
        bias = self.sim.data.qfrc_bias[QVEL_IDX]
        torque = bias + self.kp * (qdes - q) - self.kd * qvel
        return np.clip(torque, CTRL_MIN, CTRL_MAX)


def evaluate_trace(sim: Sim):
    ctrl_trace = np.array(sim._ctrl_trace, dtype=float)
    replay_steps = len(ctrl_trace)

    replay_trace = sim._trace
    replay_pelvis_x = np.array([item["pelvis_pos"][0] for item in replay_trace])
    replay_pelvis_z = np.array([item["pelvis_pos"][2] for item in replay_trace])
    replay_torso_up = np.array([item["torso_up"] for item in replay_trace])
    replay_touch = np.array([item["touch_distance"] for item in replay_trace])
    checkpoints_hit = sum(float(np.max(replay_pelvis_x) >= x) for x in CHECKPOINTS)
    checkpoint_progress = checkpoints_hit / len(CHECKPOINTS)
    stable_fraction = float(np.mean((replay_pelvis_z >= MIN_PELVIS_Z) & (replay_torso_up >= MIN_TORSO_UP)))
    best_touch_distance = float(np.min(replay_touch))

    settle_min_pelvis_z = float("inf")
    settle_min_torso_up = float("inf")
    settle_touch_distance = float("inf")
    for _ in range(500):
        sim.step(1)
        settle_min_pelvis_z = min(settle_min_pelvis_z, sim.pelvis_position()[2])
        settle_min_torso_up = min(settle_min_torso_up, sim.torso_up())
        settle_touch_distance = min(settle_touch_distance, sim.distance_to_touch())

    final_target_distance = float(np.linalg.norm(sim.pelvis_position()[:2] - TARGET_XY))
    stability_progress = max(0.0, min(1.0, stable_fraction / 0.92))
    touch_progress = max(0.0, min(1.0, 1.0 - best_touch_distance / 0.12))
    distance_progress = max(0.0, min(1.0, 1.0 - final_target_distance / 1.00))
    efficiency_progress = max(0.0, min(1.0, 2200.0 / max(replay_steps, 1)))
    s1 = checkpoint_progress
    s2 = checkpoint_progress * stability_progress
    s3 = checkpoint_progress * stability_progress * touch_progress
    s4 = checkpoint_progress * stability_progress * touch_progress * distance_progress
    progress_score = (s1 + s2 + s3 + s4) / 4.0

    passed = (
        replay_steps >= 30
        and np.linalg.norm(replay_trace[0]["pelvis_pos"][:2]) <= 0.15
        and checkpoints_hit == len(CHECKPOINTS)
        and stable_fraction >= 0.92
        and best_touch_distance <= 0.12
        and final_target_distance <= 0.22
        and settle_min_pelvis_z >= 0.75
        and settle_min_torso_up >= 0.78
        and settle_touch_distance <= 0.12
    )
    settle_posture_progress = 0.5 * max(0.0, min(1.0, settle_min_pelvis_z / 0.75)) + 0.5 * max(0.0, min(1.0, settle_min_torso_up / 0.78))
    return EvalResult(
        passed=passed,
        progress_score=progress_score,
        checkpoint_progress=checkpoint_progress,
        stability_progress=stability_progress,
        touch_progress=touch_progress,
        distance_progress=distance_progress,
        settle_posture_progress=settle_posture_progress,
        efficiency_progress=efficiency_progress,
        replay_steps=replay_steps,
        final_target_distance=final_target_distance,
        best_touch_distance=best_touch_distance,
        stable_trace_fraction=stable_fraction,
        checkpoints_hit=checkpoints_hit,
        settle_min_pelvis_z=settle_min_pelvis_z,
        settle_min_torso_up=settle_min_torso_up,
        settle_touch_distance=settle_touch_distance,
        final_pelvis_x=float(sim.pelvis_position()[0]),
        final_pelvis_z=float(sim.pelvis_position()[2]),
        final_torso_up=float(sim.torso_up()),
    )


def rollout(params, save_path=None):
    sim = Sim()
    global CTRL_MIN, CTRL_MAX
    CTRL_MIN = sim.model.actuator_ctrlrange[:, 0]
    CTRL_MAX = sim.model.actuator_ctrlrange[:, 1]
    controller = GaitController(sim, params)
    total_steps = int(params["total_time"] / sim.model.opt.timestep)
    for _ in range(total_steps):
        sim.data.ctrl[:] = controller.torque()
        sim.step(1)
    result = evaluate_trace(sim)
    if save_path is not None:
        sim.save_final_state(save_path)
    return result, sim


def sample_params(rng, center=None, scale=1.0):
    if center is None:
        center = {}

    def pick(name, lo, hi):
        if name not in center:
            return rng.uniform(lo, hi)
        span = (hi - lo) * 0.25 * scale
        return float(np.clip(center[name] + rng.normal(0.0, span), lo, hi))

    return {
        "stance_hip": pick("stance_hip", -0.55, -0.10),
        "stance_knee": pick("stance_knee", 0.35, 1.00),
        "stance_ankle": pick("stance_ankle", -0.55, -0.05),
        "torso_lean": pick("torso_lean", -0.05, 0.40),
        "arm_pitch": pick("arm_pitch", -0.6, 0.8),
        "freq": pick("freq", 0.8, 2.8),
        "phase": pick("phase", -math.pi, math.pi),
        "hip_amp": pick("hip_amp", 0.05, 0.65),
        "knee_amp": pick("knee_amp", 0.05, 1.00),
        "ankle_amp": pick("ankle_amp", 0.02, 0.50),
        "hip_roll_amp": pick("hip_roll_amp", 0.0, 0.20),
        "arm_swing_amp": pick("arm_swing_amp", 0.0, 0.9),
        "arm_roll_amp": pick("arm_roll_amp", 0.0, 0.7),
        "yaw_amp": pick("yaw_amp", 0.0, 0.20),
        "target_speed": pick("target_speed", 0.2, 1.4),
        "speed_hip_gain": pick("speed_hip_gain", 0.0, 0.6),
        "speed_torso_gain": pick("speed_torso_gain", 0.0, 0.6),
        "progress_torso_gain": pick("progress_torso_gain", 0.0, 0.5),
        "y_gain": pick("y_gain", -1.0, 8.0),
        "y_vel_gain": pick("y_vel_gain", -0.5, 3.0),
        "roll_gain": pick("roll_gain", -0.6, 1.2),
        "roll_rate_gain": pick("roll_rate_gain", -0.2, 0.5),
        "roll_ankle_gain": pick("roll_ankle_gain", -0.6, 1.2),
        "roll_ankle_rate_gain": pick("roll_ankle_rate_gain", -0.2, 0.5),
        "pitch_gain": pick("pitch_gain", -0.6, 1.2),
        "pitch_rate_gain": pick("pitch_rate_gain", -0.2, 0.6),
        "pitch_ankle_gain": pick("pitch_ankle_gain", -0.6, 1.2),
        "pitch_ankle_rate_gain": pick("pitch_ankle_rate_gain", -0.2, 0.6),
        "torso_pitch_gain": pick("torso_pitch_gain", -0.6, 1.2),
        "torso_pitch_rate_gain": pick("torso_pitch_rate_gain", -0.2, 0.6),
        "walk_time": pick("walk_time", 1.0, 3.2),
        "blend_time": pick("blend_time", 0.1, 1.0),
        "total_time": pick("total_time", 1.2, 3.4),
        "final_hip": pick("final_hip", -0.55, 0.05),
        "final_knee": pick("final_knee", 0.25, 1.10),
        "final_ankle": pick("final_ankle", -0.60, 0.10),
        "final_torso": pick("final_torso", -0.15, 0.65),
        "left_shoulder_pitch": pick("left_shoulder_pitch", -1.8, 2.0),
        "left_shoulder_roll": pick("left_shoulder_roll", 0.0, 2.6),
        "left_shoulder_yaw": pick("left_shoulder_yaw", -0.8, 2.8),
        "left_elbow": pick("left_elbow", 0.0, 2.4),
        "right_shoulder_pitch": pick("right_shoulder_pitch", -1.5, 1.8),
        "right_shoulder_roll": pick("right_shoulder_roll", -2.6, 0.0),
        "right_shoulder_yaw": pick("right_shoulder_yaw", -2.8, 0.8),
        "right_elbow": pick("right_elbow", 0.0, 2.4),
    }


def format_result(result):
    return {
        "passed": result.passed,
        "progress_score": round(result.progress_score, 4),
        "checkpoints_hit": result.checkpoints_hit,
        "stable_trace_fraction": round(result.stable_trace_fraction, 4),
        "best_touch_distance": round(result.best_touch_distance, 4),
        "final_target_distance": round(result.final_target_distance, 4),
        "settle_min_pelvis_z": round(result.settle_min_pelvis_z, 4),
        "settle_min_torso_up": round(result.settle_min_torso_up, 4),
        "settle_touch_distance": round(result.settle_touch_distance, 4),
        "final_pelvis_x": round(result.final_pelvis_x, 4),
        "final_pelvis_z": round(result.final_pelvis_z, 4),
        "final_torso_up": round(result.final_torso_up, 4),
    }


def tweak(base: dict, **updates) -> dict:
    out = dict(base)
    out.update(updates)
    return out


def witness_score(result: EvalResult) -> float:
    if result.passed:
        return 1e6
    touch_term = 3.0 * result.touch_progress
    checkpoint_term = 2.0 * result.checkpoint_progress
    stability_term = 2.0 * result.stability_progress
    distance_term = 1.8 * result.distance_progress
    posture_term = 1.2 * result.settle_posture_progress
    efficiency_term = 0.1 * result.efficiency_progress
    coupled_term = 3.0 * (
        result.checkpoint_progress
        * result.stability_progress
        * result.touch_progress
        * max(result.distance_progress, 0.35)
    )
    near_touch_bonus = 0.3 * max(0.0, min(1.0, 1.0 - result.best_touch_distance / 0.35))
    checkpoint_bonus = 0.5 if result.checkpoints_hit == 3 else 0.0
    return float(
        touch_term
        + checkpoint_term
        + stability_term
        + distance_term
        + posture_term
        + efficiency_term
        + coupled_term
        + near_touch_bonus
        + checkpoint_bonus
    )


def main():
    rng = np.random.default_rng(0)
    save_path = "/work/final_state.npz"
    best_params = None
    best_result = None
    best_sim = None

    base_path = Path(__file__).with_name("best_params.json")
    base = json.loads(base_path.read_text(encoding="utf-8"))["params"]
    seed_bank = [
        base,
        tweak(base, stance_hip=-0.50, stance_knee=1.02, stance_ankle=-0.52, torso_lean=0.07, arm_pitch=-0.18),
        tweak(base, freq=1.36, hip_amp=0.36, knee_amp=0.82, ankle_amp=0.10, arm_swing_amp=0.44),
        tweak(base, walk_time=1.95, blend_time=0.12, total_time=1.75, final_torso=-0.10, right_shoulder_pitch=1.95),
        tweak(
            base,
            walk_time=1.80,
            blend_time=0.10,
            total_time=2.10,
            final_torso=0.12,
            left_shoulder_pitch=1.65,
            left_shoulder_roll=0.35,
            left_shoulder_yaw=-0.20,
            left_elbow=0.25,
            right_shoulder_pitch=2.42,
            right_shoulder_roll=-1.15,
            right_shoulder_yaw=0.65,
            right_elbow=1.95,
        ),
        tweak(
            base,
            freq=1.24,
            hip_amp=0.28,
            knee_amp=0.72,
            ankle_amp=0.12,
            hip_roll_amp=0.03,
            arm_swing_amp=0.18,
            arm_roll_amp=0.08,
            target_speed=0.72,
            speed_hip_gain=0.35,
            speed_torso_gain=0.28,
            progress_torso_gain=0.22,
            roll_gain=0.25,
            roll_rate_gain=0.10,
            roll_ankle_gain=0.24,
            roll_ankle_rate_gain=0.08,
            pitch_gain=0.38,
            pitch_rate_gain=0.12,
            pitch_ankle_gain=0.30,
            pitch_ankle_rate_gain=0.10,
            torso_pitch_gain=0.24,
            torso_pitch_rate_gain=0.10,
            walk_time=2.20,
            blend_time=0.18,
            total_time=2.35,
            final_torso=0.00,
            left_shoulder_pitch=1.20,
            left_shoulder_roll=0.20,
            left_shoulder_yaw=-0.10,
            left_elbow=0.30,
            right_shoulder_pitch=2.25,
            right_shoulder_roll=-0.95,
            right_shoulder_yaw=0.45,
            right_elbow=1.65,
        ),
        tweak(
            base,
            freq=1.18,
            hip_amp=0.26,
            knee_amp=0.66,
            ankle_amp=0.10,
            hip_roll_amp=0.01,
            yaw_amp=0.00,
            arm_swing_amp=0.06,
            arm_roll_amp=0.04,
            target_speed=0.68,
            speed_hip_gain=0.28,
            speed_torso_gain=0.22,
            progress_torso_gain=0.20,
            roll_gain=0.18,
            roll_rate_gain=0.06,
            roll_ankle_gain=0.18,
            roll_ankle_rate_gain=0.05,
            pitch_gain=0.30,
            pitch_rate_gain=0.10,
            pitch_ankle_gain=0.24,
            pitch_ankle_rate_gain=0.08,
            torso_pitch_gain=0.20,
            torso_pitch_rate_gain=0.08,
            y_gain=3.20,
            y_vel_gain=1.20,
            walk_time=2.25,
            blend_time=0.18,
            total_time=2.40,
            final_torso=0.02,
            left_shoulder_pitch=1.10,
            left_shoulder_roll=0.10,
            left_shoulder_yaw=-0.05,
            left_elbow=0.20,
            right_shoulder_pitch=2.55,
            right_shoulder_roll=-1.25,
            right_shoulder_yaw=0.60,
            right_elbow=2.05,
        ),
        tweak(
            base,
            freq=1.48,
            hip_amp=0.32,
            knee_amp=0.68,
            ankle_amp=0.20,
            hip_roll_amp=0.06,
            arm_swing_amp=0.28,
            arm_roll_amp=0.12,
            target_speed=0.82,
            speed_hip_gain=0.40,
            speed_torso_gain=0.30,
            progress_torso_gain=0.25,
            roll_gain=0.28,
            roll_rate_gain=0.12,
            roll_ankle_gain=0.26,
            roll_ankle_rate_gain=0.10,
            pitch_gain=0.42,
            pitch_rate_gain=0.14,
            pitch_ankle_gain=0.34,
            pitch_ankle_rate_gain=0.12,
            torso_pitch_gain=0.28,
            torso_pitch_rate_gain=0.12,
            y_gain=5.0,
            y_vel_gain=2.0,
            walk_time=1.95,
            blend_time=0.16,
            total_time=2.10,
            final_torso=0.16,
            left_shoulder_pitch=2.25,
            left_shoulder_roll=1.00,
            left_shoulder_yaw=-0.10,
            left_elbow=1.85,
            right_shoulder_pitch=2.55,
            right_shoulder_roll=-1.10,
            right_shoulder_yaw=0.55,
            right_elbow=2.10,
        ),
        tweak(
            base,
            freq=1.62,
            hip_amp=0.36,
            knee_amp=0.74,
            ankle_amp=0.22,
            hip_roll_amp=0.08,
            arm_swing_amp=0.35,
            arm_roll_amp=0.14,
            target_speed=0.90,
            speed_hip_gain=0.45,
            speed_torso_gain=0.32,
            progress_torso_gain=0.28,
            roll_gain=0.30,
            roll_rate_gain=0.12,
            roll_ankle_gain=0.28,
            roll_ankle_rate_gain=0.10,
            pitch_gain=0.45,
            pitch_rate_gain=0.16,
            pitch_ankle_gain=0.36,
            pitch_ankle_rate_gain=0.12,
            torso_pitch_gain=0.30,
            torso_pitch_rate_gain=0.12,
            y_gain=6.5,
            y_vel_gain=2.5,
            walk_time=2.05,
            blend_time=0.18,
            total_time=2.20,
            final_torso=0.18,
            left_shoulder_pitch=2.35,
            left_shoulder_roll=1.05,
            left_shoulder_yaw=-0.12,
            left_elbow=1.95,
            right_shoulder_pitch=2.65,
            right_shoulder_roll=-1.20,
            right_shoulder_yaw=0.60,
            right_elbow=2.20,
        ),
        tweak(
            base,
            freq=1.42,
            hip_amp=0.30,
            knee_amp=0.64,
            ankle_amp=0.14,
            hip_roll_amp=0.05,
            yaw_amp=0.08,
            arm_swing_amp=0.16,
            arm_roll_amp=0.10,
            target_speed=0.78,
            speed_hip_gain=0.34,
            speed_torso_gain=0.26,
            progress_torso_gain=0.24,
            y_gain=4.5,
            y_vel_gain=1.50,
            roll_gain=0.24,
            roll_rate_gain=0.08,
            roll_ankle_gain=0.22,
            roll_ankle_rate_gain=0.08,
            pitch_gain=0.34,
            pitch_rate_gain=0.10,
            pitch_ankle_gain=0.28,
            pitch_ankle_rate_gain=0.10,
            torso_pitch_gain=0.24,
            torso_pitch_rate_gain=0.08,
            walk_time=1.76,
            blend_time=0.12,
            total_time=1.98,
            final_torso=0.24,
            left_shoulder_pitch=2.70,
            left_shoulder_roll=1.15,
            left_shoulder_yaw=-0.15,
            left_elbow=2.25,
            right_shoulder_pitch=2.85,
            right_shoulder_roll=-1.28,
            right_shoulder_yaw=0.66,
            right_elbow=2.40,
        ),
        tweak(
            base,
            freq=1.55,
            hip_amp=0.34,
            knee_amp=0.70,
            ankle_amp=0.16,
            hip_roll_amp=0.07,
            yaw_amp=0.10,
            arm_swing_amp=0.22,
            arm_roll_amp=0.12,
            target_speed=0.85,
            speed_hip_gain=0.38,
            speed_torso_gain=0.28,
            progress_torso_gain=0.26,
            y_gain=6.0,
            y_vel_gain=2.10,
            roll_gain=0.28,
            roll_rate_gain=0.10,
            roll_ankle_gain=0.25,
            roll_ankle_rate_gain=0.09,
            pitch_gain=0.38,
            pitch_rate_gain=0.12,
            pitch_ankle_gain=0.32,
            pitch_ankle_rate_gain=0.11,
            torso_pitch_gain=0.28,
            torso_pitch_rate_gain=0.10,
            walk_time=1.84,
            blend_time=0.12,
            total_time=2.05,
            final_torso=0.28,
            left_shoulder_pitch=2.80,
            left_shoulder_roll=1.20,
            left_shoulder_yaw=-0.18,
            left_elbow=2.35,
            right_shoulder_pitch=2.95,
            right_shoulder_roll=-1.35,
            right_shoulder_yaw=0.72,
            right_elbow=2.45,
        ),
        tweak(
            base,
            stance_hip=-0.51,
            stance_knee=0.85,
            stance_ankle=-0.35,
            torso_lean=-0.05,
            freq=1.61,
            phase=-2.04,
            hip_amp=0.11,
            knee_amp=0.85,
            ankle_amp=0.32,
            hip_roll_amp=0.20,
            arm_swing_amp=0.31,
            arm_roll_amp=0.03,
            yaw_amp=0.13,
            target_speed=1.12,
            speed_hip_gain=0.07,
            speed_torso_gain=0.38,
            progress_torso_gain=0.42,
            y_gain=2.73,
            y_vel_gain=-0.44,
            roll_gain=-0.34,
            roll_rate_gain=0.08,
            roll_ankle_gain=0.30,
            roll_ankle_rate_gain=0.07,
            pitch_gain=-0.46,
            pitch_rate_gain=-0.11,
            pitch_ankle_gain=1.12,
            pitch_ankle_rate_gain=0.01,
            torso_pitch_gain=0.67,
            torso_pitch_rate_gain=-0.03,
            walk_time=1.00,
            blend_time=0.28,
            total_time=2.06,
            final_hip=-0.31,
            final_knee=1.09,
            final_ankle=-0.54,
            final_torso=-0.15,
            left_shoulder_pitch=1.15,
            left_shoulder_roll=2.47,
            left_shoulder_yaw=1.29,
            left_elbow=0.86,
            right_shoulder_pitch=-0.23,
            right_shoulder_roll=-0.81,
            right_shoulder_yaw=0.80,
            right_elbow=0.54,
        ),
        tweak(
            base,
            stance_hip=-0.51,
            stance_knee=0.85,
            stance_ankle=-0.35,
            torso_lean=-0.03,
            freq=1.58,
            phase=-2.00,
            hip_amp=0.12,
            knee_amp=0.83,
            ankle_amp=0.30,
            hip_roll_amp=0.18,
            arm_swing_amp=0.28,
            arm_roll_amp=0.02,
            yaw_amp=0.10,
            target_speed=1.10,
            speed_hip_gain=0.08,
            speed_torso_gain=0.36,
            progress_torso_gain=0.40,
            y_gain=2.5,
            y_vel_gain=-0.20,
            roll_gain=-0.28,
            roll_rate_gain=0.08,
            roll_ankle_gain=0.26,
            roll_ankle_rate_gain=0.06,
            pitch_gain=-0.40,
            pitch_rate_gain=-0.10,
            pitch_ankle_gain=1.05,
            pitch_ankle_rate_gain=0.02,
            torso_pitch_gain=0.60,
            torso_pitch_rate_gain=-0.02,
            walk_time=1.18,
            blend_time=0.24,
            total_time=2.18,
            final_hip=-0.28,
            final_knee=1.02,
            final_ankle=-0.48,
            final_torso=-0.10,
            left_shoulder_pitch=1.05,
            left_shoulder_roll=2.35,
            left_shoulder_yaw=1.20,
            left_elbow=0.95,
            right_shoulder_pitch=-0.10,
            right_shoulder_roll=-0.72,
            right_shoulder_yaw=0.72,
            right_elbow=0.62,
        ),
        tweak(
            base,
            stance_hip=-0.49,
            stance_knee=0.88,
            stance_ankle=-0.38,
            torso_lean=-0.02,
            freq=1.52,
            phase=-1.90,
            hip_amp=0.14,
            knee_amp=0.82,
            ankle_amp=0.27,
            hip_roll_amp=0.15,
            arm_swing_amp=0.24,
            arm_roll_amp=0.02,
            yaw_amp=0.08,
            target_speed=1.05,
            speed_hip_gain=0.10,
            speed_torso_gain=0.34,
            progress_torso_gain=0.38,
            y_gain=2.2,
            y_vel_gain=0.10,
            roll_gain=-0.24,
            roll_rate_gain=0.07,
            roll_ankle_gain=0.24,
            roll_ankle_rate_gain=0.05,
            pitch_gain=-0.34,
            pitch_rate_gain=-0.08,
            pitch_ankle_gain=0.98,
            pitch_ankle_rate_gain=0.03,
            torso_pitch_gain=0.54,
            torso_pitch_rate_gain=0.00,
            walk_time=1.28,
            blend_time=0.22,
            total_time=2.24,
            final_hip=-0.24,
            final_knee=0.98,
            final_ankle=-0.42,
            final_torso=-0.06,
            left_shoulder_pitch=0.98,
            left_shoulder_roll=2.20,
            left_shoulder_yaw=1.05,
            left_elbow=1.00,
            right_shoulder_pitch=0.00,
            right_shoulder_roll=-0.65,
            right_shoulder_yaw=0.65,
            right_elbow=0.70,
        ),
        tweak(
            base,
            stance_hip=-0.38,
            stance_knee=0.98,
            stance_ankle=-0.39,
            torso_lean=-0.03,
            arm_pitch=-0.12,
            freq=1.96,
            phase=-2.94,
            hip_amp=0.19,
            knee_amp=0.92,
            ankle_amp=0.20,
            hip_roll_amp=0.18,
            arm_swing_amp=0.50,
            arm_roll_amp=0.02,
            yaw_amp=0.16,
            target_speed=0.94,
            speed_hip_gain=0.05,
            speed_torso_gain=0.29,
            progress_torso_gain=0.33,
            y_gain=3.82,
            y_vel_gain=0.56,
            roll_gain=0.20,
            roll_rate_gain=-0.05,
            roll_ankle_gain=0.47,
            roll_ankle_rate_gain=0.11,
            pitch_gain=-0.38,
            pitch_rate_gain=-0.17,
            pitch_ankle_gain=1.15,
            pitch_ankle_rate_gain=-0.07,
            torso_pitch_gain=0.22,
            torso_pitch_rate_gain=-0.12,
            walk_time=1.11,
            blend_time=0.33,
            total_time=2.08,
            final_hip=-0.36,
            final_knee=1.07,
            final_ankle=-0.60,
            final_torso=0.04,
            left_shoulder_pitch=0.81,
            left_shoulder_roll=2.08,
            left_shoulder_yaw=0.48,
            left_elbow=1.36,
            right_shoulder_pitch=-0.31,
            right_shoulder_roll=-0.93,
            right_shoulder_yaw=0.80,
            right_elbow=0.52,
        ),
        tweak(
            base,
            stance_hip=-0.36,
            stance_knee=0.94,
            stance_ankle=-0.34,
            torso_lean=-0.01,
            freq=1.82,
            phase=-2.85,
            hip_amp=0.17,
            knee_amp=0.88,
            ankle_amp=0.18,
            hip_roll_amp=0.14,
            arm_swing_amp=0.42,
            arm_roll_amp=0.02,
            yaw_amp=0.12,
            target_speed=0.98,
            speed_hip_gain=0.08,
            speed_torso_gain=0.32,
            progress_torso_gain=0.36,
            y_gain=3.4,
            y_vel_gain=0.45,
            roll_gain=0.22,
            roll_rate_gain=-0.03,
            roll_ankle_gain=0.44,
            roll_ankle_rate_gain=0.09,
            pitch_gain=-0.34,
            pitch_rate_gain=-0.14,
            pitch_ankle_gain=1.05,
            pitch_ankle_rate_gain=-0.04,
            torso_pitch_gain=0.28,
            torso_pitch_rate_gain=-0.08,
            walk_time=1.22,
            blend_time=0.30,
            total_time=2.55,
            final_hip=-0.30,
            final_knee=0.96,
            final_ankle=-0.48,
            final_torso=0.10,
            left_shoulder_pitch=0.72,
            left_shoulder_roll=1.88,
            left_shoulder_yaw=0.42,
            left_elbow=1.22,
            right_shoulder_pitch=-0.24,
            right_shoulder_roll=-0.82,
            right_shoulder_yaw=0.70,
            right_elbow=0.62,
        ),
        tweak(
            base,
            stance_hip=-0.34,
            stance_knee=0.90,
            stance_ankle=-0.30,
            torso_lean=0.00,
            freq=1.75,
            phase=-2.70,
            hip_amp=0.15,
            knee_amp=0.82,
            ankle_amp=0.16,
            hip_roll_amp=0.12,
            arm_swing_amp=0.35,
            arm_roll_amp=0.01,
            yaw_amp=0.10,
            target_speed=1.00,
            speed_hip_gain=0.10,
            speed_torso_gain=0.34,
            progress_torso_gain=0.40,
            y_gain=3.2,
            y_vel_gain=0.40,
            roll_gain=0.26,
            roll_rate_gain=-0.02,
            roll_ankle_gain=0.40,
            roll_ankle_rate_gain=0.08,
            pitch_gain=-0.30,
            pitch_rate_gain=-0.12,
            pitch_ankle_gain=0.98,
            pitch_ankle_rate_gain=-0.02,
            torso_pitch_gain=0.32,
            torso_pitch_rate_gain=-0.05,
            walk_time=1.30,
            blend_time=0.28,
            total_time=2.90,
            final_hip=-0.24,
            final_knee=0.88,
            final_ankle=-0.40,
            final_torso=0.12,
            left_shoulder_pitch=0.66,
            left_shoulder_roll=1.72,
            left_shoulder_yaw=0.35,
            left_elbow=1.10,
            right_shoulder_pitch=-0.18,
            right_shoulder_roll=-0.74,
            right_shoulder_yaw=0.60,
            right_elbow=0.70,
        ),
        ]

    max_iters = 1200
    for i in range(max_iters):
        if i < len(seed_bank):
            params = seed_bank[i]
        elif best_params is None:
            params = base
        elif rng.random() < 0.8:
            params = sample_params(rng, best_params, scale=0.18 if best_result.progress_score > 0.5 else 0.35)
        else:
            params = sample_params(rng)

        result, sim = rollout(params)
        current_score = witness_score(result)
        best_score = witness_score(best_result) if best_result is not None else float("-inf")
        if best_result is None or current_score > best_score:
            best_result = result
            best_params = dict(params)
            best_sim = sim
            sim.save_final_state(save_path)
            with open("/work/best_params.json", "w", encoding="ascii") as f:
                json.dump({"params": best_params, "result": format_result(best_result)}, f, indent=2)
            print(f"iter {i}: new best score={current_score:.4f} {json.dumps(format_result(best_result))}", flush=True)
        elif i % 10 == 0:
            print(f"iter {i}: score={current_score:.4f} {json.dumps(format_result(result))}", flush=True)

        if result.passed:
            break

    if best_result is None:
        raise RuntimeError("no rollouts completed")
    if best_sim is not None:
        best_sim.save_final_state(save_path)
    print("best", json.dumps(format_result(best_result)), flush=True)


if __name__ == "__main__":
    main()
