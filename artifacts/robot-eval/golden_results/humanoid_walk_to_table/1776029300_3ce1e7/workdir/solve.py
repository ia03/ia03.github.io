from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from sim import MIN_PELVIS_Z, MIN_TORSO_UP, Sim, TARGET_XY


QPOS_IDX = np.arange(7, 26)
QVEL_IDX = np.arange(6, 25)
CTRL_MIN = None
CTRL_MAX = None
CHECKPOINTS = (0.20, 0.40, 0.60)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def quat_to_euler_xyz(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = clamp(sinp, -1.0, 1.0)
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


@dataclass
class EvalResult:
    passed: bool
    score: float
    checkpoints_hit: int
    stable_fraction: float
    final_target_distance: float
    settle_min_pelvis_z: float
    settle_min_torso_up: float
    replay_steps: int
    final_pelvis_x: float
    final_pelvis_y: float


class GaitController:
    def __init__(self, sim: Sim, params: dict):
        self.sim = sim
        self.params = params
        self.home = sim.home_ctrl().copy()
        self.kp = np.array(
            [90, 140, 280, 340, 150, 90, 140, 280, 340, 150, 250, 60, 40, 25, 15, 60, 40, 25, 15],
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
        self.base_final[11] = params["final_arm_pitch"]
        self.base_final[15] = params["final_arm_pitch"]

    def desired_qpos(self) -> np.ndarray:
        data = self.sim.data
        t = data.time
        pelvis_x = self.sim.pelvis_position()[0]
        roll, pitch, _ = quat_to_euler_xyz(data.qpos[3:7])
        ang_roll = data.qvel[3]
        ang_pitch = data.qvel[4]

        if t < self.params["walk_time"]:
            qdes = self.base_walk.copy()
            phase = 2.0 * math.pi * self.params["freq"] * t + self.params["phase"]
            s = math.sin(phase)
            c = math.cos(phase)
            pos_s = max(0.0, s)
            neg_s = max(0.0, -s)

            qdes[0] += self.params["yaw_amp"] * c
            qdes[5] -= self.params["yaw_amp"] * c
            qdes[2] += self.params["hip_amp"] * s
            qdes[7] -= self.params["hip_amp"] * s
            qdes[3] += self.params["knee_amp"] * pos_s
            qdes[8] += self.params["knee_amp"] * neg_s
            qdes[4] -= self.params["ankle_amp"] * s
            qdes[9] += self.params["ankle_amp"] * s
            qdes[1] += self.params["hip_roll_amp"] * c
            qdes[6] -= self.params["hip_roll_amp"] * c
            qdes[11] = self.params["arm_pitch"] + self.params["arm_swing_amp"] * s
            qdes[15] = self.params["arm_pitch"] - self.params["arm_swing_amp"] * s

            speed_err = self.params["target_speed"] - data.qvel[0]
            qdes[2] += self.params["speed_hip_gain"] * speed_err
            qdes[7] += self.params["speed_hip_gain"] * speed_err
            qdes[10] += self.params["speed_torso_gain"] * speed_err
            qdes[10] += self.params["progress_torso_gain"] * max(0.0, TARGET_XY[0] - pelvis_x)
        else:
            blend = clamp((t - self.params["walk_time"]) / self.params["blend_time"], 0.0, 1.0)
            qdes = (1.0 - blend) * self.base_walk + blend * self.base_final

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
        y_correction = self.params["y_gain"] * y_err - self.params["y_vel_gain"] * y_vel
        qdes[1] += y_correction
        qdes[6] -= y_correction
        return qdes

    def torque(self) -> np.ndarray:
        q = self.sim.data.qpos[QPOS_IDX]
        qvel = self.sim.data.qvel[QVEL_IDX]
        qdes = self.desired_qpos()
        bias = self.sim.data.qfrc_bias[QVEL_IDX]
        torque = bias + self.kp * (qdes - q) - self.kd * qvel
        return np.clip(torque, CTRL_MIN, CTRL_MAX)


def evaluate_trace(sim: Sim) -> EvalResult:
    replay = sim._trace
    replay_steps = len(sim._ctrl_trace)
    pelvis = np.array([item["pelvis_pos"] for item in replay], dtype=float)
    torso_up = np.array([item["torso_up"] for item in replay], dtype=float)
    checkpoint_hits = sum(float(np.max(pelvis[:, 0]) >= x) for x in CHECKPOINTS)
    stable_fraction = float(np.mean((pelvis[:, 2] >= MIN_PELVIS_Z) & (torso_up >= MIN_TORSO_UP)))

    settle_min_pelvis_z = float("inf")
    settle_min_torso_up = float("inf")
    for _ in range(500):
        sim.step(1)
        settle_min_pelvis_z = min(settle_min_pelvis_z, sim.pelvis_position()[2])
        settle_min_torso_up = min(settle_min_torso_up, sim.torso_up())

    final_target_distance = float(np.linalg.norm(sim.pelvis_position()[:2] - TARGET_XY))
    checkpoint_progress = checkpoint_hits / len(CHECKPOINTS)
    stability_progress = clamp(stable_fraction / 0.95, 0.0, 1.0)
    distance_progress = clamp(1.0 - final_target_distance / 1.05, 0.0, 1.0)
    settle_progress = 0.5 * clamp(settle_min_pelvis_z / MIN_PELVIS_Z, 0.0, 1.0) + 0.5 * clamp(
        settle_min_torso_up / MIN_TORSO_UP, 0.0, 1.0
    )
    efficiency_progress = clamp(1800.0 / max(replay_steps, 1), 0.0, 1.0)
    score = (
        0.30 * checkpoint_progress
        + 0.25 * stability_progress
        + 0.25 * distance_progress
        + 0.15 * settle_progress
        + 0.05 * efficiency_progress
    )
    passed = (
        replay_steps >= 30
        and np.linalg.norm(replay[0]["pelvis_pos"][:2]) <= 0.15
        and checkpoint_hits == len(CHECKPOINTS)
        and stable_fraction >= 0.95
        and final_target_distance <= 0.18
        and settle_min_pelvis_z >= MIN_PELVIS_Z
        and settle_min_torso_up >= MIN_TORSO_UP
    )
    return EvalResult(
        passed=passed,
        score=score,
        checkpoints_hit=int(checkpoint_hits),
        stable_fraction=stable_fraction,
        final_target_distance=final_target_distance,
        settle_min_pelvis_z=settle_min_pelvis_z,
        settle_min_torso_up=settle_min_torso_up,
        replay_steps=replay_steps,
        final_pelvis_x=float(sim.pelvis_position()[0]),
        final_pelvis_y=float(sim.pelvis_position()[1]),
    )


def rollout(params: dict, save_path: str | None = None) -> tuple[EvalResult, Sim]:
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


def tweak(base: dict, **updates) -> dict:
    out = dict(base)
    out.update(updates)
    return out


def sample_params(rng: np.random.Generator, center: dict | None = None, scale: float = 1.0) -> dict:
    if center is None:
        center = {}

    def pick(name: str, lo: float, hi: float) -> float:
        if name not in center:
            return float(rng.uniform(lo, hi))
        span = (hi - lo) * 0.20 * scale
        return float(np.clip(center[name] + rng.normal(0.0, span), lo, hi))

    return {
        "stance_hip": pick("stance_hip", -0.56, -0.20),
        "stance_knee": pick("stance_knee", 0.55, 0.95),
        "stance_ankle": pick("stance_ankle", -0.50, 0.02),
        "torso_lean": pick("torso_lean", -0.05, 0.25),
        "arm_pitch": pick("arm_pitch", -0.4, 0.4),
        "freq": pick("freq", 0.8, 2.2),
        "phase": pick("phase", -math.pi, math.pi),
        "hip_amp": pick("hip_amp", 0.05, 0.55),
        "knee_amp": pick("knee_amp", 0.10, 0.95),
        "ankle_amp": pick("ankle_amp", 0.02, 0.35),
        "hip_roll_amp": pick("hip_roll_amp", 0.0, 0.22),
        "arm_swing_amp": pick("arm_swing_amp", 0.0, 0.6),
        "yaw_amp": pick("yaw_amp", 0.0, 0.15),
        "target_speed": pick("target_speed", 0.2, 0.95),
        "speed_hip_gain": pick("speed_hip_gain", 0.0, 0.5),
        "speed_torso_gain": pick("speed_torso_gain", 0.0, 0.5),
        "progress_torso_gain": pick("progress_torso_gain", 0.0, 0.4),
        "y_gain": pick("y_gain", 1.5, 8.0),
        "y_vel_gain": pick("y_vel_gain", 0.4, 3.0),
        "roll_gain": pick("roll_gain", -0.3, 0.8),
        "roll_rate_gain": pick("roll_rate_gain", -0.2, 0.4),
        "roll_ankle_gain": pick("roll_ankle_gain", 0.0, 1.2),
        "roll_ankle_rate_gain": pick("roll_ankle_rate_gain", -0.2, 0.4),
        "pitch_gain": pick("pitch_gain", -0.6, 0.8),
        "pitch_rate_gain": pick("pitch_rate_gain", -0.2, 0.6),
        "pitch_ankle_gain": pick("pitch_ankle_gain", 0.0, 1.2),
        "pitch_ankle_rate_gain": pick("pitch_ankle_rate_gain", -0.2, 0.4),
        "torso_pitch_gain": pick("torso_pitch_gain", -0.1, 0.8),
        "torso_pitch_rate_gain": pick("torso_pitch_rate_gain", -0.1, 0.6),
        "walk_time": pick("walk_time", 0.9, 2.0),
        "blend_time": pick("blend_time", 0.08, 0.5),
        "total_time": pick("total_time", 1.4, 2.6),
        "final_hip": pick("final_hip", -0.25, 0.02),
        "final_knee": pick("final_knee", 0.45, 0.90),
        "final_ankle": pick("final_ankle", -0.25, 0.12),
        "final_torso": pick("final_torso", -0.04, 0.18),
        "final_arm_pitch": pick("final_arm_pitch", -0.2, 0.2),
    }


def witness_score(result: EvalResult) -> float:
    if result.passed:
        return 1e6
    checkpoint_term = 2.0 * (result.checkpoints_hit / 3.0)
    stability_term = 2.0 * clamp(result.stable_fraction / 0.95, 0.0, 1.0)
    distance_term = 2.0 * clamp(1.0 - result.final_target_distance / 1.05, 0.0, 1.0)
    settle_term = 1.5 * (
        0.5 * clamp(result.settle_min_pelvis_z / MIN_PELVIS_Z, 0.0, 1.0)
        + 0.5 * clamp(result.settle_min_torso_up / MIN_TORSO_UP, 0.0, 1.0)
    )
    lane_bonus = 0.8 * clamp(1.0 - abs(result.final_pelvis_y) / 0.40, 0.0, 1.0)
    return checkpoint_term + stability_term + distance_term + settle_term + lane_bonus


def format_result(result: EvalResult) -> dict:
    return {
        "passed": result.passed,
        "score": round(result.score, 4),
        "checkpoints_hit": result.checkpoints_hit,
        "stable_fraction": round(result.stable_fraction, 4),
        "final_target_distance": round(result.final_target_distance, 4),
        "settle_min_pelvis_z": round(result.settle_min_pelvis_z, 4),
        "settle_min_torso_up": round(result.settle_min_torso_up, 4),
        "replay_steps": result.replay_steps,
        "final_pelvis_x": round(result.final_pelvis_x, 4),
        "final_pelvis_y": round(result.final_pelvis_y, 4),
    }


def main():
    rng = np.random.default_rng(0)
    save_path = "/work/final_state.npz"

    base = {
        "stance_hip": -0.54,
        "stance_knee": 1.0,
        "stance_ankle": -0.54,
        "torso_lean": 0.06,
        "arm_pitch": -0.18,
        "freq": 1.23,
        "phase": -0.99,
        "hip_amp": 0.32,
        "knee_amp": 0.79,
        "ankle_amp": 0.09,
        "hip_roll_amp": 0.20,
        "arm_swing_amp": 0.40,
        "yaw_amp": 0.0,
        "target_speed": 1.03,
        "speed_hip_gain": 0.06,
        "speed_torso_gain": 0.24,
        "progress_torso_gain": 0.27,
        "y_gain": 4.5,
        "y_vel_gain": 1.7,
        "roll_gain": -0.17,
        "roll_rate_gain": -0.07,
        "roll_ankle_gain": 0.77,
        "roll_ankle_rate_gain": -0.17,
        "pitch_gain": -0.60,
        "pitch_rate_gain": -0.20,
        "pitch_ankle_gain": 1.05,
        "pitch_ankle_rate_gain": 0.17,
        "torso_pitch_gain": 0.06,
        "torso_pitch_rate_gain": 0.45,
        "walk_time": 1.70,
        "blend_time": 0.12,
        "total_time": 2.00,
        "final_hip": -0.34,
        "final_knee": 0.72,
        "final_ankle": -0.10,
        "final_torso": 0.02,
        "final_arm_pitch": 0.0,
    }
    seed_bank = [
        base,
        tweak(base, y_gain=6.0, y_vel_gain=2.2, walk_time=1.55, total_time=1.85, final_torso=0.08),
        tweak(base, y_gain=7.0, y_vel_gain=2.6, hip_roll_amp=0.12, walk_time=1.45, total_time=1.70, final_knee=0.80, final_ankle=-0.16),
        tweak(base, phase=-0.75, hip_amp=0.28, knee_amp=0.70, target_speed=0.84, y_gain=5.5, y_vel_gain=1.9, walk_time=1.55, total_time=1.95),
        tweak(base, stance_hip=-0.46, stance_knee=0.86, stance_ankle=-0.42, target_speed=0.72, y_gain=5.8, y_vel_gain=2.1, walk_time=1.30, total_time=1.70, final_torso=0.10),
        tweak(base, stance_hip=-0.40, stance_knee=0.78, stance_ankle=-0.35, target_speed=0.62, y_gain=6.5, y_vel_gain=2.4, walk_time=1.20, total_time=1.55, final_hip=-0.22, final_knee=0.58, final_ankle=-0.05, final_torso=0.14),
        tweak(base, stance_hip=-0.42, stance_knee=0.72, stance_ankle=-0.28, torso_lean=0.12, target_speed=0.58, y_gain=5.4, y_vel_gain=2.0, walk_time=1.15, total_time=1.65, final_hip=-0.08, final_knee=0.46, final_ankle=0.05, final_torso=0.16),
        tweak(base, stance_hip=-0.48, stance_knee=0.80, stance_ankle=-0.34, torso_lean=0.10, target_speed=0.56, y_gain=6.8, y_vel_gain=2.6, walk_time=1.05, total_time=1.55, final_hip=-0.04, final_knee=0.40, final_ankle=0.08, final_torso=0.18),
    ]

    best_params = None
    best_result = None
    best_sim = None
    max_iters = 400
    for i in range(max_iters):
        if i < len(seed_bank):
            params = seed_bank[i]
        elif best_params is not None and rng.random() < 0.85:
            params = sample_params(rng, best_params, scale=0.6 if best_result.score > 0.65 else 1.0)
        else:
            params = sample_params(rng, base, scale=1.0)

        # Keep the total horizon long enough to include the whole walk and a short stand phase.
        params["blend_time"] = min(params["blend_time"], max(0.08, params["total_time"] - params["walk_time"] - 0.10))
        params["walk_time"] = min(params["walk_time"], params["total_time"] - 0.12)

        result, sim = rollout(params)
        cur = witness_score(result)
        best = witness_score(best_result) if best_result is not None else float("-inf")
        if best_result is None or cur > best:
            best_params = dict(params)
            best_result = result
            best_sim = sim
            sim.save_final_state(save_path)
            with open("/work/best_params.json", "w", encoding="ascii") as f:
                json.dump({"params": best_params, "result": format_result(best_result)}, f, indent=2)
            print(f"iter {i}: new best witness={cur:.4f} {json.dumps(format_result(best_result))}", flush=True)
        elif i % 20 == 0:
            print(f"iter {i}: witness={cur:.4f} {json.dumps(format_result(result))}", flush=True)
        if result.passed:
            break

    if best_result is None:
        raise RuntimeError("no rollout completed")
    if best_sim is not None:
        best_sim.save_final_state(save_path)
    print("best", json.dumps(format_result(best_result)), flush=True)


if __name__ == "__main__":
    main()
