import math
import os
import time
from dataclasses import dataclass

import numpy as np

from sim import MIN_PELVIS_Z, MIN_TORSO_UP, Sim, TARGET_XY, TOUCH_POINT, TOUCH_RADIUS


CHECKPOINTS = (0.25, 0.55, 0.80)
FINAL_STATE_PATH = "/work/final_state.npz"


@dataclass
class Result:
    score: float
    success: bool
    replay_steps: int
    checkpoint_hits: int
    stable_fraction: float
    best_touch: float
    final_target_dist: float
    min_settle_pelvis_z: float
    min_settle_torso_up: float
    params: np.ndarray


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def evaluate_trace(sim: Sim):
    trace = sim._trace
    pelvis_x = np.array([entry["pelvis_pos"][0] for entry in trace], dtype=float)
    pelvis_z = np.array([entry["pelvis_pos"][2] for entry in trace], dtype=float)
    torso_up = np.array([entry["torso_up"] for entry in trace], dtype=float)
    touch = np.array([entry["touch_distance"] for entry in trace], dtype=float)

    replay_len = len(sim._ctrl_trace)
    checkpoint_hits = sum(np.max(pelvis_x) >= cp for cp in CHECKPOINTS)
    stable_fraction = float(np.mean((pelvis_z >= MIN_PELVIS_Z) & (torso_up >= MIN_TORSO_UP)))
    best_touch = float(np.min(touch))

    settle_z = []
    settle_up = []
    settle_touch = []
    for _ in range(500):
        sim.step(1)
        settle_z.append(sim.pelvis_position()[2])
        settle_up.append(sim.torso_up())
        settle_touch.append(sim.distance_to_touch())

    settle_z = np.array(settle_z, dtype=float)
    settle_up = np.array(settle_up, dtype=float)
    settle_touch = np.array(settle_touch, dtype=float)

    final_target_dist = float(np.linalg.norm(sim.pelvis_position()[:2] - TARGET_XY))
    min_settle_pelvis_z = float(np.min(settle_z))
    min_settle_torso_up = float(np.min(settle_up))
    best_settle_touch = float(np.min(settle_touch))
    best_touch_any = min(best_touch, best_settle_touch)

    success = (
        replay_len >= 30
        and np.linalg.norm(trace[0]["pelvis_pos"][:2]) <= 0.15
        and checkpoint_hits == len(CHECKPOINTS)
        and stable_fraction >= 0.35
        and best_touch_any <= TOUCH_RADIUS
        and final_target_dist <= 0.45
        and min_settle_pelvis_z >= MIN_PELVIS_Z
        and min_settle_torso_up >= MIN_TORSO_UP
    )

    checkpoint_progress = checkpoint_hits / len(CHECKPOINTS)
    touch_progress = 1.0 if best_touch_any <= TOUCH_RADIUS else clamp(1.0 - (best_touch_any - TOUCH_RADIUS) / 1.0)
    stability_progress = clamp(stable_fraction / 0.35)
    distance_progress = 1.0 if final_target_dist <= 0.45 else clamp(1.0 - (final_target_dist - 0.45) / 1.0)
    settle_posture_progress = 0.5 * clamp(min_settle_pelvis_z / MIN_PELVIS_Z) + 0.5 * clamp(min_settle_torso_up / MIN_TORSO_UP)
    efficiency_progress = clamp(2200 / max(replay_len, 1))
    score = (
        0.15 * checkpoint_progress
        + 0.30 * touch_progress
        + 0.20 * stability_progress
        + 0.15 * distance_progress
        + 0.10 * settle_posture_progress
        + 0.10 * efficiency_progress
    )

    return score, success, replay_len, checkpoint_hits, stable_fraction, best_touch_any, final_target_dist, min_settle_pelvis_z, min_settle_torso_up


def run_policy(params, save_path=None):
    sim = Sim()
    home = sim.home_ctrl().copy()
    dt = sim.model.opt.timestep

    hip_bias = params[0]
    knee_bias = params[1]
    ankle_bias = params[2]
    torso_bias = params[3]
    hip_amp = params[4]
    knee_amp = params[5]
    ankle_amp = params[6]
    roll_amp = params[7]
    freq = params[8]
    torso_swing = params[9]
    stance_extra = params[10]
    arm_start = params[11]
    arm_speed = params[12]
    shoulder_pitch = params[13]
    shoulder_roll = params[14]
    shoulder_yaw = params[15]
    elbow = params[16]
    left_arm_blend = params[17]
    right_arm_blend = params[18]
    kp_scale = params[19]
    kd_scale = params[20]
    forward_k = params[21]

    kp = np.array([220, 160, 280, 320, 90, 220, 160, 280, 320, 90, 240, 70, 60, 30, 25, 70, 60, 30, 25], dtype=float) * kp_scale
    kd = np.array([14, 10, 18, 20, 8, 14, 10, 18, 20, 8, 12, 4, 4, 2, 2, 4, 4, 2, 2], dtype=float) * kd_scale

    replay_steps = 1800
    for step in range(replay_steps):
        t = step * dt
        q = sim.data.qpos[7:26]
        v = sim.data.qvel[6:25]
        pelvis_x = sim.pelvis_position()[0]
        target = home.copy()

        target[2] = hip_bias
        target[3] = knee_bias
        target[4] = ankle_bias
        target[7] = hip_bias
        target[8] = knee_bias
        target[9] = ankle_bias
        target[10] = torso_bias + forward_k * max(0.0, 0.9 - pelvis_x)

        phase = 2.0 * math.pi * freq * t
        s = math.sin(phase)
        c = math.cos(phase)
        ls = s
        rs = -s
        lc = c
        rc = -c

        target[1] += roll_amp * lc
        target[6] += roll_amp * rc
        target[2] += hip_amp * ls
        target[7] += hip_amp * rs
        target[3] += knee_amp * max(0.0, -ls) + stance_extra * max(0.0, ls)
        target[8] += knee_amp * max(0.0, -rs) + stance_extra * max(0.0, rs)
        target[4] += ankle_amp * ls
        target[9] += ankle_amp * rs
        target[10] += torso_swing * s

        arm_alpha = clamp((t - arm_start) * arm_speed)
        target[11] = left_arm_blend * shoulder_pitch * arm_alpha
        target[12] = left_arm_blend * (-shoulder_roll) * arm_alpha
        target[13] = left_arm_blend * shoulder_yaw * arm_alpha
        target[14] = left_arm_blend * elbow * arm_alpha
        target[15] = right_arm_blend * shoulder_pitch * arm_alpha
        target[16] = right_arm_blend * shoulder_roll * arm_alpha
        target[17] = right_arm_blend * (-shoulder_yaw) * arm_alpha
        target[18] = right_arm_blend * elbow * arm_alpha

        tau = sim.data.qfrc_bias[6:25].copy() + kp * (target - q) - kd * v
        sim.data.ctrl[:] = np.clip(tau, sim.model.actuator_ctrlrange[:, 0], sim.model.actuator_ctrlrange[:, 1])
        sim.step(1)

    if save_path:
        sim.save_final_state(save_path)

    metrics = evaluate_trace(sim)
    return Result(
        score=metrics[0],
        success=metrics[1],
        replay_steps=metrics[2],
        checkpoint_hits=metrics[3],
        stable_fraction=metrics[4],
        best_touch=metrics[5],
        final_target_dist=metrics[6],
        min_settle_pelvis_z=metrics[7],
        min_settle_torso_up=metrics[8],
        params=np.array(params, dtype=float),
    )


def sample_params(rng, center=None, scale=1.0):
    if center is None:
        base = np.array([
            -0.35, 0.75, -0.35, 0.15,
            0.30, 0.55, 0.10, 0.08, 1.1, 0.04, 0.10,
            1.0, 1.0, 1.2, -0.35, 0.10, -1.0, 0.0, 1.0,
            1.0, 1.0, 0.08,
        ], dtype=float)
    else:
        base = np.array(center, dtype=float).copy()

    noise = np.array([
        0.18, 0.25, 0.15, 0.18,
        0.30, 0.45, 0.18, 0.12, 0.45, 0.10, 0.20,
        0.8, 1.0, 0.9, 0.5, 0.5, 0.9, 0.6, 0.6,
        0.35, 0.35, 0.18,
    ], dtype=float) * scale
    params = base + rng.normal(0.0, noise)

    params[0] = np.clip(params[0], -1.1, 0.3)
    params[1] = np.clip(params[1], -0.2, 1.8)
    params[2] = np.clip(params[2], -1.0, 0.8)
    params[3] = np.clip(params[3], -0.6, 0.8)
    params[4] = np.clip(params[4], 0.0, 0.9)
    params[5] = np.clip(params[5], 0.0, 1.4)
    params[6] = np.clip(params[6], -0.5, 0.5)
    params[7] = np.clip(params[7], 0.0, 0.25)
    params[8] = np.clip(params[8], 0.4, 2.8)
    params[9] = np.clip(params[9], -0.3, 0.3)
    params[10] = np.clip(params[10], -0.2, 0.8)
    params[11] = np.clip(params[11], 0.0, 2.4)
    params[12] = np.clip(params[12], 0.2, 3.5)
    params[13] = np.clip(params[13], -1.2, 2.2)
    params[14] = np.clip(params[14], -1.2, 1.2)
    params[15] = np.clip(params[15], -1.2, 1.2)
    params[16] = np.clip(params[16], -1.8, 1.8)
    params[17] = np.clip(params[17], 0.0, 1.0)
    params[18] = np.clip(params[18], 0.0, 1.0)
    params[19] = np.clip(params[19], 0.4, 2.2)
    params[20] = np.clip(params[20], 0.4, 2.2)
    params[21] = np.clip(params[21], -0.2, 0.35)
    return params


def main():
    deadline = time.time() + 11.5 * 60.0
    rng = np.random.default_rng(0)
    best = None
    first_saved = False
    center = None
    scale = 1.0
    iteration = 0

    while time.time() < deadline:
        params = sample_params(rng, center=center, scale=scale)
        save_path = FINAL_STATE_PATH if not first_saved else None
        result = run_policy(params, save_path=save_path)
        first_saved = True

        if best is None or result.score > best.score:
            best = result
            center = result.params
            scale = max(0.18, scale * 0.92)
            # Save the best-so-far trajectory/state after replay but before settle.
            run_policy(result.params, save_path=FINAL_STATE_PATH)
            print(
                f"iter={iteration} score={result.score:.3f} success={result.success} "
                f"checkpoints={result.checkpoint_hits}/3 stable={result.stable_fraction:.3f} "
                f"touch={result.best_touch:.3f} target={result.final_target_dist:.3f} "
                f"settle_z={result.min_settle_pelvis_z:.3f} settle_up={result.min_settle_torso_up:.3f}",
                flush=True,
            )
        elif iteration % 20 == 0:
            print(
                f"iter={iteration} score={result.score:.3f} best={best.score:.3f} "
                f"touch={result.best_touch:.3f} target={result.final_target_dist:.3f}",
                flush=True,
            )

        if result.success:
            print("success found", flush=True)
            break
        iteration += 1

    if best is not None:
        print("best_params", np.array2string(best.params, precision=3), flush=True)
        print(
            f"best score={best.score:.3f} success={best.success} checkpoints={best.checkpoint_hits}/3 "
            f"stable={best.stable_fraction:.3f} touch={best.best_touch:.3f} "
            f"target={best.final_target_dist:.3f} settle_z={best.min_settle_pelvis_z:.3f} "
            f"settle_up={best.min_settle_torso_up:.3f}",
            flush=True,
        )
    else:
        print("no result", flush=True)

    if not os.path.exists(FINAL_STATE_PATH):
        run_policy(sample_params(rng), save_path=FINAL_STATE_PATH)


if __name__ == "__main__":
    main()
