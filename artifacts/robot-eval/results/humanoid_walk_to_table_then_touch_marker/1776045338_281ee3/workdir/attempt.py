import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import MIN_PELVIS_Z, MIN_TORSO_UP, Sim, TARGET_XY, TOUCH_POINT, TOUCH_RADIUS


DEADLINE = 1776046239.0  # 2026-04-13 02:10:39 UTC


@dataclass
class EvalResult:
    score: float
    checkpoints: int
    stable_fraction: float
    best_touch: float
    final_target: float
    settle_min_z: float
    settle_min_up: float
    replay_steps: int
    touched_in_settle: bool
    sim: Sim


def torso_pitch(sim: Sim) -> float:
    rot = sim.data.xmat[sim.torso_body_id].reshape(3, 3)
    return math.atan2(-rot[2, 0], rot[2, 2])


def rollout(params: dict, save_path: str | None = None) -> EvalResult:
    sim = Sim()
    dt = sim.model.opt.timestep
    q_home = sim.home_ctrl().copy()
    q0 = q_home.copy()
    q0[2] = -0.55
    q0[3] = 1.10
    q0[4] = -0.55
    q0[7] = -0.55
    q0[8] = 1.10
    q0[9] = -0.55
    q0[11] = -0.35
    q0[15] = -0.35

    kp = np.array([220, 180, 320, 360, 100, 220, 180, 320, 360, 100, 180, 60, 60, 40, 30, 60, 60, 40, 30], dtype=float)
    kd = np.array([20, 16, 28, 30, 10, 20, 16, 28, 30, 10, 14, 6, 6, 4, 3, 6, 6, 4, 3], dtype=float)

    steps = params["steps"]
    stride_period = params["stride_period"]
    stride_amp = params["stride_amp"]
    knee_amp = params["knee_amp"]
    roll_amp = params["roll_amp"]
    forward_lean = params["forward_lean"]
    ankle_lean = params["ankle_lean"]
    arm_start = params["arm_start"]
    reach_steps = params["reach_steps"]
    torso_yaw = params["torso_yaw"]

    for i in range(steps):
        qdes = q0.copy()
        phase = 2.0 * math.pi * i / stride_period
        s = math.sin(phase)
        c = math.cos(phase)

        lean_scale = min(1.0, i / max(1, int(0.4 / dt)))
        qdes[2] -= forward_lean * lean_scale
        qdes[4] += ankle_lean * lean_scale
        qdes[7] -= forward_lean * lean_scale
        qdes[9] += ankle_lean * lean_scale

        qdes[2] += stride_amp * s
        qdes[7] -= stride_amp * s
        qdes[3] += knee_amp * max(0.0, -s)
        qdes[8] += knee_amp * max(0.0, s)
        qdes[4] -= 0.5 * stride_amp * s
        qdes[9] += 0.5 * stride_amp * s
        qdes[1] += roll_amp * c
        qdes[6] += roll_amp * c
        qdes[10] = torso_yaw * s

        vx = float(sim.data.qvel[0])
        qdes[2] -= params["vel_gain"] * vx
        qdes[7] -= params["vel_gain"] * vx
        pitch = torso_pitch(sim)
        qdes[4] += params["pitch_gain"] * pitch
        qdes[9] += params["pitch_gain"] * pitch

        if i >= arm_start:
            a = min(1.0, (i - arm_start) / max(1, reach_steps))
            qdes[15] = -0.95 * a
            qdes[16] = -0.20 * a
            qdes[17] = -1.30 * a
            qdes[18] = 1.55 * a
            qdes[11] = 0.15 * a
            qdes[12] = 0.15 * a
            qdes[13] = 0.40 * a
            qdes[14] = 0.50 * a

        q = sim.data.qpos[7:]
        v = sim.data.qvel[6:]
        sim.data.qacc[:] = 0
        sim.data.qacc[6:] = kp * (qdes - q) - kd * v
        mujoco.mj_inverse(sim.model, sim.data)
        ctrl = np.clip(sim.data.qfrc_inverse[6:], sim.model.actuator_ctrlrange[:, 0], sim.model.actuator_ctrlrange[:, 1])
        sim.data.ctrl[:] = ctrl
        sim.step()

    settle_min_z = 10.0
    settle_min_up = 10.0
    touched_in_settle = False
    for _ in range(500):
        q = sim.data.qpos[7:]
        v = sim.data.qvel[6:]
        qdes = q0.copy()
        qdes[2] -= 0.45
        qdes[4] += 0.20
        qdes[7] -= 0.45
        qdes[9] += 0.20
        qdes[15] = -0.95
        qdes[16] = -0.20
        qdes[17] = -1.30
        qdes[18] = 1.55
        pitch = torso_pitch(sim)
        qdes[4] += params["pitch_gain"] * pitch
        qdes[9] += params["pitch_gain"] * pitch
        sim.data.qacc[:] = 0
        sim.data.qacc[6:] = kp * (qdes - q) - kd * v
        mujoco.mj_inverse(sim.model, sim.data)
        ctrl = np.clip(sim.data.qfrc_inverse[6:], sim.model.actuator_ctrlrange[:, 0], sim.model.actuator_ctrlrange[:, 1])
        sim.data.ctrl[:] = ctrl
        sim.step()
        settle_min_z = min(settle_min_z, sim.pelvis_position()[2])
        settle_min_up = min(settle_min_up, sim.torso_up())
        touched_in_settle = touched_in_settle or (sim.distance_to_touch() <= TOUCH_RADIUS)

    trace = sim._trace
    pelvis_x = np.array([t["pelvis_pos"][0] for t in trace], dtype=float)
    pelvis_z = np.array([t["pelvis_pos"][2] for t in trace], dtype=float)
    torso_up_trace = np.array([t["torso_up"] for t in trace], dtype=float)
    touch_trace = np.array([t["touch_distance"] for t in trace], dtype=float)
    stable_fraction = float(np.mean((pelvis_z >= MIN_PELVIS_Z) & (torso_up_trace >= MIN_TORSO_UP)))
    checkpoints = sum(float(np.max(pelvis_x)) >= x for x in (0.25, 0.55, 0.80))
    best_touch = float(np.min(touch_trace))
    final_target = float(np.linalg.norm(sim.pelvis_position()[:2] - TARGET_XY))
    checkpoint_progress = checkpoints / 3.0
    touch_progress = 1.0 if best_touch <= TOUCH_RADIUS else max(0.0, 1.0 - (best_touch - TOUCH_RADIUS) / 1.2)
    stability_progress = min(1.0, stable_fraction / 0.35)
    distance_progress = 1.0 if final_target <= 0.45 else max(0.0, 1.0 - (final_target - 0.45) / 1.5)
    settle_posture = 0.5 * min(1.0, settle_min_z / MIN_PELVIS_Z) + 0.5 * min(1.0, settle_min_up / MIN_TORSO_UP)
    efficiency = min(1.0, 2200.0 / max(1, len(sim._ctrl_trace)))
    score = (
        0.15 * checkpoint_progress
        + 0.30 * touch_progress
        + 0.20 * stability_progress
        + 0.15 * distance_progress
        + 0.10 * settle_posture
        + 0.10 * efficiency
    )

    if save_path is not None:
        sim.save_final_state(save_path)

    return EvalResult(
        score=score,
        checkpoints=checkpoints,
        stable_fraction=stable_fraction,
        best_touch=best_touch,
        final_target=final_target,
        settle_min_z=settle_min_z,
        settle_min_up=settle_min_up,
        replay_steps=len(sim._ctrl_trace),
        touched_in_settle=touched_in_settle,
        sim=sim,
    )


def main() -> None:
    rng = np.random.default_rng(0)
    base = {
        "steps": 1400,
        "stride_period": 220,
        "stride_amp": 0.32,
        "knee_amp": 0.60,
        "roll_amp": 0.08,
        "forward_lean": 0.32,
        "ankle_lean": 0.18,
        "vel_gain": 0.06,
        "pitch_gain": 0.18,
        "arm_start": 850,
        "reach_steps": 220,
        "torso_yaw": 0.05,
    }

    best = rollout(base, "/work/final_state.npz")
    print("baseline", best)
    best_params = dict(base)
    start = time.time()
    trial = 0
    while time.time() < DEADLINE - 45:
        trial += 1
        cand = dict(best_params)
        cand["steps"] = int(np.clip(best_params["steps"] + rng.integers(-180, 181), 900, 2200))
        for k, scale, lo, hi in [
            ("stride_period", 20, 140, 320),
            ("stride_amp", 0.06, 0.12, 0.65),
            ("knee_amp", 0.10, 0.20, 1.10),
            ("roll_amp", 0.03, 0.00, 0.20),
            ("forward_lean", 0.08, 0.00, 0.70),
            ("ankle_lean", 0.05, -0.10, 0.45),
            ("vel_gain", 0.03, -0.10, 0.20),
            ("pitch_gain", 0.05, -0.20, 0.40),
            ("torso_yaw", 0.04, -0.20, 0.20),
        ]:
            cand[k] = float(np.clip(best_params[k] + rng.normal(0.0, scale), lo, hi))
        cand["arm_start"] = int(np.clip(best_params["arm_start"] + rng.integers(-180, 181), 400, cand["steps"] - 50))
        cand["reach_steps"] = int(np.clip(best_params["reach_steps"] + rng.integers(-80, 81), 80, 400))

        res = rollout(cand)
        if (
            res.score > best.score
            or (res.checkpoints, -res.best_touch, -res.final_target, res.stable_fraction)
            > (best.checkpoints, -best.best_touch, -best.final_target, best.stable_fraction)
        ):
            best = res
            best_params = cand
            best.sim.save_final_state("/work/final_state.npz")
            print("improved", trial, best, best_params, flush=True)
        elif trial % 10 == 0:
            print(
                "trial",
                trial,
                "score",
                f"{res.score:.3f}",
                "best",
                f"{best.score:.3f}",
                "cp",
                res.checkpoints,
                "touch",
                f"{res.best_touch:.3f}",
                "target",
                f"{res.final_target:.3f}",
                "elapsed",
                f"{time.time() - start:.1f}s",
                flush=True,
            )

    print("best_final", best, best_params)
    best.sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
