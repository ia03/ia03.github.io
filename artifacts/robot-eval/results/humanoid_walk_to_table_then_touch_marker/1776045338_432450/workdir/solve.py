import math
import time
from dataclasses import dataclass

import numpy as np

from sim import MIN_PELVIS_Z, MIN_TORSO_UP, Sim, TARGET_RADIUS, TARGET_XY, TOUCH_POINT, TOUCH_RADIUS


@dataclass
class Params:
    hip_pitch_bias: float = 0.0
    knee_bias: float = 0.0
    ankle_bias: float = 0.0
    torso_bias: float = 0.0
    lean_target: float = 0.10
    lean_kp: float = 80.0
    lean_kd: float = 12.0
    torso_kp: float = 60.0
    torso_kd: float = 8.0
    roll_kp: float = 60.0
    roll_kd: float = 8.0
    gait_hz: float = 1.2
    swing_hip: float = 0.28
    swing_knee: float = 0.45
    swing_ankle: float = 0.10
    stance_hip: float = -0.08
    arm_pitch: float = -0.7
    arm_roll: float = 0.18
    arm_yaw: float = 0.0
    elbow: float = 0.45
    reach_start: float = 0.65
    reach_gain: float = 1.1

    def mutate(self, scale=0.25):
        out = Params(**self.__dict__)
        for k, v in self.__dict__.items():
            sigma = scale * (abs(v) + 0.15)
            setattr(out, k, float(v + np.random.randn() * sigma))
        out.gait_hz = float(np.clip(out.gait_hz, 0.4, 2.5))
        out.reach_start = float(np.clip(out.reach_start, 0.2, 1.0))
        return out


def stable_fraction(trace_pelvis, trace_up):
    ok = (trace_pelvis[:, 2] >= MIN_PELVIS_Z) & (trace_up >= MIN_TORSO_UP)
    return float(ok.mean()) if len(ok) else 0.0


def score_from_trace(trace_pelvis, trace_up, trace_touch, replay_steps):
    checkpoints = [0.25, 0.55, 0.80]
    checkpoint_progress = sum(np.max(trace_pelvis[:, 0]) >= x for x in checkpoints) / len(checkpoints)
    best_touch = float(np.min(trace_touch))
    touch_progress = 1.0 if best_touch <= TOUCH_RADIUS else max(0.0, 1.0 - (best_touch - TOUCH_RADIUS) / 1.5)
    stab = stable_fraction(trace_pelvis, trace_up)
    return checkpoint_progress, touch_progress, stab


def final_settle_progress(sim):
    pelvis = []
    up = []
    touch = []
    for _ in range(500):
        sim.step()
        pelvis.append(sim.pelvis_position())
        up.append(sim.torso_up())
        touch.append(sim.distance_to_touch())
    pelvis = np.array(pelvis)
    up = np.array(up)
    touch = np.array(touch)
    dist = float(np.linalg.norm(pelvis[-1, :2] - TARGET_XY))
    dist_progress = 1.0 if dist <= TARGET_RADIUS else max(0.0, 1.0 - (dist - TARGET_RADIUS) / 1.5)
    settle_posture = 0.5 * np.clip(np.min(pelvis[:, 2]) / MIN_PELVIS_Z, 0, 1) + 0.5 * np.clip(np.min(up) / MIN_TORSO_UP, 0, 1)
    touch_hold = float(np.min(touch))
    return dist, dist_progress, settle_posture, touch_hold


def controller(sim, params, t):
    q = sim.data.qpos[7:].copy()
    v = sim.data.qvel[6:].copy()
    R = sim.data.xmat[sim.torso_body_id].reshape(3, 3)
    torso_pitch = math.atan2(R[0, 2], R[2, 2])
    torso_roll = -math.atan2(R[1, 2], R[2, 2])
    dq = np.zeros_like(q)
    dq[2] = params.hip_pitch_bias
    dq[3] = params.knee_bias
    dq[4] = params.ankle_bias
    dq[7] = params.hip_pitch_bias
    dq[8] = params.knee_bias
    dq[9] = params.ankle_bias

    phase = 2 * math.pi * params.gait_hz * t
    left = math.sin(phase)
    right = -left
    for s, hip_i, knee_i, ankle_i in [(left, 2, 3, 4), (right, 7, 8, 9)]:
        if s > 0:
            dq[hip_i] += params.swing_hip * s
            dq[knee_i] += params.swing_knee * s
            dq[ankle_i] -= params.swing_ankle * s
        else:
            dq[hip_i] += params.stance_hip * s

    x = sim.pelvis_position()[0]
    reach = np.clip((x - params.reach_start) * params.reach_gain, 0.0, 1.0)
    dq[11] = params.arm_pitch * (1.0 - 0.4 * reach)
    dq[12] = params.arm_roll
    dq[13] = params.arm_yaw
    dq[14] = params.elbow + 0.6 * reach
    dq[15] = params.arm_pitch
    dq[16] = -params.arm_roll
    dq[17] = -params.arm_yaw
    dq[18] = params.elbow + 0.6 * reach

    kp = np.array([70, 90, 160, 140, 50, 70, 90, 160, 140, 50, 60, 25, 18, 10, 10, 25, 18, 10, 10], float)
    kd = np.array([6, 7, 10, 8, 3, 6, 7, 10, 8, 3, 5, 2, 1.5, 0.7, 0.7, 2, 1.5, 0.7, 0.7], float)
    tau = kp * (dq - q) - kd * v

    tau[2] += params.lean_kp * (params.lean_target - torso_pitch) - params.lean_kd * sim.data.qvel[4]
    tau[7] += params.lean_kp * (params.lean_target - torso_pitch) - params.lean_kd * sim.data.qvel[4]
    tau[4] += 0.35 * params.lean_kp * (params.lean_target - torso_pitch)
    tau[9] += 0.35 * params.lean_kp * (params.lean_target - torso_pitch)
    tau[10] += params.torso_kp * (params.torso_bias - torso_pitch) - params.torso_kd * sim.data.qvel[16]
    tau[1] += params.roll_kp * (0.0 - torso_roll) - params.roll_kd * sim.data.qvel[7]
    tau[6] += params.roll_kp * (0.0 - torso_roll) - params.roll_kd * sim.data.qvel[7]
    tau[12] += 4.0 * torso_roll
    tau[16] -= 4.0 * torso_roll
    return np.clip(tau, sim.model.actuator_ctrlrange[:, 0], sim.model.actuator_ctrlrange[:, 1])


def run_episode(params, steps=1800, save_path=None):
    sim = Sim()
    for i in range(steps):
        sim.data.ctrl[:] = controller(sim, params, sim.data.time)
        sim.step()
        if sim.pelvis_position()[2] < 0.08:
            break
    trace_pelvis = np.array([x["pelvis_pos"] for x in sim._trace])
    trace_up = np.array([x["torso_up"] for x in sim._trace])
    trace_touch = np.array([x["touch_distance"] for x in sim._trace])
    checkpoint_progress, touch_progress, stab = score_from_trace(trace_pelvis, trace_up, trace_touch, len(sim._ctrl_trace))
    dist, dist_progress, settle_posture, touch_hold = final_settle_progress(sim)
    efficiency = np.clip(2200 / max(len(sim._ctrl_trace), 1), 0, 1)
    progress = (
        0.15 * checkpoint_progress
        + 0.30 * touch_progress
        + 0.20 * np.clip(stab / 0.35, 0, 1)
        + 0.15 * dist_progress
        + 0.10 * settle_posture
        + 0.10 * efficiency
    )
    if save_path is not None:
        sim.save_final_state(save_path)
    return {
        "sim": sim,
        "progress": float(progress),
        "checkpoint_progress": checkpoint_progress,
        "touch_progress": touch_progress,
        "stability": stab,
        "distance": dist,
        "settle_posture": settle_posture,
        "best_touch": float(min(np.min(trace_touch), touch_hold)),
        "max_x": float(np.max(trace_pelvis[:, 0])),
        "steps": len(sim._ctrl_trace),
    }


def main():
    np.random.seed(0)
    best_params = Params()
    best = run_episode(best_params)
    run_episode(best_params, save_path="/work/final_state.npz")
    print("initial", best)
    start = time.time()
    deadline = start + 7.5 * 60
    it = 0
    while time.time() < deadline:
        it += 1
        scale = 0.45 if it < 20 else 0.22
        cand = best_params.mutate(scale=scale)
        res = run_episode(cand)
        if res["progress"] > best["progress"]:
            best, best_params = res, cand
            run_episode(best_params, save_path="/work/final_state.npz")
            print("improved", it, best)
        if it % 25 == 0:
            print("iter", it, "best_progress", best["progress"], "max_x", best["max_x"], "touch", best["best_touch"], "steps", best["steps"])
    print("best", best)
    print("params", best_params)


if __name__ == "__main__":
    main()
