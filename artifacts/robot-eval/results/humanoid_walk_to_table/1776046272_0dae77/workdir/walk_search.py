import math
import os
import random

import numpy as np

from sim import Sim, TARGET_XY, MIN_PELVIS_Z, MIN_TORSO_UP


ACTUATOR_LIMITS = np.array(
    [200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 200, 40, 40, 18, 18, 40, 40, 18, 18],
    dtype=float,
)


def build_joint_index(sim: Sim):
    # qpos indices for actuated joints, excluding the free joint.
    return np.array([7 + i for i in range(sim.model.nu)], dtype=int)


def rollout(sim: Sim, params: dict, steps: int = 1800):
    qadr = build_joint_index(sim)
    ctrl_trace = []
    trace = []
    sim.reset()

    kp = params["kp"]
    kd = params["kd"]
    base = sim.home_ctrl().copy()
    t0 = params.get("warmup_steps", 250)
    walk_end = params.get("walk_end", 1500)
    period = params["period"]
    amp = params["amp"]
    knee_amp = params["knee_amp"]
    ankle_amp = params["ankle_amp"]
    torso_bias = params["torso_bias"]
    torso_amp = params["torso_amp"]
    arm_amp = params["arm_amp"]
    roll_amp = params["roll_amp"]
    yaw_amp = params["yaw_amp"]
    crouch = params["crouch"]
    crouch_shift = params["crouch_shift"]
    stop_crouch = params["stop_crouch"]
    forward_bias = params["forward_bias"]
    phase_offset = params["phase_offset"]

    for t in range(steps):
        tt = t * sim.model.opt.timestep
        if t < t0:
            walk = 0.0
        elif t < walk_end:
            walk = 1.0
        else:
            # gently taper into a stop while holding a low stance
            walk = max(0.0, 1.0 - (t - walk_end) / max(1, params.get("stop_ramp", 220)))

        phase = 2.0 * math.pi * ((tt / period) + phase_offset)
        s = math.sin(phase)
        s2 = math.sin(phase + math.pi)

        des = base.copy()
        # Symmetric crouch / lean.
        des[2] = -0.4 + crouch + crouch_shift * walk
        des[7] = -0.4 + crouch + crouch_shift * walk
        des[10] = torso_bias + torso_amp * walk

        # Alternate legs.
        des[2] += walk * amp * s
        des[7] -= walk * amp * s
        des[3] = 0.8 + walk * knee_amp * max(0.0, s)
        des[8] = 0.8 + walk * knee_amp * max(0.0, -s)
        des[4] = -0.4 - walk * ankle_amp * max(0.0, s)
        des[9] = -0.4 - walk * ankle_amp * max(0.0, -s)
        des[0] = walk * yaw_amp * s
        des[5] = -walk * yaw_amp * s
        des[1] = walk * roll_amp * s
        des[6] = -walk * roll_amp * s
        des[11] = walk * arm_amp * s2
        des[15] = -walk * arm_amp * s
        des[12] = -0.5 * walk * arm_amp * s
        des[16] = 0.5 * walk * arm_amp * s

        q = sim.data.qpos[qadr]
        qd = sim.data.qvel[6:6 + sim.model.nu]
        torque = kp * (des - q) - kd * qd
        torque[10] += forward_bias * walk
        torque = np.clip(torque, -ACTUATOR_LIMITS, ACTUATOR_LIMITS)
        sim.data.ctrl[:] = torque
        ctrl_trace.append(torque.copy())
        sim.step()
        trace.append((sim.data.time, sim.pelvis_position().copy(), sim.torso_up()))

    xs = np.array([p[1][0] for p in trace], dtype=float)
    zs = np.array([p[1][2] for p in trace], dtype=float)
    ups = np.array([p[2] for p in trace], dtype=float)
    final_xy_dist = float(np.linalg.norm(sim.pelvis_position()[:2] - TARGET_XY))
    stable = float(np.mean((zs >= MIN_PELVIS_Z) & (ups >= MIN_TORSO_UP)))
    checkpoints = [0.18, 0.34, 0.45]
    hits = [float(xs.max() >= c) for c in checkpoints]
    score = (
        2.0 * xs.max()
        - 1.2 * final_xy_dist
        + 3.0 * stable
        + 0.5 * sum(hits)
        + 0.2 * float(zs.min() >= MIN_PELVIS_Z)
        + 0.2 * float(ups.min() >= MIN_TORSO_UP)
    )
    return {
        "score": score,
        "xs": xs,
        "zs": zs,
        "ups": ups,
        "final_xy_dist": final_xy_dist,
        "stable": stable,
        "hits": hits,
        "ctrl_trace": np.array(ctrl_trace, dtype=float),
        "trace": trace,
        "qpos": sim.data.qpos.copy(),
        "qvel": sim.data.qvel.copy(),
        "ctrl": sim.data.ctrl.copy(),
        "sim": sim,
    }


def candidate_from_seed(seed: int):
    rng = random.Random(seed)
    return {
        "kp": np.array([120, 120, 150, 180, 50, 120, 120, 150, 180, 50, 100, 20, 20, 10, 10, 20, 20, 10, 10], dtype=float)
        * rng.uniform(0.7, 1.3),
        "kd": np.array([12, 12, 14, 16, 5, 12, 12, 14, 16, 5, 10, 4, 4, 2, 2, 4, 4, 2, 2], dtype=float)
        * rng.uniform(0.7, 1.3),
        "period": rng.uniform(0.45, 0.8),
        "amp": rng.uniform(0.05, 0.22),
        "knee_amp": rng.uniform(0.08, 0.28),
        "ankle_amp": rng.uniform(0.02, 0.18),
        "torso_bias": rng.uniform(-0.25, 0.25),
        "torso_amp": rng.uniform(-0.10, 0.18),
        "arm_amp": rng.uniform(0.0, 0.18),
        "roll_amp": rng.uniform(0.0, 0.08),
        "yaw_amp": rng.uniform(0.0, 0.05),
        "crouch": rng.uniform(0.02, 0.18),
        "crouch_shift": rng.uniform(-0.08, 0.08),
        "stop_crouch": rng.uniform(0.02, 0.12),
        "forward_bias": rng.uniform(-25.0, 25.0),
        "phase_offset": rng.uniform(0.0, 1.0),
        "warmup_steps": rng.randint(80, 300),
        "walk_end": rng.randint(1200, 1600),
        "stop_ramp": rng.randint(120, 320),
    }


if __name__ == "__main__":
    sim = Sim()
    # A few hand-picked candidates first.
    seeds = [0, 1, 2, 3, 4, 5, 11, 22, 33, 44]
    best = None
    for s in seeds:
        params = candidate_from_seed(s)
        result = rollout(sim, params)
        print(
            "seed", s,
            "score", round(result["score"], 3),
            "max_x", round(float(result["xs"].max()), 3),
            "dist", round(result["final_xy_dist"], 3),
            "stable", round(result["stable"], 3),
            "hits", result["hits"],
            "min_z", round(float(result["zs"].min()), 3),
            "min_up", round(float(result["ups"].min()), 3),
        )
        if best is None or result["score"] > best["score"]:
            best = {**result, "params": params}
    sim = best["sim"]
    print("BEST", best["score"], best["params"])
    sim.save_final_state("/work/final_state.npz")
    print("saved", os.path.getsize("/work/final_state.npz"))
