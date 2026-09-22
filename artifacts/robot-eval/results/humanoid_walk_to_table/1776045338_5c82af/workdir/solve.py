import math
import random
from dataclasses import dataclass

import numpy as np

from sim import MIN_PELVIS_Z, MIN_TORSO_UP, Sim


@dataclass
class Params:
    freq: float = 1.6
    walk_t: float = 2.5
    hip_base: float = -0.45
    knee_base: float = 0.9
    ankle_base: float = -0.45
    arm_base: float = -0.3
    arm_amp: float = 0.8
    hip_amp: float = 0.55
    knee_amp: float = 0.6
    ankle_amp: float = 0.22
    roll_amp: float = 0.08
    pitch_walk: float = 0.08
    pitch_stop: float = 0.18
    vx_walk: float = 0.2
    stop_hip: float = -0.9
    stop_knee: float = 1.45
    stop_ankle: float = -0.55
    stop_arm: float = -1.0
    bal_p: float = 0.8
    bal_d: float = 0.01
    bal_v: float = 0.5
    ank_p: float = -0.6
    ank_d: float = 0.005
    knee_h: float = 2.0
    z_target: float = 0.72
    kp_yaw: float = 40.0
    kp_roll: float = 60.0
    kp_hip: float = 140.0
    kp_knee: float = 180.0
    kp_ank: float = 80.0
    kp_torso: float = 40.0
    kp_arm: float = 40.0
    kp_armr: float = 20.0
    kp_ay: float = 10.0
    kp_el: float = 10.0
    kd_yaw: float = 4.0
    kd_roll: float = 6.0
    kd_hip: float = 10.0
    kd_knee: float = 12.0
    kd_ank: float = 6.0
    kd_torso: float = 4.0
    kd_arm: float = 3.0
    kd_armr: float = 2.0
    kd_ay: float = 1.0
    kd_el: float = 1.0


def gains(p: Params):
    kp = np.array(
        [
            p.kp_yaw,
            p.kp_roll,
            p.kp_hip,
            p.kp_knee,
            p.kp_ank,
            p.kp_yaw,
            p.kp_roll,
            p.kp_hip,
            p.kp_knee,
            p.kp_ank,
            p.kp_torso,
            p.kp_arm,
            p.kp_armr,
            p.kp_ay,
            p.kp_el,
            p.kp_arm,
            p.kp_armr,
            p.kp_ay,
            p.kp_el,
        ]
    )
    kd = np.array(
        [
            p.kd_yaw,
            p.kd_roll,
            p.kd_hip,
            p.kd_knee,
            p.kd_ank,
            p.kd_yaw,
            p.kd_roll,
            p.kd_hip,
            p.kd_knee,
            p.kd_ank,
            p.kd_torso,
            p.kd_arm,
            p.kd_armr,
            p.kd_ay,
            p.kd_el,
            p.kd_arm,
            p.kd_armr,
            p.kd_ay,
            p.kd_el,
        ]
    )
    return kp, kd


def torso_pitch(sim: Sim):
    mat = sim.data.xmat[sim.torso_body_id].reshape(3, 3)
    return -float(mat[2, 0])


def step_controller(sim: Sim, p: Params, last_pitch: float):
    q = sim.data.qpos[7 : 7 + sim.model.nu]
    v = sim.data.qvel[6 : 6 + sim.model.nu]
    pitch = torso_pitch(sim)
    pitch_rate = (pitch - last_pitch) / sim.model.opt.timestep
    t = sim.data.time
    z = sim.pelvis_position()[2]
    vx = float(sim.data.qvel[0])
    s = math.sin(2 * math.pi * p.freq * t)
    c = math.cos(2 * math.pi * p.freq * t)
    qdes = np.zeros(sim.model.nu)

    qdes[2] = qdes[7] = p.hip_base
    qdes[3] = qdes[8] = p.knee_base
    qdes[4] = qdes[9] = p.ankle_base
    qdes[11] = p.arm_base + p.arm_amp * s
    qdes[15] = p.arm_base - p.arm_amp * s

    if t < p.walk_t:
        qdes[2] += p.hip_amp * s
        qdes[7] -= p.hip_amp * s
        qdes[3] += p.knee_amp * max(0.0, s)
        qdes[8] += p.knee_amp * max(0.0, -s)
        qdes[4] += p.ankle_amp * s
        qdes[9] -= p.ankle_amp * s
        qdes[1] = p.roll_amp * c
        qdes[6] = -p.roll_amp * c
        target_pitch = p.pitch_walk
        target_vx = p.vx_walk
    else:
        qdes[2] = qdes[7] = p.stop_hip
        qdes[3] = qdes[8] = p.stop_knee
        qdes[4] = qdes[9] = p.stop_ankle
        qdes[11] = qdes[15] = p.stop_arm
        target_pitch = p.pitch_stop
        target_vx = 0.0

    hip_corr = p.bal_p * (target_pitch - pitch) - p.bal_d * pitch_rate + p.bal_v * (target_vx - vx)
    ankle_corr = p.ank_p * (target_pitch - pitch) - p.ank_d * pitch_rate
    knee_corr = p.knee_h * (p.z_target - z)
    qdes[2] += hip_corr
    qdes[7] += hip_corr
    qdes[3] += knee_corr
    qdes[8] += knee_corr
    qdes[4] += ankle_corr
    qdes[9] += ankle_corr

    kp, kd = gains(p)
    sim.data.ctrl[:] = kp * (qdes - q) - kd * v
    sim.step()
    return pitch


def run_episode(p: Params, save_path: str | None = None):
    sim = Sim()
    last_pitch = torso_pitch(sim)
    for _ in range(2600):
        last_pitch = step_controller(sim, p, last_pitch)
        if sim.pelvis_position()[2] < 0.12:
            break
    for _ in range(500):
        last_pitch = step_controller(sim, p, last_pitch)
        if sim.pelvis_position()[2] < 0.12:
            break
    if save_path:
        sim.save_final_state(save_path)
    return summarize(sim)


def summarize(sim: Sim):
    xs = np.array([tr["pelvis_pos"][0] for tr in sim._trace])
    ys = np.array([tr["pelvis_pos"][1] for tr in sim._trace])
    zs = np.array([tr["pelvis_pos"][2] for tr in sim._trace])
    ups = np.array([tr["torso_up"] for tr in sim._trace])
    stable = float(((zs >= MIN_PELVIS_Z) & (ups >= MIN_TORSO_UP)).mean())
    max_x = float(xs.max())
    checkpoints = int(sum(max_x >= c for c in (0.18, 0.34, 0.45)))
    end = sim.pelvis_position().copy()
    dist = float(sim.distance_to_target())
    result = {
        "max_x": max_x,
        "checkpoints": checkpoints,
        "stable": stable,
        "end_x": float(end[0]),
        "end_y": float(end[1]),
        "end_z": float(end[2]),
        "dist": dist,
        "torso_up": float(sim.torso_up()),
        "score": 1.2 * checkpoints + 1.6 * stable - 2.3 * dist - 1.5 * abs(end[1]) + 0.4 * end[2] + 0.3 * sim.torso_up(),
    }
    return result


def mutate(p: Params, scale: float):
    out = Params(**vars(p))
    for name, span in {
        "freq": 0.25,
        "walk_t": 0.45,
        "hip_base": 0.18,
        "knee_base": 0.25,
        "ankle_base": 0.18,
        "arm_base": 0.5,
        "arm_amp": 0.5,
        "hip_amp": 0.25,
        "knee_amp": 0.3,
        "ankle_amp": 0.15,
        "roll_amp": 0.08,
        "pitch_walk": 0.12,
        "pitch_stop": 0.18,
        "vx_walk": 0.18,
        "stop_hip": 0.3,
        "stop_knee": 0.35,
        "stop_ankle": 0.2,
        "stop_arm": 0.7,
        "bal_p": 0.5,
        "bal_d": 0.02,
        "bal_v": 0.4,
        "ank_p": 0.4,
        "ank_d": 0.01,
        "knee_h": 1.5,
        "z_target": 0.12,
        "kp_roll": 30.0,
        "kp_hip": 60.0,
        "kp_knee": 70.0,
        "kp_ank": 40.0,
        "kd_roll": 4.0,
        "kd_hip": 4.0,
        "kd_knee": 4.0,
        "kd_ank": 3.0,
    }.items():
        setattr(out, name, getattr(out, name) + random.uniform(-span, span) * scale)
    return out


if __name__ == "__main__":
    result = run_episode(Params(), "/work/final_state.npz")
    print(result)
