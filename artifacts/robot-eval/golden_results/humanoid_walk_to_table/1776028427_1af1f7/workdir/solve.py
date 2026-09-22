import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import MIN_PELVIS_Z, MIN_TORSO_UP, Sim, TARGET_XY


CHECKPOINTS = (0.25, 0.55, 0.85)
SEARCH_DEADLINE = time.time() + 18 * 60


@dataclass
class EvalResult:
    score: float
    passed: bool
    replay_steps: int
    checkpoint_hits: int
    stable_fraction: float
    final_target_distance: float
    min_replay_z: float
    min_replay_up: float
    min_settle_z: float
    min_settle_up: float
    final_x: float


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def smoothstep(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def make_sim():
    sim = Sim()
    model = sim.model
    qadr = model.jnt_qposadr[1:]
    vadr = model.jnt_dofadr[1:]
    return sim, qadr, vadr


def gains():
    kp = np.full(19, 450.0)
    kd = np.full(19, 45.0)
    kp[10] = 260.0
    kd[10] = 26.0
    kp[11:] = 90.0
    kd[11:] = 9.0
    return kp, kd


def walk_base_pose():
    sim = Sim()
    q = sim.data.qpos[sim.model.jnt_qposadr[1:]].copy()
    q[2] = q[7] = -0.2745569792545395
    q[3] = q[8] = 0.8441808854647566
    q[4] = q[9] = -0.5559077159720693
    q[10] = 0.14467769884813364
    return q


def stand_pose():
    sim = Sim()
    q = sim.home_ctrl().copy()
    q[2] = q[7] = -0.5771889972023866
    q[3] = q[8] = 0.7631329634678474
    q[4] = q[9] = -0.5328746225273263
    q[10] = 0.38650090449848556
    q[11] = q[15] = 0.4964583740047248
    return q


WALK_BASE = walk_base_pose()
STAND_BASE = stand_pose()
KP, KD = gains()


def stand_hold_ctrl():
    sim = Sim()
    model = sim.model
    data = sim.data
    qadr = model.jnt_qposadr[1:]
    dadr = model.jnt_dofadr[1:]
    data.qpos[qadr] = STAND_BASE
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    mujoco.mj_inverse(model, data)
    return data.qfrc_inverse[dadr].copy()


STAND_HOLD_CTRL = stand_hold_ctrl()


def evaluate_rollout(sim):
    replay = sim._trace
    xs = np.array([s["pelvis_pos"][0] for s in replay], dtype=float)
    zs = np.array([s["pelvis_pos"][2] for s in replay], dtype=float)
    ups = np.array([s["torso_up"] for s in replay], dtype=float)
    stable = (zs >= MIN_PELVIS_Z) & (ups >= MIN_TORSO_UP)
    checkpoint_hits = sum(np.max(xs) >= cp for cp in CHECKPOINTS)
    stable_fraction = float(np.mean(stable))

    ctrl = sim.data.ctrl.copy()
    settle_min_z = float("inf")
    settle_min_up = float("inf")
    for _ in range(500):
        sim.data.ctrl[:] = ctrl
        sim.step(1)
        settle_min_z = min(settle_min_z, sim.pelvis_position()[2])
        settle_min_up = min(settle_min_up, sim.torso_up())
    final_dist = float(np.linalg.norm(sim.pelvis_position()[:2] - TARGET_XY))

    passed = (
        len(sim._ctrl_trace) >= 30
        and np.linalg.norm(replay[0]["pelvis_pos"][:2]) <= 0.15
        and checkpoint_hits == 3
        and stable_fraction >= 0.95
        and final_dist <= 0.18
        and settle_min_z >= MIN_PELVIS_Z
        and settle_min_up >= MIN_TORSO_UP
    )
    checkpoint_progress = checkpoint_hits / 3.0
    stability_progress = clamp(stable_fraction / 0.95, 0.0, 1.0)
    distance_progress = clamp(1.0 - final_dist / 1.05, 0.0, 1.0)
    settle_progress = 0.5 * clamp(settle_min_z / MIN_PELVIS_Z, 0.0, 1.0) + 0.5 * clamp(
        settle_min_up / MIN_TORSO_UP, 0.0, 1.0
    )
    s1 = checkpoint_progress
    s2 = checkpoint_progress * stability_progress
    s3 = checkpoint_progress * stability_progress * distance_progress
    s4 = checkpoint_progress * stability_progress * distance_progress * settle_progress
    return EvalResult(
        score=(s1 + s2 + s3 + s4) / 4.0,
        passed=passed,
        replay_steps=len(sim._ctrl_trace),
        checkpoint_hits=checkpoint_hits,
        stable_fraction=stable_fraction,
        final_target_distance=final_dist,
        min_replay_z=float(np.min(zs)) if len(zs) else float("inf"),
        min_replay_up=float(np.min(ups)) if len(ups) else float("inf"),
        min_settle_z=settle_min_z,
        min_settle_up=settle_min_up,
        final_x=float(sim.pelvis_position()[0]),
    )


def gait_target(t, p):
    q = WALK_BASE.copy()
    phase = 2.0 * math.pi * p["freq"] * t + p["phase"]
    s = math.sin(phase)
    c = math.cos(phase)
    swing_l = max(0.0, s)
    swing_r = max(0.0, -s)
    q[1] = p["roll_bias"] + p["roll_amp"] * c
    q[6] = -p["roll_bias"] - p["roll_amp"] * c
    q[2] += p["hip_bias"] + p["hip_amp"] * s
    q[7] += p["hip_bias"] - p["hip_amp"] * s
    q[3] += p["knee_stance"] * (1.0 - swing_l) + p["knee_swing"] * swing_l
    q[8] += p["knee_stance"] * (1.0 - swing_r) + p["knee_swing"] * swing_r
    q[4] += p["ankle_stance"] * (1.0 - swing_l) + p["ankle_swing"] * swing_l
    q[9] += p["ankle_stance"] * (1.0 - swing_r) + p["ankle_swing"] * swing_r
    q[10] = WALK_BASE[10] + p["torso_lean"] + p["torso_swing"] * s
    q[11] = p["arm_amp"] * s
    q[15] = -p["arm_amp"] * s
    return q


def controller_targets(t, p):
    gait_end = gait_target(p["gait_time"], p)
    if t <= p["gait_time"]:
        ramp = smoothstep(t / p["ramp_time"])
        target = gait_target(t, p)
        return (1.0 - ramp) * WALK_BASE + ramp * target
    if t <= p["gait_time"] + p["recover_time"]:
        mix = smoothstep((t - p["gait_time"]) / p["recover_time"])
        return (1.0 - mix) * gait_end + mix * STAND_BASE
    return STAND_BASE


def rollout(params, render_prefix=None):
    import imageio.v2 as imageio

    params = dict(params)
    sim, qadr, vadr = make_sim()
    frames = []
    motion_end = params["gait_time"] + params["recover_time"]
    stand_ramp = min(4.0, max(1.5, 0.15 * params["stand_time"]))
    total_steps = int((motion_end + params["stand_time"]) / sim.model.opt.timestep)
    for i in range(total_steps):
        t = sim.data.time
        q = sim.data.qpos[qadr]
        v = sim.data.qvel[vadr]
        if t <= motion_end + stand_ramp:
            qdes = controller_targets(t, params)
            ctrl = sim.data.qfrc_bias[vadr] + KP * (qdes - q) - KD * v
        else:
            ctrl = STAND_HOLD_CTRL
        sim.data.ctrl[:] = np.clip(ctrl, sim.model.actuator_ctrlrange[:, 0], sim.model.actuator_ctrlrange[:, 1])
        sim.step(1)
        if render_prefix and i % 200 == 0:
            frames.append((i, sim.render(width=480, height=360)))
        if sim.pelvis_position()[2] < 0.15:
            break
    result = evaluate_rollout(sim)
    if render_prefix:
        for idx, frame in frames:
            imageio.imwrite(f"{render_prefix}_{idx:04d}.png", frame)
    return sim, result


def sample_params(rng, base=None):
    p = {k: v for k, v in dict(base).items() if not str(k).startswith("_")} if base is not None else {
        "freq": 1.6,
        "phase": 0.0,
        "ramp_time": 0.2,
        "gait_time": 0.9,
        "recover_time": 0.35,
        "stand_time": 35.0,
        "roll_bias": 0.02,
        "roll_amp": 0.12,
        "hip_bias": -0.08,
        "hip_amp": 0.30,
        "knee_stance": -0.02,
        "knee_swing": 0.55,
        "ankle_stance": 0.05,
        "ankle_swing": -0.28,
        "torso_lean": 0.05,
        "torso_swing": 0.0,
        "arm_amp": 0.8,
    }
    p["freq"] = float(np.clip(rng.normal(p["freq"], 0.18), 1.15, 2.4))
    p["phase"] = float(((p["phase"] + rng.normal(0.0, 0.2) + math.pi) % (2 * math.pi)) - math.pi)
    p["ramp_time"] = float(np.clip(rng.normal(p["ramp_time"], 0.05), 0.08, 0.35))
    p["gait_time"] = float(np.clip(rng.normal(p["gait_time"], 0.18), 0.35, 1.8))
    p["recover_time"] = float(np.clip(rng.normal(p["recover_time"], 0.12), 0.15, 1.2))
    p["stand_time"] = float(np.clip(rng.normal(p["stand_time"], 3.0), 14.0, 35.0))
    p["roll_bias"] = float(np.clip(rng.normal(p["roll_bias"], 0.02), -0.12, 0.12))
    p["roll_amp"] = float(np.clip(rng.normal(p["roll_amp"], 0.03), 0.0, 0.22))
    p["hip_bias"] = float(np.clip(rng.normal(p["hip_bias"], 0.04), -0.25, 0.12))
    p["hip_amp"] = float(np.clip(rng.normal(p["hip_amp"], 0.08), 0.05, 0.65))
    p["knee_stance"] = float(np.clip(rng.normal(p["knee_stance"], 0.05), -0.2, 0.2))
    p["knee_swing"] = float(np.clip(rng.normal(p["knee_swing"], 0.10), 0.15, 0.95))
    p["ankle_stance"] = float(np.clip(rng.normal(p["ankle_stance"], 0.04), -0.2, 0.18))
    p["ankle_swing"] = float(np.clip(rng.normal(p["ankle_swing"], 0.08), -0.5, 0.1))
    p["torso_lean"] = float(np.clip(rng.normal(p["torso_lean"], 0.04), -0.1, 0.25))
    p["torso_swing"] = float(np.clip(rng.normal(p["torso_swing"], 0.03), -0.08, 0.08))
    p["arm_amp"] = float(np.clip(rng.normal(p["arm_amp"], 0.15), 0.0, 1.4))
    return p


def mutate_params(rng, p, scale=0.18):
    out = {k: v for k, v in dict(p).items() if not str(k).startswith("_")}
    for key in list(out.keys()):
        if key == "phase":
            out[key] = float(((out[key] + rng.normal(0.0, scale) + math.pi) % (2 * math.pi)) - math.pi)
        else:
            out[key] = float(out[key] + rng.normal(0.0, scale * max(abs(out[key]), 0.1)))
    out["freq"] = float(np.clip(out["freq"], 0.95, 2.6))
    out["ramp_time"] = float(np.clip(out["ramp_time"], 0.06, 0.5))
    out["gait_time"] = float(np.clip(out["gait_time"], 0.25, 2.2))
    out["recover_time"] = float(np.clip(out["recover_time"], 0.12, 1.5))
    out["stand_time"] = float(np.clip(out["stand_time"], 12.0, 40.0))
    out["roll_bias"] = float(np.clip(out["roll_bias"], -0.15, 0.15))
    out["roll_amp"] = float(np.clip(out["roll_amp"], 0.0, 0.25))
    out["hip_bias"] = float(np.clip(out["hip_bias"], -0.3, 0.2))
    out["hip_amp"] = float(np.clip(out["hip_amp"], 0.02, 0.8))
    out["knee_stance"] = float(np.clip(out["knee_stance"], -0.25, 0.3))
    out["knee_swing"] = float(np.clip(out["knee_swing"], 0.05, 1.1))
    out["ankle_stance"] = float(np.clip(out["ankle_stance"], -0.3, 0.25))
    out["ankle_swing"] = float(np.clip(out["ankle_swing"], -0.7, 0.2))
    out["torso_lean"] = float(np.clip(out["torso_lean"], -0.15, 0.35))
    out["torso_swing"] = float(np.clip(out["torso_swing"], -0.15, 0.15))
    out["arm_amp"] = float(np.clip(out["arm_amp"], 0.0, 1.8))
    return out


def result_line(tag, res, params):
    return (
        f"{tag} score={res.score:.3f} pass={res.passed} x={res.final_x:.3f} "
        f"dist={res.final_target_distance:.3f} cps={res.checkpoint_hits}/3 "
        f"stable={res.stable_fraction:.3f} replay_min=({res.min_replay_z:.3f},{res.min_replay_up:.3f}) "
        f"settle_min=({res.min_settle_z:.3f},{res.min_settle_up:.3f}) params={params}"
    )


def main():
    rng = np.random.default_rng(11)
    best_params = None
    best_result = None
    best_sim = None

    seed_bank = [
        {
            "freq": 2.3372704833824316,
            "phase": -0.02611205101987002,
            "ramp_time": 0.519065788966189,
            "gait_time": 0.5257254986328088,
            "recover_time": 0.6420516557461161,
            "stand_time": 35.0,
            "roll_bias": 0.014066114516835836,
            "roll_amp": 0.15556617473112389,
            "hip_bias": -0.27674127792900927,
            "hip_amp": 0.14198158370463354,
            "knee_stance": -0.17474402835718195,
            "knee_swing": 0.06971996130916033,
            "ankle_stance": 0.1447570699584855,
            "ankle_swing": -0.37036644191674867,
            "torso_lean": -0.014983517278835599,
            "torso_swing": 0.03388627125687021,
            "arm_amp": 0.7594942998892076,
        },
        {
            "freq": 2.15,
            "phase": 0.0,
            "ramp_time": 0.30,
            "gait_time": 0.75,
            "recover_time": 0.55,
            "stand_time": 35.0,
            "roll_bias": 0.03,
            "roll_amp": 0.12,
            "hip_bias": -0.16,
            "hip_amp": 0.20,
            "knee_stance": -0.10,
            "knee_swing": 0.25,
            "ankle_stance": 0.08,
            "ankle_swing": -0.24,
            "torso_lean": 0.00,
            "torso_swing": 0.00,
            "arm_amp": 0.55,
        },
        {
            "freq": 1.85,
            "phase": 0.15,
            "ramp_time": 0.25,
            "gait_time": 1.05,
            "recover_time": 0.60,
            "stand_time": 35.0,
            "roll_bias": 0.01,
            "roll_amp": 0.10,
            "hip_bias": -0.12,
            "hip_amp": 0.24,
            "knee_stance": -0.06,
            "knee_swing": 0.36,
            "ankle_stance": 0.06,
            "ankle_swing": -0.26,
            "torso_lean": 0.02,
            "torso_swing": 0.0,
            "arm_amp": 0.68,
        },
    ]

    for i in range(80):
        if time.time() > SEARCH_DEADLINE:
            break
        if i < len(seed_bank):
            params = seed_bank[i]
        elif best_params is not None and rng.random() < 0.85:
            params = mutate_params(rng, best_params, scale=0.12 if best_result.score > 0.55 else 0.20)
        else:
            params = sample_params(rng)
        sim, res = rollout(params)
        if best_result is None or res.score > best_result.score:
            best_result = res
            best_params = params
            best_sim = sim
            sim.save_final_state("/work/final_state.npz")
            print(result_line(f"best[{i}]", res, params), flush=True)
        if res.passed:
            print("PASS FOUND", flush=True)
            break

    if best_sim is not None:
        best_sim.save_final_state("/work/final_state.npz")
        print("FINAL", result_line("best", best_result, best_params), flush=True)
        _, _ = rollout(best_params, render_prefix="/work/best_frame")


if __name__ == "__main__":
    main()
