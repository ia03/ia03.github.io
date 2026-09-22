import math
import time
import numpy as np

from sim import Sim, TARGET_XY, TARGET_RADIUS, MIN_BOX_Z, MIN_PELVIS_Z, MIN_TORSO_UP, MAX_BOX_TO_PELVIS


DT = 0.002
DEADLINE = 1744510239.0  # 2026-04-13 02:10:39 UTC


def stable_mask(sim: Sim):
    pelvis = sim.pelvis_position()
    box = sim.box_position()
    return (
        pelvis[2] >= MIN_PELVIS_Z
        and sim.torso_up() >= MIN_TORSO_UP
        and box[2] >= MIN_BOX_Z
        and sim.box_distance_to_pelvis() <= MAX_BOX_TO_PELVIS
    )


def settle_metrics(sim: Sim, steps=500):
    min_pelvis_z = 1e9
    min_torso_up = 1e9
    min_box_z = 1e9
    max_box_dist = -1e9
    stable_count = 0
    for _ in range(steps):
        sim.step(1)
        pelvis = sim.pelvis_position()
        box = sim.box_position()
        torso_up = sim.torso_up()
        box_dist = sim.box_distance_to_pelvis()
        min_pelvis_z = min(min_pelvis_z, pelvis[2])
        min_torso_up = min(min_torso_up, torso_up)
        min_box_z = min(min_box_z, box[2])
        max_box_dist = max(max_box_dist, box_dist)
        stable_count += stable_mask(sim)
    return {
        "min_pelvis_z": min_pelvis_z,
        "min_torso_up": min_torso_up,
        "min_box_z": min_box_z,
        "max_box_dist": max_box_dist,
        "stable_frac": stable_count / steps,
    }


def score_candidate(trace, settle, replay_steps):
    pelvis_x = trace["pelvis_x"]
    box_x = trace["box_x"]
    stable_frac = trace["stable_frac"]
    pelvis_checkpoints = sum(pelvis_x.max() >= cp for cp in (0.05, 0.09, 0.13)) / 3.0
    box_checkpoints = sum(box_x.max() >= cp for cp in (0.00, 0.04, 0.08)) / 3.0
    pelvis_dist = np.linalg.norm(trace["pelvis_final"][:2] - TARGET_XY)
    box_dist = np.linalg.norm(trace["box_final"][:2] - TARGET_XY)
    pelvis_dist_progress = 1.0 if pelvis_dist <= TARGET_RADIUS else max(0.0, 1 - (pelvis_dist - TARGET_RADIUS) / 0.25)
    box_dist_progress = 1.0 if box_dist <= TARGET_RADIUS else max(0.0, 1 - (box_dist - TARGET_RADIUS) / 0.11)
    carry_progress = max(0.0, 1 - max(0.0, settle["max_box_dist"] - MAX_BOX_TO_PELVIS) / MAX_BOX_TO_PELVIS)
    box_height_progress = min(1.0, settle["min_box_z"] / MIN_BOX_Z)
    stability_progress = min(1.0, stable_frac / 0.90)
    efficiency = min(1.0, 2600.0 / replay_steps)
    pass_like = (
        replay_steps >= 30
        and pelvis_checkpoints == 1.0
        and box_checkpoints == 1.0
        and stable_frac >= 0.90
        and pelvis_dist <= TARGET_RADIUS
        and box_dist <= TARGET_RADIUS
        and settle["min_pelvis_z"] >= MIN_PELVIS_Z
        and settle["min_torso_up"] >= MIN_TORSO_UP
        and settle["min_box_z"] >= MIN_BOX_Z
        and settle["max_box_dist"] <= MAX_BOX_TO_PELVIS
    )
    progress = (
        0.15 * box_dist_progress
        + 0.30 * carry_progress
        + 0.25 * box_height_progress
        + 0.10 * stability_progress
        + 0.03 * box_checkpoints
        + 0.02 * pelvis_dist_progress
        + 0.15 * efficiency
    )
    bonus = 0.25 * pelvis_checkpoints + 0.25 * min(1.0, pelvis_x.max() / 0.13)
    return progress + bonus + (1.0 if pass_like else 0.0), {
        "pass_like": pass_like,
        "progress": progress,
        "pelvis_checkpoints": pelvis_checkpoints,
        "box_checkpoints": box_checkpoints,
        "stable_frac": stable_frac,
        "pelvis_max_x": float(pelvis_x.max()),
        "pelvis_dist": float(pelvis_dist),
        "box_dist": float(box_dist),
        **settle,
    }


def make_ctrl(home, p, step_idx):
    ctrl = home.copy()
    t = step_idx * DT
    ramp = min(1.0, step_idx / max(1.0, p["ramp_steps"]))
    phi = 2 * math.pi * p["freq"] * t + p["phase"]
    s = math.sin(phi)
    c = math.cos(phi)
    lf = max(0.0, s)
    rf = max(0.0, -s)

    ctrl[10] = p["torso"]

    ctrl[11] = p["l_sh_pitch"]
    ctrl[12] = p["l_sh_roll"]
    ctrl[13] = p["l_sh_yaw"]
    ctrl[14] = p["l_elbow"]
    ctrl[15] = p["r_sh_pitch"]
    ctrl[16] = p["r_sh_roll"]
    ctrl[17] = p["r_sh_yaw"]
    ctrl[18] = p["r_elbow"]

    ctrl[0] = p["yaw_bias"] + ramp * p["yaw_amp"] * s
    ctrl[5] = -p["yaw_bias"] - ramp * p["yaw_amp"] * s
    ctrl[1] = p["hip_roll_bias"] + ramp * (p["hip_roll_amp"] * s + p["hip_roll_c"] * c)
    ctrl[6] = -p["hip_roll_bias"] - ramp * (p["hip_roll_amp"] * s - p["hip_roll_c"] * c)

    ctrl[2] = home[2] + p["hip_pitch_bias"] + ramp * (p["hip_pitch_amp"] * s + p["hip_pitch_c"] * c)
    ctrl[7] = home[7] + p["hip_pitch_bias"] - ramp * (p["hip_pitch_amp"] * s - p["hip_pitch_c"] * c)

    ctrl[3] = home[3] + p["knee_bias"] + ramp * (p["knee_stance"] + p["knee_amp"] * lf - p["knee_retract"] * rf)
    ctrl[8] = home[8] + p["knee_bias"] + ramp * (p["knee_stance"] + p["knee_amp"] * rf - p["knee_retract"] * lf)

    ctrl[4] = home[4] + p["ankle_bias"] + ramp * (p["ankle_amp"] * lf - p["ankle_push"] * rf)
    ctrl[9] = home[9] + p["ankle_bias"] + ramp * (p["ankle_amp"] * rf - p["ankle_push"] * lf)
    return ctrl


def run_candidate(p, save_path=None):
    sim = Sim()
    home = sim.home_ctrl()
    pelvis_x = []
    box_x = []
    stable_count = 0
    for i in range(p["steps"]):
        sim.data.ctrl[:] = make_ctrl(home, p, i)
        sim.step(1)
        pelvis_x.append(sim.pelvis_position()[0])
        box_x.append(sim.box_position()[0])
        stable_count += stable_mask(sim)
    trace = {
        "pelvis_x": np.array(pelvis_x),
        "box_x": np.array(box_x),
        "stable_frac": stable_count / max(1, p["steps"]),
        "pelvis_final": sim.pelvis_position(),
        "box_final": sim.box_position(),
    }
    settle = settle_metrics(sim)
    score, info = score_candidate(trace, settle, p["steps"])
    if save_path is not None:
        sim.save_final_state(save_path)
    return score, info


def random_params(rng):
    return {
        "steps": int(rng.integers(650, 1300)),
        "ramp_steps": int(rng.integers(60, 220)),
        "freq": float(rng.uniform(0.45, 1.35)),
        "phase": float(rng.uniform(-math.pi, math.pi)),
        "torso": float(rng.uniform(-0.25, 0.35)),
        "yaw_bias": float(rng.uniform(-0.03, 0.03)),
        "yaw_amp": float(rng.uniform(0.0, 0.08)),
        "hip_roll_bias": float(rng.uniform(-0.12, 0.12)),
        "hip_roll_amp": float(rng.uniform(0.0, 0.22)),
        "hip_roll_c": float(rng.uniform(-0.12, 0.12)),
        "hip_pitch_bias": float(rng.uniform(-0.24, 0.18)),
        "hip_pitch_amp": float(rng.uniform(0.12, 0.55)),
        "hip_pitch_c": float(rng.uniform(-0.15, 0.15)),
        "knee_bias": float(rng.uniform(-0.15, 0.35)),
        "knee_stance": float(rng.uniform(-0.05, 0.18)),
        "knee_amp": float(rng.uniform(0.12, 0.75)),
        "knee_retract": float(rng.uniform(0.0, 0.55)),
        "ankle_bias": float(rng.uniform(-0.22, 0.18)),
        "ankle_amp": float(rng.uniform(-0.24, 0.18)),
        "ankle_push": float(rng.uniform(-0.22, 0.18)),
        "l_sh_pitch": float(rng.uniform(-1.2, 1.2)),
        "l_sh_roll": float(rng.uniform(-1.2, 1.2)),
        "l_sh_yaw": float(rng.uniform(-1.0, 1.0)),
        "l_elbow": float(rng.uniform(-1.5, 0.2)),
        "r_sh_pitch": float(rng.uniform(-1.2, 1.2)),
        "r_sh_roll": float(rng.uniform(-1.2, 1.2)),
        "r_sh_yaw": float(rng.uniform(-1.0, 1.0)),
        "r_elbow": float(rng.uniform(-1.5, 0.2)),
    }


def mutate_params(rng, base):
    out = dict(base)
    for key, sigma in [
        ("steps", 120),
        ("ramp_steps", 25),
        ("freq", 0.10),
        ("phase", 0.50),
        ("torso", 0.08),
        ("yaw_bias", 0.01),
        ("yaw_amp", 0.02),
        ("hip_roll_bias", 0.04),
        ("hip_roll_amp", 0.05),
        ("hip_roll_c", 0.04),
        ("hip_pitch_bias", 0.06),
        ("hip_pitch_amp", 0.08),
        ("hip_pitch_c", 0.05),
        ("knee_bias", 0.06),
        ("knee_stance", 0.05),
        ("knee_amp", 0.10),
        ("knee_retract", 0.08),
        ("ankle_bias", 0.05),
        ("ankle_amp", 0.06),
        ("ankle_push", 0.06),
        ("l_sh_pitch", 0.20),
        ("l_sh_roll", 0.20),
        ("l_sh_yaw", 0.16),
        ("l_elbow", 0.20),
        ("r_sh_pitch", 0.20),
        ("r_sh_roll", 0.20),
        ("r_sh_yaw", 0.16),
        ("r_elbow", 0.20),
    ]:
        if rng.random() < 0.85:
            out[key] += float(rng.normal(0.0, sigma))
    out["steps"] = int(np.clip(out["steps"], 500, 1500))
    out["ramp_steps"] = int(np.clip(out["ramp_steps"], 20, 300))
    for key, lo, hi in [
        ("freq", 0.35, 1.5),
        ("phase", -math.pi, math.pi),
        ("torso", -0.35, 0.45),
        ("yaw_bias", -0.08, 0.08),
        ("yaw_amp", 0.0, 0.12),
        ("hip_roll_bias", -0.2, 0.2),
        ("hip_roll_amp", 0.0, 0.30),
        ("hip_roll_c", -0.2, 0.2),
        ("hip_pitch_bias", -0.35, 0.25),
        ("hip_pitch_amp", 0.05, 0.75),
        ("hip_pitch_c", -0.25, 0.25),
        ("knee_bias", -0.2, 0.5),
        ("knee_stance", -0.1, 0.28),
        ("knee_amp", 0.05, 1.0),
        ("knee_retract", 0.0, 0.8),
        ("ankle_bias", -0.3, 0.25),
        ("ankle_amp", -0.3, 0.25),
        ("ankle_push", -0.3, 0.25),
        ("l_sh_pitch", -1.5, 1.5),
        ("l_sh_roll", -1.5, 1.5),
        ("l_sh_yaw", -1.2, 1.2),
        ("l_elbow", -1.8, 0.4),
        ("r_sh_pitch", -1.5, 1.5),
        ("r_sh_roll", -1.5, 1.5),
        ("r_sh_yaw", -1.2, 1.2),
        ("r_elbow", -1.8, 0.4),
    ]:
        out[key] = float(np.clip(out[key], lo, hi))
    return out


def main():
    rng = np.random.default_rng(0)
    # Deterministic seed candidates to warm-start around obvious standing/walking patterns.
    seeds = [
        {
            "steps": 900, "ramp_steps": 140, "freq": 0.75, "phase": 0.0, "torso": 0.05,
            "yaw_bias": 0.0, "yaw_amp": 0.01, "hip_roll_bias": 0.02, "hip_roll_amp": 0.08, "hip_roll_c": 0.0,
            "hip_pitch_bias": 0.05, "hip_pitch_amp": 0.22, "hip_pitch_c": 0.0,
            "knee_bias": 0.06, "knee_stance": 0.04, "knee_amp": 0.30, "knee_retract": 0.06,
            "ankle_bias": 0.02, "ankle_amp": -0.07, "ankle_push": 0.02,
            "l_sh_pitch": 0.6, "l_sh_roll": 0.35, "l_sh_yaw": 0.2, "l_elbow": -0.9,
            "r_sh_pitch": 0.6, "r_sh_roll": -0.35, "r_sh_yaw": -0.2, "r_elbow": -0.9,
        },
        {
            "steps": 1100, "ramp_steps": 90, "freq": 0.95, "phase": 0.2, "torso": 0.10,
            "yaw_bias": 0.0, "yaw_amp": 0.02, "hip_roll_bias": 0.0, "hip_roll_amp": 0.12, "hip_roll_c": 0.03,
            "hip_pitch_bias": 0.10, "hip_pitch_amp": 0.34, "hip_pitch_c": 0.04,
            "knee_bias": 0.12, "knee_stance": 0.05, "knee_amp": 0.42, "knee_retract": 0.10,
            "ankle_bias": 0.04, "ankle_amp": -0.12, "ankle_push": 0.04,
            "l_sh_pitch": 0.8, "l_sh_roll": 0.55, "l_sh_yaw": 0.35, "l_elbow": -1.1,
            "r_sh_pitch": 0.8, "r_sh_roll": -0.55, "r_sh_yaw": -0.35, "r_elbow": -1.1,
        },
    ]

    best_score = -1e9
    best_params = None
    population = []

    for seed in seeds:
        score, info = run_candidate(seed, save_path="/work/final_state.npz")
        print("seed", score, info)
        if score > best_score:
            best_score = score
            best_params = dict(seed)
            population.append((score, dict(seed)))

    while time.time() < DEADLINE - 25:
        if best_params is None or len(population) < 4:
            p = random_params(rng)
        elif rng.random() < 0.3:
            p = random_params(rng)
        else:
            parent = population[int(rng.integers(0, len(population)))][1]
            p = mutate_params(rng, parent)
        score, info = run_candidate(p)
        if score > best_score:
            best_score = score
            best_params = dict(p)
            population.append((score, dict(p)))
            population = sorted(population, key=lambda item: item[0], reverse=True)[:8]
            run_candidate(best_params, save_path="/work/final_state.npz")
            print("best", round(best_score, 4), info, best_params, flush=True)
            population.append((best_score, dict(best_params)))
            population = sorted(population, key=lambda item: item[0], reverse=True)[:8]
        elif rng.random() < 0.05:
            population.append((score, dict(p)))
            population = sorted(population, key=lambda item: item[0], reverse=True)[:8]

    if best_params is not None:
        score, info = run_candidate(best_params, save_path="/work/final_state.npz")
        print("final", score, info, best_params)


if __name__ == "__main__":
    main()
