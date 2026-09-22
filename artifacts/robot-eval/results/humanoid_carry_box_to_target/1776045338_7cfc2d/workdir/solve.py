import math
import time
import numpy as np

from sim import Sim, TARGET_XY, MIN_PELVIS_Z, MIN_TORSO_UP, MIN_BOX_Z, MAX_BOX_TO_PELVIS


def evaluate_ctrl_trace(ctrl_trace: np.ndarray):
    sim = Sim()
    init_ok = float(np.linalg.norm(sim.pelvis_position()[:2])) <= 0.15
    replay_steps = len(ctrl_trace)
    stable_steps = 0
    pelvis_x_max = -1e9
    box_x_max = -1e9

    for ctrl in ctrl_trace:
        sim.data.ctrl[:] = ctrl
        sim.step()
        pelvis = sim.pelvis_position()
        box = sim.box_position()
        pelvis_x_max = max(pelvis_x_max, float(pelvis[0]))
        box_x_max = max(box_x_max, float(box[0]))
        stable_steps += int(
            pelvis[2] >= MIN_PELVIS_Z
            and sim.torso_up() >= MIN_TORSO_UP
            and box[2] >= MIN_BOX_Z
            and sim.box_distance_to_pelvis() <= MAX_BOX_TO_PELVIS
        )

    stable_frac = stable_steps / max(1, replay_steps)
    min_settle_pelvis_z = 1e9
    min_settle_torso_up = 1e9
    min_settle_box_z = 1e9
    max_settle_box_dist = -1e9
    for _ in range(500):
        sim.step()
        pelvis = sim.pelvis_position()
        box = sim.box_position()
        min_settle_pelvis_z = min(min_settle_pelvis_z, float(pelvis[2]))
        min_settle_torso_up = min(min_settle_torso_up, float(sim.torso_up()))
        min_settle_box_z = min(min_settle_box_z, float(box[2]))
        max_settle_box_dist = max(max_settle_box_dist, float(sim.box_distance_to_pelvis()))

    pelvis_xy_dist = float(np.linalg.norm(sim.pelvis_position()[:2] - TARGET_XY))
    box_xy_dist = float(np.linalg.norm(sim.box_position()[:2] - TARGET_XY))
    metrics = {
        "passed": (
            replay_steps >= 30
            and init_ok
            and pelvis_x_max >= 0.13
            and box_x_max >= 0.08
            and stable_frac >= 0.90
            and pelvis_xy_dist <= 0.20
            and box_xy_dist <= 0.20
            and min_settle_pelvis_z >= MIN_PELVIS_Z
            and min_settle_torso_up >= MIN_TORSO_UP
            and min_settle_box_z >= MIN_BOX_Z
            and max_settle_box_dist <= MAX_BOX_TO_PELVIS
        ),
        "replay_steps": replay_steps,
        "pelvis_x_max": pelvis_x_max,
        "box_x_max": box_x_max,
        "stable_frac": stable_frac,
        "pelvis_xy_dist": pelvis_xy_dist,
        "box_xy_dist": box_xy_dist,
        "min_settle_pelvis_z": min_settle_pelvis_z,
        "min_settle_torso_up": min_settle_torso_up,
        "min_settle_box_z": min_settle_box_z,
        "max_settle_box_dist": max_settle_box_dist,
    }
    metrics["score"] = score_metrics(metrics)
    return metrics


def score_metrics(m):
    score = 0.0
    score += 5.0 * min(max(m["pelvis_x_max"], 0.0) / 0.13, 1.5)
    score += 1.0 * min(max(m["box_x_max"], 0.0) / 0.08, 1.0)
    score += 2.0 * min(m["stable_frac"] / 0.90, 1.0)
    score += 1.0 * max(0.0, 1.0 - max(0.0, m["pelvis_xy_dist"] - 0.20) / 0.25)
    score += 1.0 * max(0.0, 1.0 - max(0.0, m["box_xy_dist"] - 0.20) / 0.11)
    score += 1.0 * min(m["min_settle_pelvis_z"] / MIN_PELVIS_Z, 1.0)
    score += 1.0 * min((m["min_settle_torso_up"] + 1.0) / (MIN_TORSO_UP + 1.0), 1.0)
    score += 1.0 * min(m["min_settle_box_z"] / MIN_BOX_Z, 1.0)
    score += 1.0 * max(0.0, 1.0 - max(0.0, m["max_settle_box_dist"] - MAX_BOX_TO_PELVIS) / MAX_BOX_TO_PELVIS)
    score += 0.5 * min(2600.0 / max(1, m["replay_steps"]), 1.0)
    if m["passed"]:
        score += 100.0
    return score


def make_move_ctrl(home, arm_pose, p, t):
    phase = 2.0 * math.pi * t / max(1, p["period"])
    s = math.sin(phase)
    c = math.cos(phase)
    s2 = math.sin(2.0 * phase)
    ctrl = home.copy()
    ctrl[10:] = arm_pose
    ctrl[0] = p["yaw_amp"] * c
    ctrl[5] = -p["yaw_amp"] * c
    ctrl[1] = p["roll_bias"] + p["roll_amp"] * c
    ctrl[6] = -p["roll_bias"] - p["roll_amp"] * c
    ctrl[2] = p["l_hip_base"] + p["hip_amp"] * s + p["hip_bias"]
    ctrl[3] = p["l_knee_base"] - p["knee_amp"] * s + p["knee_bias"]
    ctrl[4] = p["l_ankle_base"] + p["ankle_amp"] * s + p["ankle_bias"]
    ctrl[7] = p["r_hip_base"] - p["hip_amp"] * s - p["hip_bias"]
    ctrl[8] = p["r_knee_base"] + p["knee_amp"] * s + p["knee_bias"]
    ctrl[9] = p["r_ankle_base"] - p["ankle_amp"] * s - p["ankle_bias"]
    ctrl[10] = p["move_torso"] + p["torso_gait"] * s + p["torso_2"] * s2
    return ctrl


def make_hold_ctrl(home, arm_pose, p):
    ctrl = home.copy()
    ctrl[10:] = arm_pose
    ctrl[1] = p["hold_roll"]
    ctrl[6] = -p["hold_roll"]
    ctrl[2] = p["hold_l_hip"]
    ctrl[3] = p["hold_l_knee"]
    ctrl[4] = p["hold_l_ankle"]
    ctrl[7] = p["hold_r_hip"]
    ctrl[8] = p["hold_r_knee"]
    ctrl[9] = p["hold_r_ankle"]
    ctrl[10] = p["hold_torso"]
    return ctrl


def rollout(params, save=False, save_path="/work/final_state.npz"):
    sim = Sim()
    home = sim.home_ctrl()
    arm_pose = np.array([
        params["hold_torso"],
        params["lsp"],
        params["lsr"],
        params["lsy"],
        params["lel"],
        params["rsp"],
        params["rsr"],
        params["rsy"],
        params["rel"],
    ])
    hold_ctrl = make_hold_ctrl(home, arm_pose, params)
    sim.data.ctrl[:] = hold_ctrl
    sim.step(params["settle0"])
    for t in range(params["move_steps"]):
        sim.data.ctrl[:] = make_move_ctrl(home, arm_pose, params, t)
        sim.step()
    sim.data.ctrl[:] = hold_ctrl
    sim.step(params["settle1"])
    if save:
        sim.save_final_state(save_path)
    ctrl_trace = np.array(sim._ctrl_trace, dtype=float)
    return evaluate_ctrl_trace(ctrl_trace), ctrl_trace


def sample_params(rng):
    return {
        "settle0": int(rng.integers(10, 50)),
        "settle1": int(rng.integers(20, 120)),
        "move_steps": int(rng.integers(40, 220)),
        "period": int(rng.integers(22, 60)),
        "move_torso": rng.uniform(-0.35, 0.35),
        "torso_gait": rng.uniform(-0.20, 0.20),
        "torso_2": rng.uniform(-0.12, 0.12),
        "hold_torso": rng.uniform(-0.10, 0.20),
        "lsp": rng.uniform(0.0, 1.1),
        "lsr": rng.uniform(0.0, 0.55),
        "lsy": rng.uniform(-0.35, 0.35),
        "lel": rng.uniform(0.0, 1.35),
        "rsp": rng.uniform(0.0, 1.1),
        "rsr": -rng.uniform(0.0, 0.55),
        "rsy": rng.uniform(-0.35, 0.35),
        "rel": rng.uniform(0.0, 1.35),
        "l_hip_base": rng.uniform(-0.58, -0.18),
        "l_knee_base": rng.uniform(0.45, 1.10),
        "l_ankle_base": rng.uniform(-0.75, -0.15),
        "r_hip_base": rng.uniform(-0.58, -0.18),
        "r_knee_base": rng.uniform(0.45, 1.10),
        "r_ankle_base": rng.uniform(-0.75, -0.15),
        "hip_amp": rng.uniform(0.00, 0.30),
        "knee_amp": rng.uniform(0.00, 0.40),
        "ankle_amp": rng.uniform(0.00, 0.25),
        "hip_bias": rng.uniform(-0.12, 0.12),
        "knee_bias": rng.uniform(-0.18, 0.18),
        "ankle_bias": rng.uniform(-0.12, 0.12),
        "yaw_amp": rng.uniform(-0.10, 0.10),
        "roll_bias": rng.uniform(-0.10, 0.10),
        "roll_amp": rng.uniform(0.00, 0.12),
        "hold_roll": rng.uniform(-0.05, 0.05),
        "hold_l_hip": rng.uniform(-0.50, -0.22),
        "hold_l_knee": rng.uniform(0.55, 0.95),
        "hold_l_ankle": rng.uniform(-0.55, -0.20),
        "hold_r_hip": rng.uniform(-0.50, -0.22),
        "hold_r_knee": rng.uniform(0.55, 0.95),
        "hold_r_ankle": rng.uniform(-0.55, -0.20),
    }


def mutate_params(base, rng):
    p = dict(base)
    for k, v in list(p.items()):
        if isinstance(v, int):
            scale = max(2, abs(v) // 5)
            p[k] = max(1, int(v + rng.integers(-scale, scale + 1)))
        else:
            p[k] = float(v + rng.normal(0.0, 0.06))
    return p


def main():
    rng = np.random.default_rng(0)
    best_params = {
        "settle0": 20,
        "settle1": 60,
        "move_steps": 60,
        "period": 36,
        "move_torso": 0.0,
        "torso_gait": 0.0,
        "torso_2": 0.0,
        "hold_torso": 0.0,
        "lsp": 0.2,
        "lsr": 0.1,
        "lsy": 0.0,
        "lel": 0.2,
        "rsp": 0.2,
        "rsr": -0.1,
        "rsy": 0.0,
        "rel": 0.2,
        "l_hip_base": -0.4,
        "l_knee_base": 0.8,
        "l_ankle_base": -0.4,
        "r_hip_base": -0.4,
        "r_knee_base": 0.8,
        "r_ankle_base": -0.4,
        "hip_amp": 0.0,
        "knee_amp": 0.0,
        "ankle_amp": 0.0,
        "hip_bias": 0.0,
        "knee_bias": 0.0,
        "ankle_bias": 0.0,
        "yaw_amp": 0.0,
        "roll_bias": 0.0,
        "roll_amp": 0.0,
        "hold_roll": 0.0,
        "hold_l_hip": -0.4,
        "hold_l_knee": 0.8,
        "hold_l_ankle": -0.4,
        "hold_r_hip": -0.4,
        "hold_r_knee": 0.8,
        "hold_r_ankle": -0.4,
    }
    best_metrics, _ = rollout(best_params, save=True)
    best_score = best_metrics["score"]
    print("initial", best_metrics, flush=True)

    end_time = time.time() + 8.5 * 60.0
    it = 0
    while time.time() < end_time:
        if it < 30:
            params = sample_params(rng)
        elif rng.random() < 0.7:
            params = mutate_params(best_params, rng)
        else:
            params = sample_params(rng)
        metrics, _ = rollout(params, save=False)
        if metrics["score"] > best_score:
            best_params = params
            best_score = metrics["score"]
            best_metrics, _ = rollout(best_params, save=True)
            print(f"iter {it} best", best_metrics, flush=True)
            if best_metrics["passed"]:
                break
        it += 1

    print("best_params", best_params)
    print("best_metrics", best_metrics)


if __name__ == "__main__":
    main()
