import json
import os
import sys
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
H1_SCENE_XML = os.path.join(MENAGERIE, "unitree_h1", "scene.xml")
TABLE_CENTER = (1.35, 0.0, 0.38)
TABLE_HALF_SIZE = (0.22, 0.32, 0.38)
TARGET_XY = np.array([1.00, 0.0])
TARGET_RADIUS = 0.45
TOUCH_POINT = np.array([1.18, 0.0, 0.83])
TOUCH_RADIUS = 0.12
MIN_PELVIS_Z = 0.23
MIN_TORSO_UP = 0.55
FORWARD_SIM_STEPS = 500
START_RADIUS = 0.15
PROGRESS_CHECKPOINTS = (0.25, 0.55, 0.80)
MIN_TRACE_SAMPLES = 30
MIN_STABLE_TRACE_FRACTION = 0.35
TRACE_POS_TOL = 0.04
TRACE_SCALAR_TOL = 0.03
IDEAL_REPLAY_STEPS = 2200


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def staged_progress(*values):
    vals = [clamp01(v) for v in values]
    if not vals:
        return 0.0
    total = 0.0
    prefix = 1.0
    weight_total = 0.0
    for index, value in enumerate(vals, start=1):
        prefix *= value
        total += index * prefix
        weight_total += index
    return float(total / weight_total)


def weighted_subscore(*components):
    total_weight = sum(weight for _, _, weight in components)
    if total_weight <= 0:
        return 0.0
    return float(sum(clamp01(value) * weight for _, value, weight in components) / total_weight)


def build_model():
    spec = mujoco.MjSpec.from_file(H1_SCENE_XML)
    world = spec.worldbody
    table = world.add_body(name="table")
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(TABLE_CENTER),
        size=list(TABLE_HALF_SIZE),
        rgba=[0.55, 0.35, 0.2, 1],
    )
    world.add_geom(
        name="floor_target_marker",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[float(TARGET_XY[0]), float(TARGET_XY[1]), 0.005],
        size=[TARGET_RADIUS, 0.005, 0],
        contype=0,
        conaffinity=0,
        rgba=[0.15, 0.7, 0.25, 0.30],
    )
    world.add_site(
        name="touch_marker",
        pos=list(TOUCH_POINT),
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[TOUCH_RADIUS, 0, 0],
        rgba=[0.9, 0.15, 0.15, 0.6],
    )
    return spec.compile()


def reset_canonical(model, data):
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def grade(path):
    try:
        saved = np.load(path)
    except FileNotFoundError:
        return False, {"error": "no final_state.npz produced"}
    except Exception as exc:
        return False, {"error": f"failed to load npz: {exc}"}

    model = build_model()
    data = mujoco.MjData(model)
    if "ctrl_trace" not in saved.files:
        return False, {"error": "ctrl_trace missing from saved state"}
    ctrl_trace = saved["ctrl_trace"]
    if ctrl_trace.ndim != 2 or ctrl_trace.shape[1] != model.nu:
        return False, {"error": "ctrl_trace has unexpected shape"}
    reset_canonical(model, data)

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    left_elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_elbow_link")
    right_elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_elbow_link")
    trace_pelvis = []
    trace_torso_up = []
    trace_touch_distance = []
    for ctrl in ctrl_trace:
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        trace_pelvis.append(data.xpos[pelvis_id].copy())
        trace_torso_up.append(float(data.xmat[torso_id].reshape(3, 3)[2, 2]))
        trace_touch_distance.append(
            min(
                float(np.linalg.norm(data.xpos[left_elbow_id] - TOUCH_POINT)),
                float(np.linalg.norm(data.xpos[right_elbow_id] - TOUCH_POINT)),
            )
        )
    trace_pelvis = np.array(trace_pelvis, dtype=float).reshape(-1, 3)
    trace_torso_up = np.array(trace_torso_up, dtype=float)
    trace_touch_distance = np.array(trace_touch_distance, dtype=float)
    start_distance = float(np.linalg.norm(trace_pelvis[0, :2])) if trace_pelvis.size else float("inf")
    checkpoint_hits = [bool(np.any(trace_pelvis[:, 0] >= checkpoint)) for checkpoint in PROGRESS_CHECKPOINTS] if trace_pelvis.size else [False] * len(PROGRESS_CHECKPOINTS)
    stable_mask = (trace_pelvis[:, 2] >= MIN_PELVIS_Z) & (trace_torso_up >= MIN_TORSO_UP) if trace_pelvis.size else np.array([], dtype=bool)
    stable_fraction = float(np.mean(stable_mask)) if stable_mask.size else 0.0
    best_trace_touch = float(np.min(trace_touch_distance)) if trace_touch_distance.size else float("inf")

    min_pelvis_z = float(data.xpos[pelvis_id][2])
    min_torso_up = float(data.xmat[torso_id].reshape(3, 3)[2, 2])
    best_touch = min(
        float(np.linalg.norm(data.xpos[left_elbow_id] - TOUCH_POINT)),
        float(np.linalg.norm(data.xpos[right_elbow_id] - TOUCH_POINT)),
    )
    for _ in range(FORWARD_SIM_STEPS):
        mujoco.mj_step(model, data)
        min_pelvis_z = min(min_pelvis_z, float(data.xpos[pelvis_id][2]))
        min_torso_up = min(min_torso_up, float(data.xmat[torso_id].reshape(3, 3)[2, 2]))
        best_touch = min(
            best_touch,
            float(np.linalg.norm(data.xpos[left_elbow_id] - TOUCH_POINT)),
            float(np.linalg.norm(data.xpos[right_elbow_id] - TOUCH_POINT)),
        )

    pelvis_pos = data.xpos[pelvis_id].copy()
    distance = float(np.linalg.norm(pelvis_pos[:2] - TARGET_XY))
    success = (
        trace_pelvis.shape[0] >= MIN_TRACE_SAMPLES
        and start_distance <= START_RADIUS
        and all(checkpoint_hits)
        and stable_fraction >= MIN_STABLE_TRACE_FRACTION
        and best_trace_touch <= TOUCH_RADIUS
        and distance <= TARGET_RADIUS
        and min_pelvis_z >= MIN_PELVIS_Z
        and min_torso_up >= MIN_TORSO_UP
        and best_touch <= TOUCH_RADIUS
    )
    checkpoint_progress = sum(checkpoint_hits) / len(PROGRESS_CHECKPOINTS)
    stability_progress = clamp01(stable_fraction / MIN_STABLE_TRACE_FRACTION)
    best_touch_observed = min(best_trace_touch, best_touch)
    distance_progress = 1.0 if distance <= TARGET_RADIUS else clamp01(1.0 - ((distance - TARGET_RADIUS) / TARGET_RADIUS))
    touch_progress = 1.0 if best_touch_observed <= TOUCH_RADIUS else clamp01(1.0 - ((best_touch_observed - TOUCH_RADIUS) / TOUCH_RADIUS))
    settle_posture_progress = 0.5 * clamp01(min_pelvis_z / MIN_PELVIS_Z) + 0.5 * clamp01(min_torso_up / MIN_TORSO_UP)
    replay_steps = int(ctrl_trace.shape[0])
    efficiency_progress = clamp01(IDEAL_REPLAY_STEPS / max(replay_steps, 1))
    score_components = [
        {"name": "checkpoint_progress", "value": checkpoint_progress, "weight": 0.15},
        {"name": "touch_progress", "value": touch_progress, "weight": 0.30},
        {"name": "stability_progress", "value": stability_progress, "weight": 0.20},
        {"name": "distance_progress", "value": distance_progress, "weight": 0.15},
        {"name": "settle_posture_progress", "value": settle_posture_progress, "weight": 0.10},
        {"name": "efficiency_progress", "value": efficiency_progress, "weight": 0.10},
    ]
    progress_score = weighted_subscore(*[(item["name"], item["value"], item["weight"]) for item in score_components])
    return success, {
        "pelvis_xyz_after": [float(x) for x in pelvis_pos],
        "start_distance_from_origin": start_distance,
        "checkpoint_hits": checkpoint_hits,
        "trace_samples": int(trace_pelvis.shape[0]),
        "replay_steps": int(ctrl_trace.shape[0]),
        "stable_trace_fraction": stable_fraction,
        "distance_to_target": distance,
        "best_trace_touch_distance": best_trace_touch,
        "best_touch_distance_after": best_touch,
        "target_radius": TARGET_RADIUS,
        "touch_radius": TOUCH_RADIUS,
        "min_pelvis_z_during_sim": min_pelvis_z,
        "min_torso_up_during_sim": min_torso_up,
        "checkpoint_progress": checkpoint_progress,
        "stability_progress": stability_progress,
        "distance_progress": distance_progress,
        "touch_progress": touch_progress,
        "settle_posture_progress": settle_posture_progress,
        "efficiency_progress": efficiency_progress,
        "progress_score": progress_score,
        "score_method": "weighted_subscores",
        "score_components": score_components,
    }


if __name__ == "__main__":
    final_path = sys.argv[1] if len(sys.argv) > 1 else "/work/final_state.npz"
    ok, info = grade(final_path)
    print(json.dumps({"success": ok, "info": info}))
    sys.exit(0 if ok else 1)
