import json
import os
import sys
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
H1_SCENE_XML = os.path.join(MENAGERIE, "unitree_h1", "scene.xml")
BOX_SIZE = (0.08, 0.05, 0.07)
BOX_MASS = 0.6
BOX_INIT_POS = (0.14, 0.0, 0.88)
TARGET_XY = np.array([0.25, 0.0])
TARGET_RADIUS = 0.20
MIN_PELVIS_Z = 0.74
MIN_TORSO_UP = 0.50
MIN_BOX_Z = 0.72
MAX_BOX_TO_PELVIS = 0.42
FORWARD_SIM_STEPS = 500
START_RADIUS = 0.15
PROGRESS_CHECKPOINTS = (0.05, 0.09, 0.13)
MIN_TRACE_SAMPLES = 30
MIN_STABLE_TRACE_FRACTION = 0.90
TRACE_POS_TOL = 0.05
TRACE_SCALAR_TOL = 0.03
IDEAL_REPLAY_STEPS = 2600


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def target_progress(distance, radius, baseline):
    if distance <= radius:
        return 1.0
    return clamp01(1.0 - (distance - radius) / max(baseline - radius, 1e-6))


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
    pedestal = world.add_body(name="pedestal")
    pedestal.add_geom(
        name="pedestal_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[BOX_INIT_POS[0], BOX_INIT_POS[1], 0.40],
        size=[0.12, 0.10, 0.40],
        rgba=[0.45, 0.45, 0.5, 1],
    )
    box = world.add_body(name="carry_box", pos=list(BOX_INIT_POS))
    box.add_freejoint()
    box.add_geom(
        name="carry_box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(BOX_SIZE),
        mass=BOX_MASS,
        friction=[1.3, 0.4, 0.1],
        rgba=[0.85, 0.65, 0.15, 1],
    )
    world.add_geom(
        name="target_marker",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[float(TARGET_XY[0]), float(TARGET_XY[1]), 0.005],
        size=[TARGET_RADIUS, 0.005, 0],
        contype=0,
        conaffinity=0,
        rgba=[0.15, 0.7, 0.25, 0.30],
    )
    return spec.compile()


def reset_canonical(model, data):
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    else:
        mujoco.mj_resetData(model, data)
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "carry_box")
    box_joint_id = int(model.body_jntadr[box_body_id])
    box_qpos_adr = model.jnt_qposadr[box_joint_id]
    data.qpos[box_qpos_adr:box_qpos_adr + 3] = BOX_INIT_POS
    data.qpos[box_qpos_adr + 3:box_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    box_qvel_adr = model.jnt_dofadr[box_joint_id]
    data.qvel[box_qvel_adr:box_qvel_adr + 6] = 0.0
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
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "carry_box")
    trace_pelvis = []
    trace_torso_up = []
    trace_box = []
    trace_box_distance_to_pelvis = []
    for ctrl in ctrl_trace:
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        trace_pelvis.append(data.xpos[pelvis_id].copy())
        trace_torso_up.append(float(data.xmat[torso_id].reshape(3, 3)[2, 2]))
        trace_box.append(data.xpos[box_id].copy())
        trace_box_distance_to_pelvis.append(float(np.linalg.norm(data.xpos[box_id] - data.xpos[pelvis_id])))
    trace_pelvis = np.array(trace_pelvis, dtype=float).reshape(-1, 3)
    trace_torso_up = np.array(trace_torso_up, dtype=float)
    trace_box = np.array(trace_box, dtype=float).reshape(-1, 3)
    trace_box_distance_to_pelvis = np.array(trace_box_distance_to_pelvis, dtype=float)
    start_distance = float(np.linalg.norm(trace_pelvis[0, :2])) if trace_pelvis.size else float("inf")
    checkpoint_hits = [bool(np.any(trace_pelvis[:, 0] >= checkpoint)) for checkpoint in PROGRESS_CHECKPOINTS] if trace_pelvis.size else [False] * len(PROGRESS_CHECKPOINTS)
    stable_mask = (
        (trace_pelvis[:, 2] >= MIN_PELVIS_Z)
        & (trace_torso_up >= MIN_TORSO_UP)
        & (trace_box[:, 2] >= MIN_BOX_Z)
        & (trace_box_distance_to_pelvis <= MAX_BOX_TO_PELVIS)
    ) if trace_pelvis.size else np.array([], dtype=bool)
    stable_fraction = float(np.mean(stable_mask)) if stable_mask.size else 0.0
    box_checkpoint_hits = [bool(np.any(trace_box[:, 0] >= checkpoint - 0.05)) for checkpoint in PROGRESS_CHECKPOINTS] if trace_box.size else [False] * len(PROGRESS_CHECKPOINTS)

    min_pelvis_z = float(data.xpos[pelvis_id][2])
    min_torso_up = float(data.xmat[torso_id].reshape(3, 3)[2, 2])
    min_box_z = float(data.xpos[box_id][2])
    max_box_to_pelvis = float(np.linalg.norm(data.xpos[box_id] - data.xpos[pelvis_id]))
    for _ in range(FORWARD_SIM_STEPS):
        mujoco.mj_step(model, data)
        min_pelvis_z = min(min_pelvis_z, float(data.xpos[pelvis_id][2]))
        min_torso_up = min(min_torso_up, float(data.xmat[torso_id].reshape(3, 3)[2, 2]))
        min_box_z = min(min_box_z, float(data.xpos[box_id][2]))
        max_box_to_pelvis = max(max_box_to_pelvis, float(np.linalg.norm(data.xpos[box_id] - data.xpos[pelvis_id])))

    pelvis_pos = data.xpos[pelvis_id].copy()
    box_pos = data.xpos[box_id].copy()
    pelvis_dist = float(np.linalg.norm(pelvis_pos[:2] - TARGET_XY))
    box_dist = float(np.linalg.norm(box_pos[:2] - TARGET_XY))
    success = (
        trace_pelvis.shape[0] >= MIN_TRACE_SAMPLES
        and start_distance <= START_RADIUS
        and all(checkpoint_hits)
        and all(box_checkpoint_hits)
        and stable_fraction >= MIN_STABLE_TRACE_FRACTION
        and pelvis_dist <= TARGET_RADIUS
        and box_dist <= TARGET_RADIUS
        and min_pelvis_z >= MIN_PELVIS_Z
        and min_torso_up >= MIN_TORSO_UP
        and min_box_z >= MIN_BOX_Z
        and max_box_to_pelvis <= MAX_BOX_TO_PELVIS
    )
    pelvis_checkpoint_progress = sum(checkpoint_hits) / len(PROGRESS_CHECKPOINTS)
    box_checkpoint_progress = sum(box_checkpoint_hits) / len(PROGRESS_CHECKPOINTS)
    stability_progress = clamp01(stable_fraction / MIN_STABLE_TRACE_FRACTION)
    pelvis_distance_progress = target_progress(
        pelvis_dist,
        TARGET_RADIUS,
        float(np.linalg.norm(TARGET_XY)),
    )
    box_distance_progress = target_progress(
        box_dist,
        TARGET_RADIUS,
        float(np.linalg.norm(TARGET_XY - np.array([BOX_INIT_POS[0], BOX_INIT_POS[1]]))),
    )
    carry_proximity_progress = clamp01(1.0 - max(0.0, max_box_to_pelvis - MAX_BOX_TO_PELVIS) / MAX_BOX_TO_PELVIS)
    replay_steps = int(ctrl_trace.shape[0])
    efficiency_progress = clamp01(IDEAL_REPLAY_STEPS / max(replay_steps, 1))
    box_height_progress = clamp01(min_box_z / MIN_BOX_Z)
    score_components = [
        {"name": "box_distance_progress", "value": box_distance_progress, "weight": 0.15},
        {"name": "carry_proximity_progress", "value": carry_proximity_progress, "weight": 0.30},
        {"name": "box_height_progress", "value": box_height_progress, "weight": 0.25},
        {"name": "stability_progress", "value": stability_progress, "weight": 0.10},
        {"name": "box_checkpoint_progress", "value": box_checkpoint_progress, "weight": 0.03},
        {"name": "pelvis_distance_progress", "value": pelvis_distance_progress, "weight": 0.02},
        {"name": "pelvis_checkpoint_progress", "value": pelvis_checkpoint_progress, "weight": 0.00},
        {"name": "efficiency_progress", "value": efficiency_progress, "weight": 0.15},
    ]
    progress_score = weighted_subscore(*[(item["name"], item["value"], item["weight"]) for item in score_components])
    return success, {
        "pelvis_xyz_after": [float(x) for x in pelvis_pos],
        "box_xyz_after": [float(x) for x in box_pos],
        "start_distance_from_origin": start_distance,
        "checkpoint_hits": checkpoint_hits,
        "box_checkpoint_hits": box_checkpoint_hits,
        "trace_samples": int(trace_pelvis.shape[0]),
        "replay_steps": int(ctrl_trace.shape[0]),
        "stable_trace_fraction": stable_fraction,
        "pelvis_distance_to_target": pelvis_dist,
        "box_distance_to_target": box_dist,
        "target_radius": TARGET_RADIUS,
        "min_pelvis_z_during_sim": min_pelvis_z,
        "min_torso_up_during_sim": min_torso_up,
        "min_box_z_during_sim": min_box_z,
        "max_box_to_pelvis_during_sim": max_box_to_pelvis,
        "pelvis_checkpoint_progress": pelvis_checkpoint_progress,
        "box_checkpoint_progress": box_checkpoint_progress,
        "stability_progress": stability_progress,
        "pelvis_distance_progress": pelvis_distance_progress,
        "box_distance_progress": box_distance_progress,
        "carry_proximity_progress": carry_proximity_progress,
        "box_height_progress": box_height_progress,
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
