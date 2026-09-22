import json
import os
import sys
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
PANDA_XML = os.path.join(MENAGERIE, "franka_emika_panda", "panda.xml")
TABLE_CENTER = (0.5, 0.0, 0.2)
TABLE_HALF_SIZE = (0.26, 0.26, 0.2)
BLOCK_HALF = 0.025
RED_INIT_POS = (0.48, -0.09, 0.4 + BLOCK_HALF)
GREEN_INIT_POS = (0.63, 0.08, 0.4 + BLOCK_HALF)
FORWARD_SIM_STEPS = 500
ALIGN_TOL = 0.018
MIN_STACK_DZ = 0.040
MAX_STACK_DZ = 0.070
IDEAL_REPLAY_STEPS = 900
IDEAL_XY_ERR = 0.008
IDEAL_Z_ERR = 0.006
IDEAL_YAW_ALIGNMENT = 0.99


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def arithmetic_progress(*values):
    vals = [clamp01(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def weighted_subscore(*components):
    total_weight = sum(weight for _, _, weight in components)
    if total_weight <= 0:
        return 0.0
    return float(sum(clamp01(value) * weight for _, value, weight in components) / total_weight)


def body_x_axis(data, body_id):
    return data.xmat[body_id].reshape(3, 3)[:, 0]


def build_model():
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.85, 0.85, 0.85, 1])
    table = world.add_body(name="table", pos=list(TABLE_CENTER))
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TABLE_HALF_SIZE), rgba=[0.55, 0.35, 0.2, 1])
    red = world.add_body(name="red_block", pos=list(RED_INIT_POS))
    red.add_freejoint()
    red.add_geom(name="red_block_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BLOCK_HALF, BLOCK_HALF, BLOCK_HALF], mass=0.08, friction=[1.6, 0.4, 0.1], rgba=[0.85, 0.2, 0.2, 1])
    green = world.add_body(name="green_block", pos=list(GREEN_INIT_POS))
    green.add_freejoint()
    green.add_geom(name="green_block_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BLOCK_HALF, BLOCK_HALF, BLOCK_HALF], mass=0.12, friction=[1.8, 0.5, 0.1], rgba=[0.2, 0.8, 0.3, 1])
    return spec.compile()


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
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for ctrl in ctrl_trace:
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    red_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
    green_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_block")
    for _ in range(FORWARD_SIM_STEPS):
        mujoco.mj_step(model, data)

    red_pos = data.xpos[red_bid].copy()
    green_pos = data.xpos[green_bid].copy()
    dx = float(red_pos[0] - green_pos[0])
    dy = float(red_pos[1] - green_pos[1])
    dz = float(red_pos[2] - green_pos[2])

    xy_error = float(np.linalg.norm([dx, dy]))
    aligned = abs(dx) <= ALIGN_TOL and abs(dy) <= ALIGN_TOL
    stacked_height = MIN_STACK_DZ <= dz <= MAX_STACK_DZ
    success = aligned and stacked_height
    replay_steps = int(ctrl_trace.shape[0])
    target_stack_dz = (MIN_STACK_DZ + MAX_STACK_DZ) / 2.0
    z_error_from_target = float(abs(dz - target_stack_dz))
    xy_margin = float(ALIGN_TOL - max(abs(dx), abs(dy)))
    dz_margin = float(min(dz - MIN_STACK_DZ, MAX_STACK_DZ - dz))
    max_xy_err = max(abs(dx), abs(dy))
    if max_xy_err <= IDEAL_XY_ERR:
        xy_progress = 1.0
    else:
        xy_progress = clamp01(1.0 - ((max_xy_err - IDEAL_XY_ERR) / (ALIGN_TOL - IDEAL_XY_ERR)))
    dz_half_range = (MAX_STACK_DZ - MIN_STACK_DZ) / 2.0
    if z_error_from_target <= IDEAL_Z_ERR:
        z_progress = 1.0
    else:
        z_progress = clamp01(1.0 - ((z_error_from_target - IDEAL_Z_ERR) / (dz_half_range - IDEAL_Z_ERR)))
    red_x_axis = body_x_axis(data, red_bid)
    green_x_axis = body_x_axis(data, green_bid)
    red_xy = red_x_axis[:2]
    green_xy = green_x_axis[:2]
    red_xy_norm = max(float(np.linalg.norm(red_xy)), 1e-8)
    green_xy_norm = max(float(np.linalg.norm(green_xy)), 1e-8)
    yaw_alignment = float(abs(np.dot(red_xy, green_xy) / (red_xy_norm * green_xy_norm)))
    rotation_progress = 1.0 if yaw_alignment >= IDEAL_YAW_ALIGNMENT else clamp01((yaw_alignment - 0.90) / (IDEAL_YAW_ALIGNMENT - 0.90))
    efficiency_progress = clamp01(7000 / max(replay_steps, 1))
    completion_progress = 1.0 if success else 0.0
    score_components = [
        {"name": "completion_progress", "value": completion_progress, "weight": 0.45},
        {"name": "xy_progress", "value": xy_progress, "weight": 0.25},
        {"name": "z_progress", "value": z_progress, "weight": 0.18},
        {"name": "rotation_progress", "value": rotation_progress, "weight": 0.10},
        {"name": "efficiency_progress", "value": efficiency_progress, "weight": 0.02},
    ]
    progress_score = weighted_subscore(*[(item["name"], item["value"], item["weight"]) for item in score_components])
    return success, {
        "red_xyz_after": [float(x) for x in red_pos],
        "green_xyz_after": [float(x) for x in green_pos],
        "delta_xyz": [dx, dy, dz],
        "xy_error": xy_error,
        "xy_margin": xy_margin,
        "target_stack_dz": target_stack_dz,
        "z_error_from_target": z_error_from_target,
        "dz_margin": dz_margin,
        "yaw_alignment": yaw_alignment,
        "completion_progress": completion_progress,
        "xy_progress": xy_progress,
        "z_progress": z_progress,
        "rotation_progress": rotation_progress,
        "efficiency_progress": efficiency_progress,
        "progress_score": progress_score,
        "score_method": "weighted_subscores",
        "score_components": score_components,
        "aligned": aligned,
        "stacked_height": stacked_height,
        "replay_steps": replay_steps,
        "alignment_tolerance": ALIGN_TOL,
        "stack_height_range": [MIN_STACK_DZ, MAX_STACK_DZ],
    }


if __name__ == "__main__":
    final_path = sys.argv[1] if len(sys.argv) > 1 else "/work/final_state.npz"
    ok, info = grade(final_path)
    print(json.dumps({"success": ok, "info": info}))
    sys.exit(0 if ok else 1)
