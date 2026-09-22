import json
import os
import sys
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
PANDA_XML = os.path.join(MENAGERIE, "franka_emika_panda", "panda.xml")
TABLE_CENTER = (0.5, 0.0, 0.2)
TABLE_HALF_SIZE = (0.26, 0.26, 0.2)
CUP_RADIUS = 0.022
CUP_HALF_HEIGHT = 0.035
CUP_MASS = 0.05
CUP_FRICTION = (2.0, 0.5, 0.1)
CUP_INIT_POS = (0.55, 0.15, 0.4 + CUP_HALF_HEIGHT)
WALL_CENTER = (0.49, 0.0, 0.48)
WALL_SIZE = (0.015, 0.11, 0.045)
SUCCESS_CUP_Z = 0.56
SUCCESS_CUP_X_MAX = 0.42
FORWARD_SIM_STEPS = 500
MIN_TRACE_SAMPLES = 20
RETURN_X_CHECKPOINT = 0.48
TRACE_POS_TOL = 0.03
IDEAL_REPLAY_STEPS = 2600


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
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.85, 0.85, 0.85, 1])
    table = world.add_body(name="table", pos=list(TABLE_CENTER))
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TABLE_HALF_SIZE), rgba=[0.55, 0.35, 0.2, 1])
    wall = world.add_body(name="wall")
    wall.add_geom(name="wall_geom", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(WALL_CENTER), size=list(WALL_SIZE), rgba=[0.8, 0.8, 0.85, 1])
    cup = world.add_body(name="cup", pos=list(CUP_INIT_POS))
    cup.add_freejoint()
    cup.add_geom(name="cup_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_RADIUS, CUP_HALF_HEIGHT, 0], mass=CUP_MASS, friction=list(CUP_FRICTION), rgba=[0.2, 0.5, 0.9, 1])
    return spec.compile()


def gripper_geom_ids(model):
    ids = set()
    for body_name in ("left_finger", "right_finger"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) == body_id:
                ids.add(geom_id)
    return ids


def has_gripper_cup_contact(model, data, cup_gid, finger_gids):
    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        if cup_gid in (g1, g2) and (g1 in finger_gids or g2 in finger_gids):
            return True
    return False


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
    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    cup_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")
    finger_gids = gripper_geom_ids(model)
    trace_cup = []
    trace_contact = []
    for ctrl in ctrl_trace:
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        trace_cup.append(data.xpos[cup_bid].copy())
        trace_contact.append(float(has_gripper_cup_contact(model, data, cup_gid, finger_gids)))
    trace_cup = np.array(trace_cup, dtype=float).reshape(-1, 3)
    trace_contact = np.array(trace_contact, dtype=float)
    had_grasp = bool(np.any(trace_contact > 0.5))
    returned_past_wall = bool(np.any(trace_cup[:, 0] <= RETURN_X_CHECKPOINT))
    lifted_in_trace = bool(np.any(trace_cup[:, 2] >= SUCCESS_CUP_Z))
    best_trace_x = float(np.min(trace_cup[:, 0])) if trace_cup.size else float("inf")
    best_trace_z = float(np.max(trace_cup[:, 2])) if trace_cup.size else float("-inf")
    min_z = float(data.xpos[cup_bid][2])
    contacts = 0
    for _ in range(FORWARD_SIM_STEPS):
        mujoco.mj_step(model, data)
        cup_z = float(data.xpos[cup_bid][2])
        min_z = min(min_z, cup_z)
        if has_gripper_cup_contact(model, data, cup_gid, finger_gids):
            contacts += 1
    cup_pos = data.xpos[cup_bid].copy()
    success = (
        trace_cup.shape[0] >= MIN_TRACE_SAMPLES
        and had_grasp
        and returned_past_wall
        and lifted_in_trace
        and float(cup_pos[0]) <= SUCCESS_CUP_X_MAX
        and float(cup_pos[2]) >= SUCCESS_CUP_Z
        and (contacts / FORWARD_SIM_STEPS) >= 0.5
    )
    contact_fraction = contacts / FORWARD_SIM_STEPS
    grasp_progress = 1.0 if had_grasp else 0.0
    return_progress = clamp01((CUP_INIT_POS[0] - best_trace_x) / (CUP_INIT_POS[0] - RETURN_X_CHECKPOINT))
    lift_progress = clamp01((best_trace_z - CUP_INIT_POS[2]) / (SUCCESS_CUP_Z - CUP_INIT_POS[2]))
    settle_contact_progress = clamp01(contact_fraction / 0.5)
    final_return_progress = clamp01((CUP_INIT_POS[0] - float(cup_pos[0])) / (CUP_INIT_POS[0] - SUCCESS_CUP_X_MAX))
    final_height_progress = clamp01((float(cup_pos[2]) - CUP_INIT_POS[2]) / (SUCCESS_CUP_Z - CUP_INIT_POS[2]))
    replay_steps = int(ctrl_trace.shape[0])
    efficiency_progress = clamp01(IDEAL_REPLAY_STEPS / max(replay_steps, 1))
    score_components = [
        {"name": "grasp_progress", "value": grasp_progress, "weight": 0.05},
        {"name": "return_progress", "value": return_progress, "weight": 0.15},
        {"name": "lift_progress", "value": lift_progress, "weight": 0.10},
        {"name": "final_return_progress", "value": final_return_progress, "weight": 0.10},
        {"name": "final_height_progress", "value": final_height_progress, "weight": 0.25},
        {"name": "settle_contact_progress", "value": settle_contact_progress, "weight": 0.25},
        {"name": "efficiency_progress", "value": efficiency_progress, "weight": 0.10},
    ]
    progress_score = weighted_subscore(*[(item["name"], item["value"], item["weight"]) for item in score_components])
    return success, {
        "cup_xyz_after": [float(x) for x in cup_pos],
        "trace_samples": int(trace_cup.shape[0]),
        "replay_steps": int(ctrl_trace.shape[0]),
        "had_grasp_in_trace": had_grasp,
        "returned_past_wall_in_trace": returned_past_wall,
        "lifted_in_trace": lifted_in_trace,
        "best_trace_x": best_trace_x,
        "best_trace_z": best_trace_z,
        "x_return_margin": float(RETURN_X_CHECKPOINT - best_trace_x),
        "lift_height_margin": float(best_trace_z - SUCCESS_CUP_Z),
        "min_cup_z_during_sim": min_z,
        "contact_fraction_during_sim": contact_fraction,
        "grasp_progress": grasp_progress,
        "return_progress": return_progress,
        "lift_progress": lift_progress,
        "final_return_progress": final_return_progress,
        "final_height_progress": final_height_progress,
        "settle_contact_progress": settle_contact_progress,
        "efficiency_progress": efficiency_progress,
        "progress_score": progress_score,
        "score_method": "weighted_subscores",
        "score_components": score_components,
        "threshold": SUCCESS_CUP_Z,
        "x_threshold": SUCCESS_CUP_X_MAX,
    }


if __name__ == "__main__":
    final_path = sys.argv[1] if len(sys.argv) > 1 else "/work/final_state.npz"
    ok, info = grade(final_path)
    print(json.dumps({"success": ok, "info": info}))
    sys.exit(0 if ok else 1)
