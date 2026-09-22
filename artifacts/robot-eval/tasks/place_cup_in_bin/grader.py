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
CUP_INIT_POS = (0.48, -0.12, 0.4 + CUP_HALF_HEIGHT)
BIN_CENTER = (0.7, 0.14, 0.4)
BIN_INNER_HALF = (0.05, 0.05)
BIN_WALL_THICKNESS = 0.008
BIN_WALL_HEIGHT = 0.06
BIN_FLOOR_THICKNESS = 0.01
FORWARD_SIM_STEPS = 500
MIN_TRACE_SAMPLES = 20
MIN_LIFT_Z = 0.50
TRACE_POS_TOL = 0.03
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
    cup = world.add_body(name="cup", pos=list(CUP_INIT_POS))
    cup.add_freejoint()
    cup.add_geom(name="cup_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_RADIUS, CUP_HALF_HEIGHT, 0], mass=CUP_MASS, friction=list(CUP_FRICTION), rgba=[0.2, 0.5, 0.9, 1])
    bx, by, bz = BIN_CENTER
    floor_z = bz + BIN_FLOOR_THICKNESS
    wall_z = bz + BIN_WALL_HEIGHT / 2.0
    half_x, half_y = BIN_INNER_HALF
    t = BIN_WALL_THICKNESS
    bin_body = world.add_body(name="bin")
    bin_body.add_geom(name="bin_floor", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[bx, by, floor_z / 2.0], size=[half_x + t, half_y + t, BIN_FLOOR_THICKNESS / 2.0], rgba=[0.2, 0.2, 0.2, 1])
    bin_body.add_geom(name="bin_wall_left", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[bx - half_x - t / 2.0, by, wall_z], size=[t / 2.0, half_y + t, BIN_WALL_HEIGHT / 2.0], rgba=[0.2, 0.2, 0.2, 1])
    bin_body.add_geom(name="bin_wall_right", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[bx + half_x + t / 2.0, by, wall_z], size=[t / 2.0, half_y + t, BIN_WALL_HEIGHT / 2.0], rgba=[0.2, 0.2, 0.2, 1])
    bin_body.add_geom(name="bin_wall_front", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[bx, by - half_y - t / 2.0, wall_z], size=[half_x, t / 2.0, BIN_WALL_HEIGHT / 2.0], rgba=[0.2, 0.2, 0.2, 1])
    bin_body.add_geom(name="bin_wall_back", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[bx, by + half_y + t / 2.0, wall_z], size=[half_x, t / 2.0, BIN_WALL_HEIGHT / 2.0], rgba=[0.2, 0.2, 0.2, 1])
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


def cup_inside_bin(cup_pos):
    bx, by, _ = BIN_CENTER
    dx = abs(float(cup_pos[0]) - bx)
    dy = abs(float(cup_pos[1]) - by)
    return dx <= BIN_INNER_HALF[0] - 0.004 and dy <= BIN_INNER_HALF[1] - 0.004 and float(cup_pos[2]) >= 0.42


def cup_upright(qpos):
    quat = qpos[10:14]
    quat = quat / np.linalg.norm(quat)
    z_axis = np.array([
        2 * (quat[1] * quat[3] + quat[0] * quat[2]),
        2 * (quat[2] * quat[3] - quat[0] * quat[1]),
        1 - 2 * (quat[1] ** 2 + quat[2] ** 2),
    ])
    return abs(float(z_axis[2])) >= 0.85


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
    lifted_in_trace = bool(np.any(trace_cup[:, 2] >= MIN_LIFT_Z))
    best_trace_z = float(np.max(trace_cup[:, 2])) if trace_cup.size else float("-inf")
    late_contact_fraction = float(np.mean(trace_contact[-10:] > 0.5)) if trace_contact.size >= 10 else float(np.mean(trace_contact > 0.5))
    release_contacts = 0
    for _ in range(FORWARD_SIM_STEPS):
        mujoco.mj_step(model, data)
        if has_gripper_cup_contact(model, data, cup_gid, finger_gids):
            release_contacts += 1
    cup_pos = data.xpos[cup_bid].copy()
    bin_xy_error = float(np.linalg.norm(cup_pos[:2] - np.array(BIN_CENTER[:2])))
    quat = data.qpos[10:14]
    quat = quat / np.linalg.norm(quat)
    cup_up_z = float(1 - 2 * (quat[1] ** 2 + quat[2] ** 2))
    axis_verticality = abs(cup_up_z)
    success = (
        trace_cup.shape[0] >= MIN_TRACE_SAMPLES
        and had_grasp
        and lifted_in_trace
        and late_contact_fraction < 0.5
        and cup_inside_bin(cup_pos)
        and cup_upright(data.qpos)
        and (release_contacts / FORWARD_SIM_STEPS) < 0.1
    )
    grasp_progress = 1.0 if had_grasp else 0.0
    lift_progress = clamp01((best_trace_z - CUP_INIT_POS[2]) / (MIN_LIFT_Z - CUP_INIT_POS[2]))
    placement_progress = 1.0 if cup_inside_bin(cup_pos) else clamp01(1.0 - (bin_xy_error / np.linalg.norm(np.array(CUP_INIT_POS[:2]) - np.array(BIN_CENTER[:2]))))
    release_progress = clamp01(1.0 - (release_contacts / FORWARD_SIM_STEPS) / 0.1)
    upright_progress = clamp01(axis_verticality / 0.85)
    inside_bin_progress = 1.0 if cup_inside_bin(cup_pos) else 0.0
    score_components = [
        {"name": "grasp_progress", "value": grasp_progress, "weight": 0.05},
        {"name": "lift_progress", "value": lift_progress, "weight": 0.10},
        {"name": "inside_bin_progress", "value": inside_bin_progress, "weight": 0.40},
        {"name": "placement_progress", "value": placement_progress, "weight": 0.10},
        {"name": "release_progress", "value": release_progress, "weight": 0.10},
        {"name": "upright_progress", "value": upright_progress, "weight": 0.25},
    ]
    progress_score = weighted_subscore(*[(item["name"], item["value"], item["weight"]) for item in score_components])
    replay_steps = int(ctrl_trace.shape[0])
    return success, {
        "cup_xyz_after": [float(x) for x in cup_pos],
        "trace_samples": int(trace_cup.shape[0]),
        "replay_steps": int(ctrl_trace.shape[0]),
        "had_grasp_in_trace": had_grasp,
        "lifted_in_trace": lifted_in_trace,
        "best_trace_z": best_trace_z,
        "lift_height_margin": float(best_trace_z - MIN_LIFT_Z),
        "late_contact_fraction_in_trace": late_contact_fraction,
        "bin_xy_error": bin_xy_error,
        "inside_bin": cup_inside_bin(cup_pos),
        "cup_up_z": cup_up_z,
        "cup_axis_verticality": axis_verticality,
        "upright": cup_upright(data.qpos),
        "contact_fraction_during_sim": release_contacts / FORWARD_SIM_STEPS,
        "grasp_progress": grasp_progress,
        "lift_progress": lift_progress,
        "inside_bin_progress": inside_bin_progress,
        "placement_progress": placement_progress,
        "release_progress": release_progress,
        "upright_progress": upright_progress,
        "progress_score": progress_score,
        "score_method": "weighted_subscores",
        "score_components": score_components,
    }


if __name__ == "__main__":
    final_path = sys.argv[1] if len(sys.argv) > 1 else "/work/final_state.npz"
    ok, info = grade(final_path)
    print(json.dumps({"success": ok, "info": info}))
    sys.exit(0 if ok else 1)
