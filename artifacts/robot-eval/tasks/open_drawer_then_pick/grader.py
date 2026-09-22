import json
import os
import sys
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
PANDA_XML = os.path.join(MENAGERIE, "franka_emika_panda", "panda.xml")
TABLE_CENTER = (0.5, 0.0, 0.2)
TABLE_HALF_SIZE = (0.30, 0.28, 0.2)
CABINET_POS = (0.58, -0.02, 0.44)
CABINET_HALF = (0.09, 0.12, 0.05)
DRAWER_HALF = (0.075, 0.105, 0.03)
DRAWER_RANGE = (0.0, 0.16)
BLOCK_HALF_X = 0.028
BLOCK_HALF_Y = 0.028
BLOCK_HALF_Z = 0.03
BLOCK_INIT_POS = (0.645, 0.0, 0.4 + BLOCK_HALF_Z)
DRAWER_SUCCESS = 0.05
BLOCK_RETRIEVAL_Z = 0.50
BLOCK_SUCCESS_Z = 0.595
FORWARD_SIM_STEPS = 500
MIN_TRACE_SAMPLES = 20
TRACE_POS_TOL = 0.03
TRACE_SCALAR_TOL = 0.02
DRAWER_HANDLE_HALF = (0.012, 0.05, 0.012)
DRAWER_HANDLE_X = DRAWER_HALF[0] + 0.025


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
    cabinet = world.add_body(name="cabinet", pos=list(CABINET_POS))
    cabinet.add_geom(name="cabinet_left", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -CABINET_HALF[1], 0], size=[CABINET_HALF[0], 0.01, CABINET_HALF[2]], rgba=[0.45, 0.45, 0.5, 1])
    cabinet.add_geom(name="cabinet_right", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, CABINET_HALF[1], 0], size=[CABINET_HALF[0], 0.01, CABINET_HALF[2]], rgba=[0.45, 0.45, 0.5, 1])
    cabinet.add_geom(name="cabinet_back", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-CABINET_HALF[0], 0, 0], size=[0.01, CABINET_HALF[1], CABINET_HALF[2]], rgba=[0.45, 0.45, 0.5, 1])
    cabinet.add_geom(name="cabinet_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -CABINET_HALF[2]], size=[CABINET_HALF[0], CABINET_HALF[1], 0.01], rgba=[0.45, 0.45, 0.5, 1])
    drawer = cabinet.add_body(name="drawer")
    drawer.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=list(DRAWER_RANGE), damping=2.0)
    cabinet.add_geom(name="cabinet_top", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, CABINET_HALF[2]], size=[CABINET_HALF[0], CABINET_HALF[1], 0.01], rgba=[0.45, 0.45, 0.5, 1])
    drawer.add_geom(name="drawer_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -DRAWER_HALF[2] + 0.008], size=[DRAWER_HALF[0], DRAWER_HALF[1], 0.008], rgba=[0.72, 0.52, 0.28, 1])
    drawer.add_geom(name="drawer_back", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-DRAWER_HALF[0] + 0.01, 0, 0], size=[0.01, DRAWER_HALF[1], DRAWER_HALF[2]], rgba=[0.72, 0.52, 0.28, 1])
    drawer.add_geom(name="drawer_left", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -DRAWER_HALF[1] + 0.01, 0], size=[DRAWER_HALF[0], 0.01, DRAWER_HALF[2]], rgba=[0.72, 0.52, 0.28, 1])
    drawer.add_geom(name="drawer_right", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, DRAWER_HALF[1] - 0.01, 0], size=[DRAWER_HALF[0], 0.01, DRAWER_HALF[2]], rgba=[0.72, 0.52, 0.28, 1])
    drawer.add_geom(name="drawer_front", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[DRAWER_HALF[0], 0, 0], size=[0.01, DRAWER_HALF[1], DRAWER_HALF[2]], rgba=[0.75, 0.55, 0.3, 1])
    drawer.add_geom(name="drawer_handle", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[DRAWER_HANDLE_X, 0, 0.0], size=list(DRAWER_HANDLE_HALF), rgba=[0.15, 0.15, 0.15, 1])
    spec.add_actuator(
        name="drawer_motor",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target="drawer_slide",
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        biastype=mujoco.mjtBias.mjBIAS_NONE,
        ctrllimited=True,
        ctrlrange=[-1.0, 1.0],
    )
    block = world.add_body(name="block", pos=list(BLOCK_INIT_POS))
    block.add_freejoint()
    block.add_geom(
        name="block_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[BLOCK_HALF_X, BLOCK_HALF_Y, BLOCK_HALF_Z],
        mass=0.03,
        friction=[1.8, 0.5, 0.1],
        rgba=[0.85, 0.2, 0.2, 1],
    )
    return spec.compile()


def gripper_geom_ids(model):
    ids = set()
    for body_name in ("left_finger", "right_finger"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) == body_id:
                ids.add(geom_id)
    return ids


def has_gripper_block_contact(model, data, block_gid, finger_gids):
    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        if block_gid in (g1, g2) and (g1 in finger_gids or g2 in finger_gids):
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

    drawer_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    drawer_qpos_adr = model.jnt_qposadr[drawer_joint_id]
    block_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    block_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom")
    finger_gids = gripper_geom_ids(model)
    trace_drawer_open = []
    trace_block = []
    trace_contact = []
    for ctrl in ctrl_trace:
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        trace_drawer_open.append(float(data.qpos[drawer_qpos_adr]))
        trace_block.append(data.xpos[block_bid].copy())
        trace_contact.append(float(has_gripper_block_contact(model, data, block_gid, finger_gids)))
    trace_drawer_open = np.array(trace_drawer_open, dtype=float)
    trace_block = np.array(trace_block, dtype=float).reshape(-1, 3)
    trace_contact = np.array(trace_contact, dtype=float)
    max_drawer_open = float(np.max(trace_drawer_open)) if trace_drawer_open.size else 0.0
    max_block_z = float(np.max(trace_block[:, 2])) if trace_block.size else float("-inf")
    open_indices = np.flatnonzero(trace_drawer_open >= DRAWER_SUCCESS)
    lift_indices = np.flatnonzero(trace_block[:, 2] >= BLOCK_RETRIEVAL_Z)
    opened_before_lift = open_indices.size > 0 and lift_indices.size > 0 and int(open_indices[0]) <= int(lift_indices[0])

    contact_frames = 0
    for _ in range(FORWARD_SIM_STEPS):
        mujoco.mj_step(model, data)
        if has_gripper_block_contact(model, data, block_gid, finger_gids):
            contact_frames += 1

    drawer_open = float(data.qpos[drawer_qpos_adr])
    block_pos = data.xpos[block_bid].copy()
    success = (
        trace_block.shape[0] >= MIN_TRACE_SAMPLES
        and opened_before_lift
        and drawer_open >= DRAWER_SUCCESS
        and float(block_pos[2]) >= BLOCK_SUCCESS_Z
        and (contact_frames / FORWARD_SIM_STEPS) >= 0.4
    )
    contact_fraction = contact_frames / FORWARD_SIM_STEPS
    drawer_progress = clamp01(max_drawer_open / DRAWER_SUCCESS)
    retrieval_progress = clamp01((max_block_z - BLOCK_INIT_POS[2]) / (BLOCK_RETRIEVAL_Z - BLOCK_INIT_POS[2]))
    final_height_progress = clamp01((float(block_pos[2]) - BLOCK_INIT_POS[2]) / (BLOCK_SUCCESS_Z - BLOCK_INIT_POS[2]))
    hold_progress = clamp01(contact_fraction / 0.4)
    order_progress = 1.0 if opened_before_lift else 0.0
    ordered_retrieval_progress = order_progress * retrieval_progress
    final_drawer_progress = clamp01(drawer_open / DRAWER_SUCCESS)
    score_components = [
        {"name": "drawer_progress", "value": drawer_progress, "weight": 0.22},
        {"name": "ordered_retrieval_progress", "value": ordered_retrieval_progress, "weight": 0.28},
        {"name": "final_height_progress", "value": final_height_progress, "weight": 0.22},
        {"name": "hold_progress", "value": hold_progress, "weight": 0.18},
        {"name": "final_drawer_progress", "value": final_drawer_progress, "weight": 0.10},
    ]
    progress_score = weighted_subscore(*[(item["name"], item["value"], item["weight"]) for item in score_components])
    replay_steps = int(ctrl_trace.shape[0])
    return success, {
        "drawer_open_after": drawer_open,
        "block_xyz_after": [float(x) for x in block_pos],
        "trace_samples": int(trace_block.shape[0]),
        "replay_steps": int(ctrl_trace.shape[0]),
        "opened_before_lift": opened_before_lift,
        "max_drawer_open_in_trace": max_drawer_open,
        "max_block_z_in_trace": max_block_z,
        "drawer_open_margin": float(max_drawer_open - DRAWER_SUCCESS),
        "retrieval_height_margin": float(max_block_z - BLOCK_RETRIEVAL_Z),
        "final_height_margin": float(float(block_pos[2]) - BLOCK_SUCCESS_Z),
        "contact_fraction_during_sim": contact_fraction,
        "drawer_threshold": DRAWER_SUCCESS,
        "retrieval_threshold_z": BLOCK_RETRIEVAL_Z,
        "block_threshold_z": BLOCK_SUCCESS_Z,
        "drawer_progress": drawer_progress,
        "retrieval_progress": retrieval_progress,
        "ordered_retrieval_progress": ordered_retrieval_progress,
        "final_height_progress": final_height_progress,
        "hold_progress": hold_progress,
        "order_progress": order_progress,
        "final_drawer_progress": final_drawer_progress,
        "progress_score": progress_score,
        "score_method": "weighted_subscores",
        "score_components": score_components,
    }


if __name__ == "__main__":
    final_path = sys.argv[1] if len(sys.argv) > 1 else "/work/final_state.npz"
    ok, info = grade(final_path)
    print(json.dumps({"success": ok, "info": info}))
    sys.exit(0 if ok else 1)
