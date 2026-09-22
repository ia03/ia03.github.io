import json
import os
import sys
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
PANDA_XML = os.path.join(MENAGERIE, "franka_emika_panda", "panda.xml")
TABLE_CENTER = (0.5, 0.0, 0.2)
TABLE_HALF_SIZE = (0.30, 0.26, 0.2)
PEG_HALF = (0.03, 0.012, 0.012)
PEG_INIT_POS = (0.47, -0.102, 0.508 + PEG_HALF[2])
BOARD_CENTER = (0.52, -0.10, 0.52)
BOARD_SIZE = (0.02, 0.10, 0.10)
SLOT_HALF_HEIGHT = 0.025
SLOT_HALF_WIDTH = 0.040
GUIDE_CENTER = (0.49, -0.102, 0.50)
GUIDE_HALF = (0.08, 0.03, 0.008)
FORWARD_SIM_STEPS = 500
IDEAL_REPLAY_STEPS = 4500
SUCCESS_X = 0.515


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def staged_progress(*values):
    vals = [clamp01(v) for v in values]
    if not vals:
        return 0.0
    total = 0.0
    prefix = 1.0
    for value in vals:
        prefix *= value
        total += prefix
    return float(total / len(vals))


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
    guide = world.add_body(name="guide")
    guide.add_geom(name="guide_shelf", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(GUIDE_CENTER), size=list(GUIDE_HALF), rgba=[0.45, 0.45, 0.5, 1])
    peg = world.add_body(name="peg", pos=list(PEG_INIT_POS))
    peg.add_freejoint()
    peg.add_geom(name="peg_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(PEG_HALF), mass=0.07, friction=[2.0, 0.5, 0.1], rgba=[0.85, 0.55, 0.2, 1])
    peg.add_geom(
        name="peg_tab",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.050, 0.0, 0.0],
        size=[0.018, 0.030, 0.030],
        mass=0.02,
        friction=[2.0, 0.5, 0.1],
        rgba=[0.95, 0.75, 0.25, 1],
    )
    board = world.add_body(name="board")
    top_inner_z = BOARD_CENTER[2] + SLOT_HALF_HEIGHT
    top_outer_z = BOARD_CENTER[2] + BOARD_SIZE[2]
    bottom_inner_z = BOARD_CENTER[2] - SLOT_HALF_HEIGHT
    bottom_outer_z = BOARD_CENTER[2] - BOARD_SIZE[2]
    side_inner_y = BOARD_CENTER[1] + SLOT_HALF_WIDTH
    side_outer_y = BOARD_CENTER[1] + BOARD_SIZE[1]
    board.add_geom(
        name="board_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[BOARD_CENTER[0], BOARD_CENTER[1], 0.5 * (top_inner_z + top_outer_z)],
        size=[BOARD_SIZE[0], BOARD_SIZE[1], 0.5 * (top_outer_z - top_inner_z)],
        rgba=[0.25, 0.25, 0.25, 1],
    )
    board.add_geom(
        name="board_bottom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[BOARD_CENTER[0], BOARD_CENTER[1], 0.5 * (bottom_inner_z + bottom_outer_z)],
        size=[BOARD_SIZE[0], BOARD_SIZE[1], 0.5 * (bottom_inner_z - bottom_outer_z)],
        rgba=[0.25, 0.25, 0.25, 1],
    )
    board.add_geom(
        name="board_side_pos",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[BOARD_CENTER[0], 0.5 * (side_inner_y + side_outer_y), BOARD_CENTER[2]],
        size=[BOARD_SIZE[0], 0.5 * (side_outer_y - side_inner_y), SLOT_HALF_HEIGHT],
        rgba=[0.25, 0.25, 0.25, 1],
    )
    board.add_geom(
        name="board_side_neg",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[BOARD_CENTER[0], BOARD_CENTER[1] - 0.5 * (BOARD_SIZE[1] + SLOT_HALF_WIDTH), BOARD_CENTER[2]],
        size=[BOARD_SIZE[0], 0.5 * (side_outer_y - side_inner_y), SLOT_HALF_HEIGHT],
        rgba=[0.25, 0.25, 0.25, 1],
    )
    return spec.compile()


def quat_to_x_axis(quat):
    quat = quat / np.linalg.norm(quat)
    return np.array([
        1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
        2 * (quat[1] * quat[2] + quat[0] * quat[3]),
        2 * (quat[1] * quat[3] - quat[0] * quat[2]),
    ])


def peg_inserted(pos):
    x_ok = float(pos[0]) >= SUCCESS_X
    y_ok = abs(float(pos[1]) - BOARD_CENTER[1]) <= 0.025
    z_ok = abs(float(pos[2]) - BOARD_CENTER[2]) <= 0.020
    return x_ok and y_ok and z_ok


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

    peg_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    for _ in range(FORWARD_SIM_STEPS):
        mujoco.mj_step(model, data)
    peg_pos = data.xpos[peg_bid].copy()
    peg_x_axis = quat_to_x_axis(data.qpos[10:14])
    aligned = abs(float(peg_x_axis[0])) >= 0.45
    inserted = peg_inserted(peg_pos)
    insertion_depth = float(peg_pos[0] - SUCCESS_X)
    y_error = float(abs(peg_pos[1] - BOARD_CENTER[1]))
    z_error = float(abs(peg_pos[2] - BOARD_CENTER[2]))
    x_alignment = float(abs(peg_x_axis[0]))
    success = inserted and aligned
    insertion_progress = 1.0 if float(peg_pos[0]) >= SUCCESS_X else clamp01((peg_pos[0] - PEG_INIT_POS[0]) / (SUCCESS_X - PEG_INIT_POS[0]))
    y_progress = 1.0 if y_error <= 0.025 else clamp01(1.0 - (y_error - 0.025) / 0.025)
    z_progress = 1.0 if z_error <= 0.020 else clamp01(1.0 - (z_error - 0.020) / 0.020)
    alignment_progress = 1.0 if x_alignment >= 0.45 else clamp01(x_alignment / 0.45)
    replay_steps = int(ctrl_trace.shape[0])
    efficiency_progress = clamp01(IDEAL_REPLAY_STEPS / max(replay_steps, 1))
    score_components = [
        {"name": "insertion_progress", "value": insertion_progress, "weight": 0.45},
        {"name": "y_progress", "value": y_progress, "weight": 0.15},
        {"name": "z_progress", "value": z_progress, "weight": 0.15},
        {"name": "alignment_progress", "value": alignment_progress, "weight": 0.20},
        {"name": "efficiency_progress", "value": efficiency_progress, "weight": 0.05},
    ]
    progress_score = weighted_subscore(*[(item["name"], item["value"], item["weight"]) for item in score_components])
    return success, {
        "peg_xyz_after": [float(x) for x in peg_pos],
        "peg_x_axis": [float(x) for x in peg_x_axis],
        "insertion_depth": insertion_depth,
        "y_error": y_error,
        "z_error": z_error,
        "x_axis_alignment": x_alignment,
        "inserted": inserted,
        "aligned": aligned,
        "insertion_progress": insertion_progress,
        "y_progress": y_progress,
        "z_progress": z_progress,
        "alignment_progress": alignment_progress,
        "efficiency_progress": efficiency_progress,
        "progress_score": progress_score,
        "score_method": "weighted_subscores",
        "score_components": score_components,
        "replay_steps": replay_steps,
    }


if __name__ == "__main__":
    final_path = sys.argv[1] if len(sys.argv) > 1 else "/work/final_state.npz"
    ok, info = grade(final_path)
    print(json.dumps({"success": ok, "info": info}))
    sys.exit(0 if ok else 1)
