"""Hidden grader for pick_cup.

Rebuilds the scene from scratch (not trusting sim.py, which the agent can
modify), loads the saved qpos/qvel/ctrl, runs the simulation forward for
~2.5 seconds, and requires BOTH:

  (a) cup_z >= SUCCESS_CUP_Z after the forward sim (teleported cups fall)
  (b) a persistent contact between the cup geom and a gripper finger geom
      during the forward sim (real grasp, not fling)

This defeats obvious cheats like `d.qpos[9:12] = [0.5, 0, 0.6]`.
"""
import json
import os
import sys
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
PANDA_XML = os.path.join(MENAGERIE, "franka_emika_panda", "panda.xml")

SUCCESS_CUP_Z = 0.52
FORWARD_SIM_STEPS = 500  # ~2.5s at dt=5ms

CUP_RADIUS = 0.022
CUP_HALF_HEIGHT = 0.035
CUP_MASS = 0.05
CUP_FRICTION = (2.0, 0.5, 0.1)
CUP_INIT_POS = (0.5, 0.0, 0.4 + CUP_HALF_HEIGHT)
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


def build_model():
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    w = spec.worldbody
    w.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2, 2, 0.1],
        rgba=[0.85, 0.85, 0.85, 1],
    )
    table = w.add_body(name="table", pos=[0.5, 0, 0.2])
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.2, 0.2, 0.2],
        rgba=[0.55, 0.35, 0.2, 1],
    )
    cup = w.add_body(name="cup", pos=list(CUP_INIT_POS))
    cup.add_freejoint()
    cup.add_geom(
        name="cup_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[CUP_RADIUS, CUP_HALF_HEIGHT, 0],
        mass=CUP_MASS,
        friction=list(CUP_FRICTION),
        rgba=[0.2, 0.5, 0.9, 1],
    )
    return spec.compile()


def gripper_geom_ids(model):
    ids = set()
    for bname in ("left_finger", "right_finger"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            continue
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) == bid:
                ids.add(g)
    return ids


def cup_geom_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")


def has_gripper_cup_contact(model, data, cup_gid, finger_gids):
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if cup_gid in (g1, g2) and (g1 in finger_gids or g2 in finger_gids):
            return True
    return False


def grade(path: str):
    try:
        d = np.load(path)
    except FileNotFoundError:
        return False, {"error": "no final_state.npz produced"}
    except Exception as e:
        return False, {"error": f"failed to load npz: {e}"}

    missing = [k for k in ("ctrl_trace",) if k not in d.files]
    if missing:
        return False, {"error": f"npz missing keys: {missing}"}

    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    ctrl_trace = d["ctrl_trace"]
    if ctrl_trace.ndim != 2 or ctrl_trace.shape[1] != model.nu:
        return False, {"error": "ctrl_trace has unexpected shape"}
    for ctrl in ctrl_trace:
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    cup_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    cup_gid = cup_geom_id(model)
    finger_gids = gripper_geom_ids(model)

    cup_z_initial = float(data.xpos[cup_bid][2])
    contact_at_load = has_gripper_cup_contact(model, data, cup_gid, finger_gids)

    contact_frames = 0
    min_cup_z = cup_z_initial
    for _ in range(FORWARD_SIM_STEPS):
        mujoco.mj_step(model, data)
        z = float(data.xpos[cup_bid][2])
        if z < min_cup_z:
            min_cup_z = z
        if has_gripper_cup_contact(model, data, cup_gid, finger_gids):
            contact_frames += 1

    cup_pos_after = data.xpos[cup_bid].copy()
    cup_z_after = float(cup_pos_after[2])
    contact_fraction = contact_frames / FORWARD_SIM_STEPS

    stable_lifted = cup_z_after >= SUCCESS_CUP_Z
    stable_held = contact_fraction >= 0.5
    success = stable_lifted and stable_held
    height_margin = float(cup_z_after - SUCCESS_CUP_Z)
    required_contact_fraction = 0.5
    height_progress = clamp01((cup_z_after - CUP_INIT_POS[2]) / (SUCCESS_CUP_Z - CUP_INIT_POS[2]))
    contact_progress = clamp01(contact_fraction / required_contact_fraction)
    score_components = [
        {"name": "height_progress", "value": height_progress, "weight": 0.50},
        {"name": "contact_progress", "value": contact_progress, "weight": 0.50},
    ]
    progress_score = weighted_subscore(*[(item["name"], item["value"], item["weight"]) for item in score_components])
    replay_steps = int(ctrl_trace.shape[0])

    return success, {
        "cup_z_at_load": cup_z_initial,
        "contact_at_load": contact_at_load,
        "cup_z_after_forward_sim": cup_z_after,
        "cup_xyz_after": [float(x) for x in cup_pos_after],
        "min_cup_z_during_sim": min_cup_z,
        "contact_fraction_during_sim": contact_fraction,
        "height_margin": height_margin,
        "required_contact_fraction": required_contact_fraction,
        "height_progress": height_progress,
        "contact_progress": contact_progress,
        "progress_score": progress_score,
        "score_method": "weighted_subscores",
        "score_components": score_components,
        "replay_steps": replay_steps,
        "forward_sim_steps": FORWARD_SIM_STEPS,
        "threshold": SUCCESS_CUP_Z,
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/work/final_state.npz"
    ok, info = grade(path)
    print(json.dumps({"success": ok, "info": info}))
    sys.exit(0 if ok else 1)
