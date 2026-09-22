import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


ARM_DOF = 7
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0
HAND_TO_PINCH_CENTER = 0.1029


@dataclass
class AttemptResult:
    name: str
    max_cup_z: float
    final_cup_z: float
    held_steps: int
    best_label: str


def orientation_error(current, target):
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )


def clamp_arm_joints(model, q):
    out = q.copy()
    for i in range(ARM_DOF):
        lo, hi = model.jnt_range[i]
        out[i] = np.clip(out[i], lo, hi)
    return out


def make_rot(yaw):
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def solve_arm_pose(target_pos, target_rot, seed, iters=200):
    sim = Sim()
    model, data = sim.model, sim.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    data.qpos[:ARM_DOF] = seed
    mujoco.mj_forward(model, data)

    for _ in range(iters):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)

        cur_pos = data.xpos[hand_id].copy()
        cur_rot = data.xmat[hand_id].reshape(3, 3).copy()
        pos_err = target_pos - cur_pos
        rot_err = orientation_error(cur_rot, target_rot)
        err = np.concatenate([2.0 * pos_err, 0.6 * rot_err])
        J = np.vstack([2.0 * jacp[:, :ARM_DOF], 0.6 * jacr[:, :ARM_DOF]])
        dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(ARM_DOF), J.T @ err)
        data.qpos[:ARM_DOF] = clamp_arm_joints(model, data.qpos[:ARM_DOF] + 0.7 * dq)
        mujoco.mj_forward(model, data)

    cur_pos = data.xpos[hand_id].copy()
    cur_rot = data.xmat[hand_id].reshape(3, 3).copy()
    pos_err = np.linalg.norm(target_pos - cur_pos)
    rot_err = np.linalg.norm(orientation_error(cur_rot, target_rot))
    return data.qpos[:ARM_DOF].copy(), pos_err, rot_err


def plan_waypoints(cup_pos, yaw, descend_offset, lift_offset):
    rot = make_rot(yaw)
    offset = rot @ np.array([0.0, 0.0, HAND_TO_PINCH_CENTER])
    pregrasp = cup_pos + np.array([0.0, 0.0, 0.10]) - offset
    grasp = cup_pos + np.array([0.0, 0.0, descend_offset]) - offset
    lift = cup_pos + np.array([0.0, 0.0, lift_offset]) - offset

    seeds = [
        np.zeros(7),
        np.array([0.0, 0.8, 0.0, -1.8, 0.0, 2.6, 0.8]),
        np.array([0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.7]),
        np.array([1.0, 0.7, -0.6, -1.8, 0.0, 1.9, 0.1]),
        np.array([-1.0, 0.7, 0.6, -1.8, 0.0, 1.9, -0.1]),
    ]

    def best_pose(target, seed_hint):
        best = None
        for seed in [seed_hint] + seeds:
            q, pe, re = solve_arm_pose(target, rot, seed)
            score = pe + 0.2 * re
            if best is None or score < best[0]:
                best = (score, q)
        return best[1]

    q_pre = best_pose(pregrasp, seeds[0])
    q_grasp = best_pose(grasp, q_pre)
    q_lift = best_pose(lift, q_grasp)
    return rot, q_pre, q_grasp, q_lift


def maybe_save(sim, best, label):
    cup_z = float(sim.cup_position()[2])
    if cup_z > best["z"]:
        sim.save_final_state("/work/final_state.npz")
        best["z"] = cup_z
        best["label"] = label


def track_target(sim, q_target, grip_target, steps, best, label):
    for i in range(steps):
        sim.data.ctrl[:ARM_DOF] = q_target
        sim.data.ctrl[7] = grip_target
        sim.step(10)
        maybe_save(sim, best, f"{label}-{i}")


def interpolate_track(sim, q_start, q_end, grip_start, grip_end, chunks, best, label):
    for i in range(chunks):
        alpha = (i + 1) / chunks
        q = (1.0 - alpha) * q_start + alpha * q_end
        grip = (1.0 - alpha) * grip_start + alpha * grip_end
        sim.data.ctrl[:ARM_DOF] = q
        sim.data.ctrl[7] = grip
        sim.step(12)
        maybe_save(sim, best, f"{label}-{i}")


def cup_contacts(sim):
    model, data = sim.model, sim.data
    cup_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")
    names = []
    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom1 == cup_geom or con.geom2 == cup_geom:
            other = con.geom2 if con.geom1 == cup_geom else con.geom1
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other))
    return names


def run_attempt(name, yaw, descend_offset, lift_offset, hold_after_close, hold_after_lift, best):
    sim = Sim()
    cup0 = sim.cup_position().copy()
    _, q_pre, q_grasp, q_lift = plan_waypoints(cup0, yaw, descend_offset, lift_offset)
    q_home = sim.data.qpos[:ARM_DOF].copy()

    interpolate_track(sim, q_home, q_pre, GRIPPER_OPEN, GRIPPER_OPEN, 70, best, f"{name}-pre")
    track_target(sim, q_pre, GRIPPER_OPEN, 20, best, f"{name}-prehold")
    interpolate_track(sim, q_pre, q_grasp, GRIPPER_OPEN, GRIPPER_OPEN, 50, best, f"{name}-down")
    track_target(sim, q_grasp, GRIPPER_OPEN, 20, best, f"{name}-settle-open")
    interpolate_track(sim, q_grasp, q_grasp, GRIPPER_OPEN, GRIPPER_CLOSED, 60, best, f"{name}-close")
    track_target(sim, q_grasp, GRIPPER_CLOSED, hold_after_close, best, f"{name}-hold")
    interpolate_track(sim, q_grasp, q_lift, GRIPPER_CLOSED, GRIPPER_CLOSED, 70, best, f"{name}-lift")
    track_target(sim, q_lift, GRIPPER_CLOSED, hold_after_lift, best, f"{name}-post")

    cup_z = float(sim.cup_position()[2])
    max_z = max(best["z"], cup_z)
    held_steps = 0
    for _ in range(10):
        if sim.cup_position()[2] >= 0.52:
            held_steps += 10
        sim.step(10)
    print(
        f"{name}: final_z={cup_z:.4f} contacts={cup_contacts(sim)} q_lift={np.round(q_lift, 3)}"
    )
    return AttemptResult(name, max_z, cup_z, held_steps, best["label"])


def main():
    sim = Sim()
    sim.save_final_state("/work/final_state.npz")
    best = {"z": float(sim.cup_position()[2]), "label": "initial"}

    attempts = [
        ("yaw0", 0.0, -0.010, 0.16, 20, 40),
        ("yaw90", math.pi / 2, -0.010, 0.16, 20, 40),
        ("yaw45", math.pi / 4, -0.012, 0.16, 25, 45),
        ("yawm45", -math.pi / 4, -0.012, 0.16, 25, 45),
        ("yaw180", math.pi, -0.010, 0.16, 20, 40),
    ]

    results = [run_attempt(*attempt, best) for attempt in attempts]
    print("best", best)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
