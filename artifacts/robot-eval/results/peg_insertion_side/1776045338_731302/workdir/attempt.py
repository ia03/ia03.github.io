import math
import os
import sys
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


ARM_DOF = 7
FINGER_CTRL = 7
HAND_BODY = "hand"


@dataclass
class Waypoint:
    pos: np.ndarray
    hold_steps: int
    quat: np.ndarray | None = None
    grip: float = 0.0


def body_pose(sim: Sim, body_name: str):
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = sim.data.xpos[bid].copy()
    rot = sim.data.xmat[bid].reshape(3, 3).copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.reshape(-1))
    return bid, pos, rot, quat


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    return np.array(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    )


def orientation_error(current, target):
    qerr = quat_mul(target, quat_conj(current))
    if qerr[0] < 0:
        qerr = -qerr
    return 2.0 * qerr[1:]


def control_to_pose(sim: Sim, body_id: int, target_pos, target_quat, grip=0.0, steps=1, tool_local=None):
    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    arm_slice = slice(0, ARM_DOF)
    q_min = sim.model.actuator_ctrlrange[:ARM_DOF, 0]
    q_max = sim.model.actuator_ctrlrange[:ARM_DOF, 1]
    q_home = np.zeros(ARM_DOF)
    for _ in range(steps):
        cur_pos = sim.data.xpos[body_id].copy()
        cur_rot = sim.data.xmat[body_id].reshape(3, 3).copy()
        cur_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, cur_rot.reshape(-1))
        control_pos = cur_pos if tool_local is None else cur_pos + cur_rot @ tool_local
        pos_err = target_pos - control_pos
        rot_err = orientation_error(cur_quat, target_quat)
        task = np.concatenate([26.0 * pos_err, 18.0 * rot_err])
        if tool_local is None:
            mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, body_id)
        else:
            mujoco.mj_jac(sim.model, sim.data, jacp, jacr, control_pos, body_id)
        J = np.vstack([jacp[:, arm_slice], jacr[:, arm_slice]])
        H = J.T @ J + 2e-3 * np.eye(ARM_DOF)
        g = J.T @ task + 4e-2 * (q_home - sim.data.qpos[:ARM_DOF])
        dq = np.linalg.solve(H, g)
        q_des = np.clip(sim.data.qpos[:ARM_DOF] + 0.035 * dq, q_min, q_max)
        sim.data.ctrl[:ARM_DOF] = q_des
        sim.data.ctrl[FINGER_CTRL] = grip
        sim.step()


def follow_waypoints(sim: Sim, waypoints, tool_local=None):
    body_id, _, _, base_quat = body_pose(sim, HAND_BODY)
    for wp in waypoints:
        target_quat = base_quat if wp.quat is None else wp.quat
        for _ in range(wp.hold_steps):
            control_to_pose(sim, body_id, wp.pos, target_quat, grip=wp.grip, tool_local=tool_local)


def peg_alignment(sim: Sim):
    peg_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    peg_rot = sim.data.xmat[peg_id].reshape(3, 3)
    return abs(float(peg_rot[0, 0]))


def progress_metrics(sim: Sim):
    peg = sim.peg_position()
    insertion = 1.0 if peg[0] >= 0.515 else np.clip((peg[0] - 0.47) / (0.515 - 0.47), 0, 1)
    yerr = abs(peg[1] + 0.10)
    zerr = abs(peg[2] - 0.52)
    yprog = 1.0 if yerr <= 0.025 else np.clip(1 - (yerr - 0.025) / 0.025, 0, 1)
    zprog = 1.0 if zerr <= 0.020 else np.clip(1 - (zerr - 0.020) / 0.020, 0, 1)
    aprog = 1.0 if peg_alignment(sim) >= 0.45 else np.clip(peg_alignment(sim) / 0.45, 0, 1)
    eprog = np.clip(4500 / max(1, len(sim._ctrl_trace)), 0, 1)
    score = 0.45 * insertion + 0.15 * yprog + 0.15 * zprog + 0.20 * aprog + 0.05 * eprog
    return peg, peg_alignment(sim), score


def run_attempt(
    save_path="/work/final_state.npz",
    render_dir=None,
    approach_z=0.532,
    approach_y=-0.102,
    push_end_x=0.548,
    tool_z=0.090,
):
    sim = Sim()
    _, _, _, hand_quat = body_pose(sim, HAND_BODY)
    tool_local = np.array([0.0, 0.0, tool_z])

    # Use the tab as a pushing target: approach from robot side, then drive along +x.
    waypoints = [
        Waypoint(np.array([0.33, -0.05, 0.72]), 220, hand_quat, 0.0),
        Waypoint(np.array([0.38, approach_y, 0.62]), 180, hand_quat, 0.0),
        Waypoint(np.array([0.405, approach_y, approach_z]), 180, hand_quat, 0.0),
        Waypoint(np.array([0.425, approach_y, approach_z]), 120, hand_quat, 0.0),
        Waypoint(np.array([0.455, approach_y, approach_z]), 140, hand_quat, 0.0),
        Waypoint(np.array([0.490, approach_y, approach_z]), 180, hand_quat, 0.0),
        Waypoint(np.array([0.520, approach_y, approach_z]), 200, hand_quat, 0.0),
        Waypoint(np.array([push_end_x, approach_y, approach_z]), 220, hand_quat, 0.0),
        Waypoint(np.array([push_end_x, approach_y, approach_z + 0.020]), 140, hand_quat, 0.0),
    ]

    # Early mandatory save after a few hundred steps.
    for wp in waypoints[:3]:
        follow_waypoints(sim, [wp], tool_local=tool_local)
    sim.save_final_state(save_path)

    follow_waypoints(sim, waypoints[3:], tool_local=tool_local)
    sim.data.ctrl[:ARM_DOF] = sim.data.qpos[:ARM_DOF]
    sim.data.ctrl[FINGER_CTRL] = 0.0
    sim.step(300)
    sim.save_final_state(save_path)

    peg_pos, alignment, score = progress_metrics(sim)
    print("final peg", peg_pos)
    print("alignment", alignment)
    print("score_est", score)
    print("steps", len(sim._ctrl_trace))

    if render_dir:
        os.makedirs(render_dir, exist_ok=True)
        for idx, cam in enumerate([-1]):
            img = sim.render(width=640, height=480, camera=cam)
            from imageio.v3 import imwrite
            imwrite(os.path.join(render_dir, f"final_{idx}.png"), img)


if __name__ == "__main__":
    kwargs = {}
    for arg in sys.argv[1:]:
        key, value = arg.split("=", 1)
        kwargs[key] = float(value)
    run_attempt(render_dir="/work/renders", **kwargs)
