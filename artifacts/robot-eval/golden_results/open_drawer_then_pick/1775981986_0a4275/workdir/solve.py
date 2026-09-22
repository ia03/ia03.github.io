import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def orientation_error(target, current):
    err = (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )
    return 0.5 * err


@dataclass
class PoseTarget:
    pos: np.ndarray
    rot: np.ndarray
    grip: float


class Controller:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.hand_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.arm_dofs = np.arange(7)
        self.home_q = np.array([0.0, -0.6, 0.0, -2.3, 0.0, 1.7, 0.8], dtype=float)
        self.qmin = self.model.actuator_ctrlrange[:7, 0].copy()
        self.qmax = self.model.actuator_ctrlrange[:7, 1].copy()

    def hand_pose(self):
        pos = self.data.xpos[self.hand_body].copy()
        rot = self.data.xmat[self.hand_body].reshape(3, 3).copy()
        return pos, rot

    def step_towards(self, target: PoseTarget, pos_gain=7.0, rot_gain=2.5, damping=1e-2):
        pos, rot = self.hand_pose()
        pos_err = target.pos - pos
        rot_err = orientation_error(target.rot, rot)
        err = np.concatenate([pos_gain * pos_err, rot_gain * rot_err])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        A = J.T @ J + damping * np.eye(7)
        posture = 0.03 * (self.home_q - self.data.qpos[:7])
        dq = np.linalg.solve(A, J.T @ err + posture)
        q_des = np.clip(self.data.qpos[:7] + dq, self.qmin, self.qmax)
        self.data.ctrl[:7] = q_des
        self.data.ctrl[7] = target.grip
        self.sim.step()
        return np.linalg.norm(pos_err), np.linalg.norm(rot_err)

    def hold(self, target: PoseTarget, steps: int):
        for _ in range(steps):
            self.step_towards(target)

    def move(self, target: PoseTarget, steps: int, settle=20):
        start_pos, start_rot = self.hand_pose()
        start_grip = float(self.data.ctrl[7])
        for i in range(steps):
            a = (i + 1) / steps
            interp_pos = (1 - a) * start_pos + a * target.pos
            interp_rot = start_rot @ exp_so3(log_so3(start_rot.T @ target.rot) * a)
            interp_grip = (1 - a) * start_grip + a * target.grip
            self.step_towards(PoseTarget(interp_pos, interp_rot, interp_grip))
        self.hold(target, settle)


def log_so3(R):
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    w = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    )
    return 0.5 * theta / math.sin(theta) * w


def exp_so3(w):
    theta = np.linalg.norm(w)
    if theta < 1e-8:
        return np.eye(3)
    k = w / theta
    K = np.array(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
        dtype=float,
    )
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def run():
    sim = Sim()
    ctrl = Controller(sim)

    top_down = rot_z(math.pi / 4.0) @ np.diag([1.0, 1.0, -1.0])
    front_in = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    targets = [
        PoseTarget(np.array([0.52, -0.02, 0.58]), top_down, 255.0),
        PoseTarget(np.array([0.71, -0.02, 0.54]), top_down, 255.0),
        PoseTarget(np.array([0.74, -0.02, 0.47]), top_down, 180.0),
        PoseTarget(np.array([0.82, -0.02, 0.47]), top_down, 150.0),
        PoseTarget(np.array([0.80, -0.02, 0.60]), top_down, 150.0),
        PoseTarget(np.array([0.74, -0.02, 0.54]), front_in, 255.0),
        PoseTarget(np.array([0.70, -0.02, 0.46]), front_in, 255.0),
        PoseTarget(np.array([0.66, -0.02, 0.44]), front_in, 255.0),
        PoseTarget(np.array([0.63, -0.02, 0.44]), front_in, 40.0),
        PoseTarget(np.array([0.63, -0.02, 0.44]), front_in, 10.0),
        PoseTarget(np.array([0.63, -0.02, 0.54]), front_in, 10.0),
        PoseTarget(np.array([0.63, -0.02, 0.68]), front_in, 10.0),
    ]
    steps = [120, 160, 120, 160, 80, 120, 120, 100, 80, 120, 120, 120]

    for idx, (t, n) in enumerate(zip(targets, steps)):
        ctrl.move(t, n, settle=25 if idx < 3 else 15)
        if idx == 3 or idx == 7:
            sim.save_final_state("/work/final_state.npz")

    sim.step(250)
    sim.save_final_state("/work/final_state.npz")

    print("drawer_open", sim.drawer_open_amount())
    print("block_pos", sim.block_position())
    print("contact", sim.has_gripper_block_contact())
    print("steps", len(sim._ctrl_trace))


if __name__ == "__main__":
    run()
