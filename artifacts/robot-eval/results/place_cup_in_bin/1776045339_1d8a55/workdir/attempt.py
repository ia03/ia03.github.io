import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import BIN_CENTER, CUP_INIT_POS, Sim


HAND_Q = np.array([0.0, 0.0, 0.0, -1.6, 0.0, 1.6, 0.785], dtype=float)
BIN_XY = np.array(BIN_CENTER[:2], dtype=float)
CUP_XY = np.array(CUP_INIT_POS[:2], dtype=float)


@dataclass
class Phase:
    name: str
    pos: np.ndarray
    grip: float
    steps: int


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    xyz = q[1:]
    n = np.linalg.norm(xyz)
    if n < 1e-9:
        return np.zeros(3)
    angle = 2.0 * math.atan2(n, q[0])
    return xyz / n * angle


class Controller:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.hand_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.left_finger = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.qmin = self.model.actuator_ctrlrange[:7, 0].copy()
        self.qmax = self.model.actuator_ctrlrange[:7, 1].copy()
        self.hand_target_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
        self.best_cup_z = float(self.sim.cup_position()[2])
        self.ever_contact = False

    def finger_midpoint(self) -> np.ndarray:
        return 0.5 * (self.data.xpos[self.left_finger] + self.data.xpos[self.right_finger])

    def hand_quat(self) -> np.ndarray:
        quat = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quat, self.data.xmat[self.hand_body])
        return quat

    def pose_error(self, pos_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos_err = pos_target - self.finger_midpoint()
        q_cur = self.hand_quat()
        q_err = quat_mul(self.hand_target_quat, quat_conj(q_cur))
        rot_err = quat_to_rotvec(q_err)
        return pos_err, rot_err

    def step_toward(self, pos_target: np.ndarray, grip_target: float, substeps: int = 1):
        for _ in range(substeps):
            jacp_l = np.zeros((3, self.model.nv))
            jacr_l = np.zeros((3, self.model.nv))
            jacp_r = np.zeros((3, self.model.nv))
            jacr_r = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp_l, jacr_l, self.left_finger)
            mujoco.mj_jacBody(self.model, self.data, jacp_r, jacr_r, self.right_finger)
            jacp = 0.5 * (jacp_l[:, :7] + jacp_r[:, :7])
            jacr = 0.5 * (jacr_l[:, :7] + jacr_r[:, :7])
            pos_err, rot_err = self.pose_error(pos_target)
            task = np.concatenate([14.0 * pos_err, 2.5 * rot_err])
            jac = np.vstack([jacp, jacr])
            damping = 1e-3
            lhs = jac.T @ jac + damping * np.eye(7)
            rhs = jac.T @ task + 0.03 * (HAND_Q - self.data.qpos[:7])
            dq = np.linalg.solve(lhs, rhs)
            q_des = np.clip(self.data.qpos[:7] + 0.22 * dq, self.qmin, self.qmax)
            self.data.ctrl[:7] = q_des
            self.data.ctrl[7] = float(np.clip(grip_target, 0.0, 255.0))
            self.sim.step()
            cup_z = float(self.sim.cup_position()[2])
            self.best_cup_z = max(self.best_cup_z, cup_z)
            self.ever_contact = self.ever_contact or self.sim.has_gripper_cup_contact()


def run_attempt(save_path: str = "/work/final_state.npz", render_dir: str | None = None):
    sim = Sim()
    ctrl = Controller(sim)

    phases = [
        Phase("approach_high", np.array([CUP_XY[0], CUP_XY[1], 0.60]), 255.0, 450),
        Phase("approach_low", np.array([CUP_XY[0], CUP_XY[1], 0.455]), 255.0, 350),
        Phase("close", np.array([CUP_XY[0], CUP_XY[1], 0.452]), 0.0, 320),
        Phase("lift", np.array([CUP_XY[0], CUP_XY[1], 0.62]), 0.0, 420),
        Phase("move_bin_high", np.array([BIN_XY[0], BIN_XY[1], 0.64]), 0.0, 800),
        Phase("lower_bin", np.array([BIN_XY[0], BIN_XY[1], 0.50]), 0.0, 420),
        Phase("release", np.array([BIN_XY[0], BIN_XY[1], 0.51]), 255.0, 420),
        Phase("retreat", np.array([BIN_XY[0], BIN_XY[1], 0.62]), 255.0, 500),
        Phase("settle", np.array([BIN_XY[0], BIN_XY[1], 0.70]), 255.0, 500),
    ]

    frames = []
    for i, phase in enumerate(phases):
        for step in range(phase.steps):
            ctrl.step_toward(phase.pos, phase.grip)
            if render_dir and step % 100 == 0:
                frames.append((phase.name, sim.render(width=320, height=240)))
        if i == 4 and not os.path.exists(save_path):
            sim.save_final_state(save_path)
    sim.save_final_state(save_path)

    if render_dir:
        os.makedirs(render_dir, exist_ok=True)
        for idx, (name, frame) in enumerate(frames):
            path = os.path.join(render_dir, f"{idx:03d}_{name}.png")
            from imageio.v3 import imwrite

            imwrite(path, frame)

    print("best_cup_z", ctrl.best_cup_z)
    print("ever_contact", ctrl.ever_contact)
    print("final_cup_pos", sim.cup_position())
    print("final_hand_mid", ctrl.finger_midpoint())
    return sim, ctrl


if __name__ == "__main__":
    run_attempt()
