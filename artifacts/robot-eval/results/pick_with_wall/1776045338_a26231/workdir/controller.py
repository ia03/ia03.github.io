import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


HAND_BODY = "hand"
INNER_PAD_GEOMS = (69, 77)
FINAL_STATE_PATH = "/work/final_state.npz"


def rotation_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(R_cur[:, 0], R_des[:, 0])
        + np.cross(R_cur[:, 1], R_des[:, 1])
        + np.cross(R_cur[:, 2], R_des[:, 2])
    )


def pose_from_midpoint(midpoint: np.ndarray, pinch_axis_world: np.ndarray) -> np.ndarray:
    y_axis = pinch_axis_world / np.linalg.norm(pinch_axis_world)
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


@dataclass
class Stage:
    midpoint: np.ndarray
    rot: np.ndarray
    grip: float
    steps: int
    pos_gain: float = 6.0
    rot_gain: float = 2.5
    label: str = ""


class Controller:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.hand_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.pad_offset_local = self._compute_pad_midpoint_offset()
        self.last_contact_step = -1

    def _compute_pad_midpoint_offset(self) -> np.ndarray:
        pts = np.array([self.sim.data.geom_xpos[g] for g in INNER_PAD_GEOMS])
        midpoint = pts.mean(axis=0)
        R = self.sim.data.xmat[self.hand_body_id].reshape(3, 3)
        return R.T @ (midpoint - self.sim.data.xpos[self.hand_body_id])

    def hand_pose(self):
        pos = self.sim.data.xpos[self.hand_body_id].copy()
        rot = self.sim.data.xmat[self.hand_body_id].reshape(3, 3).copy()
        return pos, rot

    def current_midpoint(self) -> np.ndarray:
        hand_pos, hand_rot = self.hand_pose()
        return hand_pos + hand_rot @ self.pad_offset_local

    def step_to_pose(self, midpoint_target: np.ndarray, rot_target: np.ndarray, grip: float, pos_gain: float, rot_gain: float):
        hand_pos, hand_rot = self.hand_pose()
        midpoint = hand_pos + hand_rot @ self.pad_offset_local
        pos_err = midpoint_target - midpoint
        rot_err = rotation_error(hand_rot, rot_target)

        jacp = np.zeros((3, self.sim.model.nv))
        jacr = np.zeros((3, self.sim.model.nv))
        mujoco.mj_jacBody(self.sim.model, self.sim.data, jacp, jacr, self.hand_body_id)
        offset = hand_rot @ self.pad_offset_local
        jac_mid = jacp + np.cross(jacr.T, offset).T
        J = np.vstack([pos_gain * jac_mid[:, :7], rot_gain * jacr[:, :7]])
        err = np.concatenate([pos_gain * pos_err, rot_gain * rot_err])
        damping = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        q_target = np.clip(
            self.sim.data.qpos[:7] + dq,
            self.sim.model.actuator_ctrlrange[:7, 0],
            self.sim.model.actuator_ctrlrange[:7, 1],
        )
        self.sim.data.ctrl[:7] = q_target
        self.sim.data.ctrl[7] = grip
        self.sim.step(1)
        if self.sim.has_gripper_cup_contact():
            self.last_contact_step = len(self.sim._ctrl_trace)

    def run_stage(self, stage: Stage):
        for _ in range(stage.steps):
            self.step_to_pose(stage.midpoint, stage.rot, stage.grip, stage.pos_gain, stage.rot_gain)


def summarize(sim: Sim) -> dict:
    trace = sim._trace
    best_x = min(float(entry["cup_pos"][0]) for entry in trace)
    best_z = max(float(entry["cup_pos"][2]) for entry in trace)
    final = sim.cup_position()
    return {
        "best_x": best_x,
        "best_z": best_z,
        "final_pos": final.copy(),
        "contact_seen": any(entry["cup_contact"] > 0.5 for entry in trace),
        "steps": len(sim._ctrl_trace),
        "time": float(sim.data.time),
    }


def attempt(sequence_name: str, waypoints: list[Stage], save_early_step: int | None = None) -> dict:
    sim = Sim()
    ctrl = Controller(sim)
    for idx, stage in enumerate(waypoints):
        ctrl.run_stage(stage)
        if save_early_step is not None and idx == save_early_step:
            sim.save_final_state(FINAL_STATE_PATH)
    sim.save_final_state(FINAL_STATE_PATH)
    stats = summarize(sim)
    print(sequence_name, stats)
    return stats


def build_sequence(approach_y: float, grip_height: float, carry_y: float, carry_x: float, carry_z: float, final_x: float, final_z: float):
    cup = np.array([0.55, approach_y, 0.435], dtype=float)
    pinch_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    rot = pose_from_midpoint(cup, pinch_axis)
    above = cup + np.array([0.0, 0.0, 0.18])
    pregrasp = cup + np.array([0.0, 0.0, 0.08])
    grasp = np.array([0.55, approach_y, grip_height], dtype=float)
    lift = np.array([0.55, approach_y, carry_z], dtype=float)
    retreat1 = np.array([carry_x, carry_y, carry_z], dtype=float)
    retreat2 = np.array([final_x, carry_y, final_z], dtype=float)
    hold = np.array([final_x, carry_y, final_z], dtype=float)
    return [
        Stage(above, rot, 0.0, 500, label="above"),
        Stage(pregrasp, rot, 0.0, 400, label="pregrasp"),
        Stage(grasp, rot, 0.0, 400, pos_gain=7.0, label="grasp_align"),
        Stage(grasp, rot, 255.0, 350, pos_gain=7.0, label="close"),
        Stage(lift, rot, 255.0, 500, label="lift"),
        Stage(retreat1, rot, 255.0, 700, label="retreat1"),
        Stage(retreat2, rot, 255.0, 500, label="retreat2"),
        Stage(hold, rot, 255.0, 600, label="hold"),
    ]


if __name__ == "__main__":
    configs = [
        ("baseline", 0.15, 0.435, 0.10, 0.42, 0.66, 0.38, 0.60),
        ("lower_grasp", 0.15, 0.430, 0.10, 0.42, 0.67, 0.37, 0.61),
        ("side_carry", 0.15, 0.435, 0.06, 0.40, 0.67, 0.36, 0.61),
    ]
    best = None
    for idx, cfg in enumerate(configs):
        name, approach_y, grip_height, carry_y, carry_x, carry_z, final_x, final_z = cfg
        stats = attempt(
            name,
            build_sequence(approach_y, grip_height, carry_y, carry_x, carry_z, final_x, final_z),
            save_early_step=2 if idx == 0 else None,
        )
        score = (
            1.0 * stats["contact_seen"]
            + max(0.0, 0.55 - stats["best_x"])
            + max(0.0, stats["best_z"] - 0.435)
            + max(0.0, 0.55 - stats["final_pos"][0])
            + max(0.0, stats["final_pos"][2] - 0.435)
        )
        if best is None or score > best[0]:
            best = (score, name, stats)
    print("best", best)
    print("saved", os.path.exists(FINAL_STATE_PATH), FINAL_STATE_PATH)
