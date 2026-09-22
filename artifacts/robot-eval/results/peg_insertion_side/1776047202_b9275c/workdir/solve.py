#!/usr/bin/env python
"""
Heuristic controller for the Peg Insertion Side task.

Runs a single rollout from the canonical start state and saves:
  /work/final_state.npz

The saved file contains the control trace used by the grader.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import mujoco

from sim import Sim


def _quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    mujoco.mju_mulQuat(out, np.asarray(q1, dtype=float), np.asarray(q2, dtype=float))
    return out


def _orientation_error(R_curr: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    q_curr = _mat_to_quat(R_curr)
    q_des = _mat_to_quat(R_des)
    q_err = _quat_mul(q_des, _quat_conj(q_curr))
    w = np.zeros(3, dtype=float)
    mujoco.mju_quat2Vel(w, q_err, 1.0)
    return w


def _dls_solve(J: np.ndarray, e: np.ndarray, damping: float) -> np.ndarray:
    # Solve: dq = J^T (J J^T + λI)^-1 e
    JJt = J @ J.T
    JJt.flat[:: JJt.shape[0] + 1] += damping
    return J.T @ np.linalg.solve(JJt, e)


@dataclass
class ControllerGains:
    kp_pos: float = 6.0
    kp_rot: float = 3.0
    damping: float = 1e-3
    max_dq: float = 0.15


class PandaIKController:
    def __init__(self, sim: Sim, hand_body: str = "hand", gains: ControllerGains | None = None):
        self.sim = sim
        self.m = sim.model
        self.d = sim.data
        self.gains = gains or ControllerGains()
        self.hand_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, hand_body)
        self.peg_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "peg")

        # 7 arm joints are the first 7 qpos entries in this model.
        self.arm_qpos_slice = slice(0, 7)
        self.arm_nv = 7

        # Desired end-effector orientation: x->world x, y->world y, z->down.
        self.R_des = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)

        self._jacp = np.zeros((3, self.m.nv), dtype=float)
        self._jacr = np.zeros((3, self.m.nv), dtype=float)

    def hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        p = np.array(self.d.xpos[self.hand_id], dtype=float)
        R = np.array(self.d.xmat[self.hand_id], dtype=float).reshape(3, 3)
        return p, R

    def peg_pose(self) -> tuple[np.ndarray, np.ndarray]:
        p = np.array(self.d.xpos[self.peg_id], dtype=float)
        R = np.array(self.d.xmat[self.peg_id], dtype=float).reshape(3, 3)
        return p, R

    def step_to(self, p_des: np.ndarray, use_orientation: bool = True, gripper: float = 255.0):
        p_curr, R_curr = self.hand_pose()
        pos_err = (p_des - p_curr) * self.gains.kp_pos

        mujoco.mj_jacBody(self.m, self.d, self._jacp, self._jacr, self.hand_id)
        Jp = self._jacp[:, : self.arm_nv]

        if use_orientation:
            rot_err = _orientation_error(R_curr, self.R_des) * self.gains.kp_rot
            Jr = self._jacr[:, : self.arm_nv]
            J = np.vstack([Jp, Jr])
            e = np.concatenate([pos_err, rot_err])
        else:
            J = Jp
            e = pos_err

        dq = _dls_solve(J, e, self.gains.damping)
        dq = np.clip(dq, -self.gains.max_dq, self.gains.max_dq)

        qpos = self.d.qpos[self.arm_qpos_slice].copy()
        qpos = qpos + dq

        # Clamp to actuator ranges.
        for i in range(self.arm_nv):
            lo, hi = self.m.actuator_ctrlrange[i]
            qpos[i] = float(np.clip(qpos[i], lo, hi))

        self.d.ctrl[: self.arm_nv] = qpos
        self.d.ctrl[7] = float(np.clip(gripper, *self.m.actuator_ctrlrange[7]))


def compute_progress(sim: Sim) -> dict[str, float]:
    m = sim.model
    d = sim.data
    peg_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peg")
    p = np.array(d.xpos[peg_id], dtype=float)
    R = np.array(d.xmat[peg_id], dtype=float).reshape(3, 3)
    x_axis_alignment = float(np.dot(R[:, 0], np.array([1.0, 0.0, 0.0])))

    final_x, final_y, final_z = map(float, p)
    insertion_progress = 1.0 if final_x >= 0.515 else float(np.clip((final_x - 0.47) / (0.515 - 0.47), 0.0, 1.0))
    y_err = abs(final_y - (-0.10))
    y_progress = 1.0 if y_err <= 0.025 else float(np.clip(1.0 - (y_err - 0.025) / 0.025, 0.0, 1.0))
    z_err = abs(final_z - 0.52)
    z_progress = 1.0 if z_err <= 0.020 else float(np.clip(1.0 - (z_err - 0.020) / 0.020, 0.0, 1.0))
    alignment_progress = 1.0 if abs(x_axis_alignment) >= 0.45 else float(np.clip(abs(x_axis_alignment) / 0.45, 0.0, 1.0))

    return {
        "final_peg_x": final_x,
        "final_peg_y": final_y,
        "final_peg_z": final_z,
        "x_axis_alignment": x_axis_alignment,
        "insertion_progress": insertion_progress,
        "y_progress": y_progress,
        "z_progress": z_progress,
        "alignment_progress": alignment_progress,
    }


def main():
    sim = Sim()
    ctrl = PandaIKController(sim)

    # Phases: approach above tab -> descend -> push.
    peg_p0, _ = ctrl.peg_pose()

    # Keep a stable lane and height; track peg y/z lightly during pushing.
    pre_above = np.array([peg_p0[0] - 0.12, peg_p0[1], peg_p0[2] + 0.12], dtype=float)
    pre_contact = np.array([peg_p0[0] - 0.10, peg_p0[1], peg_p0[2] + 0.04], dtype=float)

    # Approach.
    for _ in range(900):
        ctrl.step_to(pre_above, use_orientation=True, gripper=255.0)
        sim.step(1)

    for _ in range(650):
        ctrl.step_to(pre_contact, use_orientation=True, gripper=255.0)
        sim.step(1)

    # Initial save: ensures a replay exists even if later phases fail.
    sim.save_final_state("/work/final_state.npz")

    # Push: move the hand forward along +x while keeping y/z aligned to peg.
    push_steps = 2600
    x_start = float(pre_contact[0])
    x_end = 0.62
    for t in range(push_steps):
        alpha = (t + 1) / push_steps
        x_des = (1 - alpha) * x_start + alpha * x_end
        peg_p, _ = ctrl.peg_pose()
        y_des = float(0.8 * peg_p[1] + 0.2 * (-0.102))
        z_des = float(0.8 * peg_p[2] + 0.2 * (0.52))
        p_des = np.array([x_des, y_des, z_des + 0.01], dtype=float)
        ctrl.step_to(p_des, use_orientation=True, gripper=255.0)
        sim.step(1)

        # Overwrite the best-so-far state once we start making forward progress.
        if t in (200, 800, 1400, 2200, 2599):
            sim.save_final_state("/work/final_state.npz")

    # Hold still to let the peg settle while keeping light contact.
    for _ in range(800):
        peg_p, _ = ctrl.peg_pose()
        p_des = np.array([0.62, float(peg_p[1]), float(peg_p[2]) + 0.01], dtype=float)
        ctrl.step_to(p_des, use_orientation=True, gripper=255.0)
        sim.step(1)

    # Final overwrite.
    sim.save_final_state("/work/final_state.npz")
    metrics = compute_progress(sim)
    print("metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})
    print("saved:", os.path.abspath("/work/final_state.npz"))


if __name__ == "__main__":
    main()

