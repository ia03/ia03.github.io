#!/usr/bin/env python3
"""
Baseline policy for the Stack Blocks task.

Writes a replayable control trace + final simulator state to /work/final_state.npz
via sim.save_final_state().
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import mujoco

from sim import Sim


@dataclass
class IKConfig:
    body_name: str = "hand"
    damping: float = 2e-3
    step_scale: float = 0.75
    max_dq: float = 0.12
    tol: float = 0.004
    null_k: float = 0.08
    ee_offset: float = 0.0584


class IK:
    def __init__(self, sim: Sim, cfg: IKConfig):
        self.sim = sim
        self.cfg = cfg
        self.model = sim.model
        self.data = sim.data
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cfg.body_name)
        self._jacp = np.zeros((3, self.model.nv), dtype=float)
        self._jacr = np.zeros((3, self.model.nv), dtype=float)

        # Arm is the first 7 DOFs in this model.
        self.arm_dofs = slice(0, 7)
        self.joint_ranges = self.model.actuator_ctrlrange[:7].copy()
        self._q_nom = np.zeros(7, dtype=float)
        self._ee_offset = float(cfg.ee_offset)

    def hand_pos(self) -> np.ndarray:
        return np.array(self.data.xpos[self.body_id])

    def step_towards(self, target_pos: np.ndarray, gripper: float) -> float:
        mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, self.body_id)
        Jp_hand = self._jacp[:, self.arm_dofs]  # (3, 7)
        Jr = self._jacr[:, self.arm_dofs]  # (3, 7)

        # Control a point on the hand located along the hand +Z axis (towards the fingers).
        R = self.data.xmat[self.body_id].reshape(3, 3)
        r_world = R[:, 2] * self._ee_offset
        ee_pos = self.hand_pos() + r_world

        err = target_pos - ee_pos
        err_norm = float(np.linalg.norm(err))

        # Point Jacobian: Jp_point = Jp_hand + ω×r
        Jp = np.empty_like(Jp_hand)
        for j in range(7):
            Jp[:, j] = Jp_hand[:, j] + np.cross(Jr[:, j], r_world)

        JJt = Jp @ Jp.T
        pinv = Jp.T @ np.linalg.solve(JJt + (self.cfg.damping * np.eye(3)), np.eye(3))  # (7, 3)
        dq_task = pinv @ err

        q = self.data.qpos[self.arm_dofs].copy()
        N = np.eye(7) - pinv @ Jp
        dq = dq_task + N @ (self.cfg.null_k * (self._q_nom - q))
        dq = np.clip(dq * self.cfg.step_scale, -self.cfg.max_dq, self.cfg.max_dq)

        q_des = q + dq
        q_des = np.clip(q_des, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

        self.data.ctrl[:7] = q_des
        self.data.ctrl[7] = gripper
        self.sim.step(1)

        return err_norm

    def goto(self, target_pos: np.ndarray, gripper: float, max_steps: int, tol: float | None = None) -> int:
        if tol is None:
            tol = self.cfg.tol
        steps = 0
        for _ in range(max_steps):
            steps += 1
            err = self.step_towards(target_pos, gripper)
            if err <= tol:
                break
        return steps


def diagnostics(sim: Sim, label: str) -> None:
    pos = sim.block_positions()
    dx, dy, dz = pos["red"] - pos["green"]
    print(f"[{label}] dx={dx:+.4f} dy={dy:+.4f} dz={dz:+.4f} t={sim.data.time:.3f} steps={len(sim._ctrl_trace)}")


def run(seed: int = 0, save_path: str = "/work/final_state.npz") -> None:
    np.random.seed(seed)
    sim = Sim()
    ik = IK(sim, IKConfig())

    def hold(n: int, gripper: float) -> None:
        # Hold current arm targets to avoid sudden jumps.
        sim.data.ctrl[:7] = sim.data.qpos[:7]
        sim.data.ctrl[7] = gripper
        sim.step(n)

    # --- High-level scripted sequence ---
    # Open gripper.
    hold(100, gripper=255.0)

    # Targets relative to blocks.
    red = sim.block_positions()["red"]
    green = sim.block_positions()["green"]

    above_z = 0.60
    grasp_z = 0.455
    lift_z = 0.66

    # Approach above red.
    ik.goto(np.array([red[0], red[1], above_z]), gripper=255.0, max_steps=1400, tol=0.006)
    # Descend to grasp.
    ik.goto(np.array([red[0], red[1], grasp_z]), gripper=255.0, max_steps=900, tol=0.004)
    # Close gripper and settle briefly.
    hold(50, gripper=80.0)
    hold(350, gripper=0.0)

    # Lift.
    ik.goto(np.array([red[0], red[1], lift_z]), gripper=0.0, max_steps=1000, tol=0.006)

    # Move above green.
    green = sim.block_positions()["green"]
    ik.goto(np.array([green[0], green[1], lift_z]), gripper=0.0, max_steps=1700, tol=0.007)

    # Descend for placement; stop based on red height relative to green.
    target_place_xy = np.array([green[0], green[1]])
    place_z = 0.58
    for _ in range(1200):
        cur_green = sim.block_positions()["green"]
        cur_red = sim.block_positions()["red"]
        desired_dz = 0.055
        # Bias slightly high to reduce pushing the green block.
        if (cur_red[2] - cur_green[2]) <= (desired_dz + 0.006):
            break
        place_z = max(place_z - 4e-5, 0.46)
        place_target = np.array([target_place_xy[0], target_place_xy[1], place_z])
        ik.step_towards(place_target, gripper=0.0)

    # Open to release.
    hold(60, gripper=120.0)
    hold(220, gripper=255.0)

    # Retreat upward and slightly back-left to avoid bumping the stack.
    retreat = np.array([green[0] - 0.10, green[1] - 0.12, lift_z])
    ik.goto(retreat, gripper=255.0, max_steps=1600, tol=0.01)

    # Let the scene settle while holding position.
    hold(900, gripper=255.0)

    diagnostics(sim, "final")
    sim.save_final_state(save_path)
    print(f"Saved {os.path.abspath(save_path)}")


if __name__ == "__main__":
    run()
