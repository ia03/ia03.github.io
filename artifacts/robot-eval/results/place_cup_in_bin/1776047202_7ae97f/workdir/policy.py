"""Heuristic controller to pick and place the cup into the bin.

Produces /work/final_state.npz via sim.save_final_state(). The grader replays the
saved ctrl_trace from the canonical initial state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import BIN_CENTER, CUP_INIT_POS, Sim


def _rotation_error_vec(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    # Orientation error in world frame.
    # See e.g. "A Mathematical Introduction to Robotic Manipulation" (Murray et al.).
    return 0.5 * (
        np.cross(R_cur[:, 0], R_des[:, 0])
        + np.cross(R_cur[:, 1], R_des[:, 1])
        + np.cross(R_cur[:, 2], R_des[:, 2])
    )


@dataclass
class IKGains:
    kp_pos: float = 4.0
    damping: float = 0.12
    max_dq: float = 0.030  # rad per sim step
    k_null: float = 0.05


class PandaPolicy:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self.hand_id < 0:
            raise RuntimeError("Could not find body 'hand'")
        self.left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        if self.left_finger_id < 0 or self.right_finger_id < 0:
            raise RuntimeError("Could not find finger bodies")

        self.arm_act_ctrlrange = self.model.actuator_ctrlrange[:7].copy()
        self._arm_lo = self.arm_act_ctrlrange[:, 0].copy()
        self._arm_hi = self.arm_act_ctrlrange[:, 1].copy()
        self.des_q = np.clip(self.data.qpos[:7].copy(), self._arm_lo, self._arm_hi)

        self.gains = IKGains()
        self.q_bias = self.des_q.copy()

        # Stats
        self.best_cup_z = float(self.sim.cup_position()[2])
        self.best_contact = 0.0
        self._contact_steps = 0
        self._total_steps = 0

    def _set_ctrl(self, q_target: np.ndarray, gripper_ctrl: float):
        self.data.ctrl[:7] = np.clip(np.asarray(q_target, dtype=float), self._arm_lo, self._arm_hi)
        self.data.ctrl[7] = float(np.clip(gripper_ctrl, 0.0, 255.0))

    def _ik_delta(self, target_pos: np.ndarray) -> np.ndarray:
        m, d = self.model, self.data
        # Track the midpoint between fingertips; targeting the 'hand' origin is
        # noticeably offset from the actual grasp point.
        jacp_l = np.zeros((3, m.nv), dtype=float)
        jacr_l = np.zeros((3, m.nv), dtype=float)
        mujoco.mj_jacBody(m, d, jacp_l, jacr_l, self.left_finger_id)
        jacp_r = np.zeros((3, m.nv), dtype=float)
        jacr_r = np.zeros((3, m.nv), dtype=float)
        mujoco.mj_jacBody(m, d, jacp_r, jacr_r, self.right_finger_id)
        J = 0.5 * (jacp_l[:, :7] + jacp_r[:, :7])  # 3x7

        eef_pos = 0.5 * (d.xpos[self.left_finger_id] + d.xpos[self.right_finger_id])
        pos_err = target_pos - eef_pos
        pos_err = np.clip(pos_err, -0.08, 0.08)
        task = self.gains.kp_pos * pos_err

        A = J @ J.T + (self.gains.damping**2) * np.eye(3)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(3))  # 7x3
        dq_task = J_pinv @ task

        # Bias toward a reasonable posture, but use the *current* joint angles
        # to avoid integrator windup and oscillations.
        q_cur = d.qpos[:7].copy()
        N = np.eye(7) - J_pinv @ J
        dq_null = self.gains.k_null * (self.q_bias - q_cur)
        dq = dq_task + N @ dq_null
        return np.clip(dq, -self.gains.max_dq, self.gains.max_dq)

    def step_towards(self, target_pos, gripper_ctrl: float):
        dq = self._ik_delta(np.asarray(target_pos, dtype=float))
        q_target = self.data.qpos[:7].copy() + dq
        self.des_q = np.clip(q_target, self._arm_lo, self._arm_hi)
        self._set_ctrl(self.des_q, gripper_ctrl)
        self.sim.step(1)

        cup_z = float(self.sim.cup_position()[2])
        self.best_cup_z = max(self.best_cup_z, cup_z)
        contact = float(self.sim.has_gripper_cup_contact())
        self.best_contact = max(self.best_contact, contact)
        self._contact_steps += int(contact > 0.5)
        self._total_steps += 1

    def run_segment(self, target_pos, gripper_ctrl: float, steps: int):
        for _ in range(int(steps)):
            self.step_towards(target_pos, gripper_ctrl)

    def hold_joint_pose(self, q: np.ndarray, gripper_ctrl: float, steps: int):
        self.des_q = np.clip(np.asarray(q, dtype=float).copy(), self._arm_lo, self._arm_hi)
        self.q_bias = self.des_q.copy()
        for _ in range(int(steps)):
            self._set_ctrl(self.des_q, gripper_ctrl)
            self.sim.step(1)

    def contact_fraction_recent(self, n: int = 200) -> float:
        # Uses the internal trace, sampled every TRACE_EVERY_STEPS.
        if not self.sim._trace:
            return 0.0
        arr = np.array([e["cup_contact"] for e in self.sim._trace], dtype=float)
        if arr.size == 0:
            return 0.0
        return float(np.mean(arr[-max(1, min(arr.size, n)) :]))


def main():
    t0 = time.time()
    sim = Sim()
    pol = PandaPolicy(sim)

    cup = np.array(CUP_INIT_POS, dtype=float)
    bin_center = np.array(BIN_CENTER, dtype=float)

    # Drive to a sane start pose inside joint limits (joint4 must be negative).
    home_q = np.array([0.0, -0.55, 0.0, -2.20, 0.0, 2.00, 0.80], dtype=float)
    pol.hold_joint_pose(home_q, gripper_ctrl=255.0, steps=700)

    # Targets (hand body position targets in world frame).
    high_cup = cup + np.array([0.0, 0.0, 0.36])
    above_cup = cup + np.array([0.0, 0.0, 0.20])
    pregrasp = cup + np.array([0.0, 0.0, 0.10])
    grasp = cup + np.array([0.0, 0.0, 0.00])
    lift = cup + np.array([0.0, 0.0, 0.26])  # ensure z>=0.50 at some point
    high_bin = bin_center + np.array([0.0, 0.0, 0.36])
    above_bin = bin_center + np.array([0.0, 0.0, 0.24])
    place = bin_center + np.array([0.0, 0.0, 0.05])
    retreat = np.array([0.50, 0.0, 0.70], dtype=float)

    # 1) Approach with gripper open.
    pol.run_segment(high_cup, gripper_ctrl=255.0, steps=1000)
    pol.run_segment(above_cup, gripper_ctrl=255.0, steps=750)
    pol.run_segment(pregrasp, gripper_ctrl=255.0, steps=550)

    # Save a plausible early attempt (mandatory).
    sim.save_final_state("/work/final_state.npz")

    # 2) Descend to grasp.
    pol.run_segment(grasp, gripper_ctrl=255.0, steps=520)

    # 3) Close gripper and stabilize.
    pol.run_segment(grasp, gripper_ctrl=0.0, steps=700)

    # 4) Lift.
    pol.run_segment(lift, gripper_ctrl=0.0, steps=850)

    # 5) Transport over bin.
    pol.run_segment(high_bin, gripper_ctrl=0.0, steps=1200)
    pol.run_segment(above_bin, gripper_ctrl=0.0, steps=850)

    # 6) Lower into bin.
    pol.run_segment(place, gripper_ctrl=0.0, steps=950)

    # 7) Release and retreat to break contact.
    pol.run_segment(place, gripper_ctrl=255.0, steps=700)
    pol.run_segment(retreat, gripper_ctrl=255.0, steps=900)

    sim.save_final_state("/work/final_state.npz")

    cup_pos = sim.cup_position()
    dt = time.time() - t0
    print(
        "done",
        f"wall={dt:.2f}s",
        f"sim_t={sim.data.time:.3f}s",
        f"best_cup_z={pol.best_cup_z:.3f}",
        f"final_cup={cup_pos}",
        f"recent_contact_frac={pol.contact_fraction_recent():.3f}",
    )


if __name__ == "__main__":
    main()
