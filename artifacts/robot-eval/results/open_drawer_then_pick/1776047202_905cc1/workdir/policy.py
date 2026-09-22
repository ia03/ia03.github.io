"""Heuristic controller for 'Open Drawer Then Pick' task.

Runs a staged script:
1) Open drawer via dedicated actuator.
2) Move hand above the presented block.
3) Descend, close gripper, and lift.
4) Hold steady for stable settle.

Writes /work/final_state.npz via sim.save_final_state().
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


@dataclass
class IKConfig:
    joint_count: int = 7
    damping: float = 0.06
    step_size: float = 0.65
    max_step_norm: float = 0.35
    tol: float = 0.004


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _joint_ctrl_ranges(sim: Sim) -> tuple[np.ndarray, np.ndarray]:
    lo = np.zeros(7, dtype=float)
    hi = np.zeros(7, dtype=float)
    for i in range(7):
        lo[i] = float(sim.model.actuator_ctrlrange[i, 0])
        hi[i] = float(sim.model.actuator_ctrlrange[i, 1])
    return lo, hi


def _dls_step(sim: Sim, J: np.ndarray, err: np.ndarray, q_des: np.ndarray, cfg: IKConfig) -> tuple[np.ndarray, float]:
    """Shared damped-least-squares solver step."""
    err_norm = float(np.linalg.norm(err))

    lam = cfg.damping
    JJt = J @ J.T
    A = JJt + (lam * lam) * np.eye(3)
    dq = J.T @ np.linalg.solve(A, err)

    step = cfg.step_size * dq
    step_norm = float(np.linalg.norm(step))
    if step_norm > cfg.max_step_norm:
        step *= cfg.max_step_norm / (step_norm + 1e-9)

    q_new = q_des.copy()
    q_cur = sim.data.qpos[: cfg.joint_count].copy()
    q_new[: cfg.joint_count] = q_cur + step[: cfg.joint_count]
    lo, hi = _joint_ctrl_ranges(sim)
    q_new[: cfg.joint_count] = np.clip(q_new[: cfg.joint_count], lo, hi)
    return q_new, err_norm


def ik_step_hand_to_target(sim: Sim, target_pos: np.ndarray, q_des: np.ndarray, cfg: IKConfig) -> tuple[np.ndarray, float]:
    """One damped-least-squares IK update toward target_pos for the hand body."""
    body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cur_pos = sim.data.xpos[body_id].copy()
    err = (target_pos - cur_pos).astype(float)

    # Jacobian wrt all DoFs; we only use the first 7 arm joints.
    jacp = np.zeros((3, sim.model.nv), dtype=float)
    jacr = np.zeros((3, sim.model.nv), dtype=float)
    mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, body_id)
    J = jacp[:, : cfg.joint_count]
    return _dls_step(sim, J, err, q_des, cfg)


def ik_step_finger_center_to_target(sim: Sim, target_center: np.ndarray, q_des: np.ndarray, cfg: IKConfig) -> tuple[np.ndarray, float]:
    """IK update driving the average (left_finger, right_finger) body position to target_center."""
    lf_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    rf_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    lf = sim.data.xpos[lf_id].copy()
    rf = sim.data.xpos[rf_id].copy()
    center = 0.5 * (lf + rf)
    err = (target_center - center).astype(float)

    jacp_l = np.zeros((3, sim.model.nv), dtype=float)
    jacr_l = np.zeros((3, sim.model.nv), dtype=float)
    jacp_r = np.zeros((3, sim.model.nv), dtype=float)
    jacr_r = np.zeros((3, sim.model.nv), dtype=float)
    mujoco.mj_jacBody(sim.model, sim.data, jacp_l, jacr_l, lf_id)
    mujoco.mj_jacBody(sim.model, sim.data, jacp_r, jacr_r, rf_id)
    J = 0.5 * (jacp_l[:, : cfg.joint_count] + jacp_r[:, : cfg.joint_count])
    return _dls_step(sim, J, err, q_des, cfg)


def drive_ik(sim: Sim, target_pos: np.ndarray, q_des: np.ndarray, steps: int, cfg: IKConfig, *, drawer_ctrl: float, grip_ctrl: float) -> np.ndarray:
    """Run closed-loop IK for a number of sim steps (finger-center)."""
    reached = 0
    for _ in range(steps):
        q_des, err_norm = ik_step_finger_center_to_target(sim, target_pos, q_des, cfg)
        sim.data.ctrl[:7] = q_des[:7]
        sim.data.ctrl[7] = grip_ctrl
        sim.data.ctrl[8] = drawer_ctrl
        sim.step(1)
        if err_norm < cfg.tol:
            reached += 1
            if reached >= 20:
                break
        else:
            reached = 0
    return q_des


def run(seed: int = 0) -> None:
    np.random.seed(seed)
    sim = Sim()
    cfg = IKConfig()

    # Controls: 0..6 joints, 7 gripper (0=closed, 255=open), 8 drawer motor (-1..1 force-ish)
    q_des = sim.data.qpos[:7].copy()
    sim.data.ctrl[:] = 0.0
    sim.data.ctrl[:7] = q_des
    sim.data.ctrl[7] = 255.0  # open gripper
    sim.data.ctrl[8] = 0.0

    # Stage 1: open drawer fully.
    for _ in range(600):
        sim.data.ctrl[:7] = q_des
        sim.data.ctrl[7] = 255.0
        sim.data.ctrl[8] = 1.0
        sim.step(1)
    # Hold the drawer open with a small positive command.
    drawer_hold = 0.25
    for _ in range(200):
        sim.data.ctrl[:7] = q_des
        sim.data.ctrl[7] = 255.0
        sim.data.ctrl[8] = drawer_hold
        sim.step(1)

    # Mandatory early save: at least we have the drawer genuinely opened.
    sim.save_final_state("/work/final_state.npz")

    # Get a stable estimate of where the block ended up after opening.
    for _ in range(100):
        sim.data.ctrl[:7] = q_des
        sim.data.ctrl[7] = 255.0
        sim.data.ctrl[8] = drawer_hold
        sim.step(1)
    block_pos = sim.block_position().copy()

    # Stage 2: move above the block (top-down approach).
    xy_backoff = np.array([-0.030, 0.0, 0.0], dtype=float)
    # Drive finger-center above the block (avoid roof/cabinet geometry).
    above_center = block_pos + xy_backoff + np.array([0.00, 0.00, 0.26], dtype=float)
    q_des = drive_ik(sim, above_center, q_des, steps=1100, cfg=cfg, drawer_ctrl=drawer_hold, grip_ctrl=255.0)

    # Stage 3: descend to grasp height (hand is ~5.8cm above fingertips in this model).
    block_pos = sim.block_position().copy()
    grasp_center = block_pos + xy_backoff + np.array([0.00, 0.00, 0.015], dtype=float)
    q_des = drive_ik(sim, grasp_center, q_des, steps=1500, cfg=cfg, drawer_ctrl=drawer_hold, grip_ctrl=255.0)

    # Stage 4: close gripper while holding pose.
    block_pos = sim.block_position().copy()
    grasp_center = block_pos + xy_backoff + np.array([0.00, 0.00, 0.015], dtype=float)
    for t in range(160):
        # Re-IK while closing to keep centered on the (possibly moved) block.
        block_pos = sim.block_position().copy()
        grasp_center = block_pos + xy_backoff + np.array([0.00, 0.00, 0.015], dtype=float)
        q_des, _ = ik_step_finger_center_to_target(sim, grasp_center, q_des, cfg)
        frac = (t + 1) / 160.0
        grip = 255.0 * (1.0 - frac)  # 255=open, 0=closed
        sim.data.ctrl[:7] = q_des
        sim.data.ctrl[7] = grip
        sim.data.ctrl[8] = drawer_hold
        sim.step(1)

    # Let the grasp "seat" before lifting.
    for _ in range(200):
        block_pos = sim.block_position().copy()
        grasp_center = block_pos + xy_backoff + np.array([0.00, 0.00, 0.015], dtype=float)
        q_des, _ = ik_step_finger_center_to_target(sim, grasp_center, q_des, cfg)
        sim.data.ctrl[:7] = q_des
        sim.data.ctrl[7] = 0.0
        sim.data.ctrl[8] = drawer_hold
        sim.step(1)

    # Quick diagnostics after closing.
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    lf_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    rf_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    bpos = sim.block_position().copy()
    hpos = sim.data.xpos[hand_id].copy()
    lfpos = sim.data.xpos[lf_id].copy()
    rfpos = sim.data.xpos[rf_id].copy()
    print("after_close block:", np.round(bpos, 4))
    print("after_close hand :", np.round(hpos, 4))
    print("after_close lf/rf:", np.round(lfpos, 4), np.round(rfpos, 4))
    print("after_close dz fingers-block:", float(lfpos[2] - bpos[2]), float(rfpos[2] - bpos[2]))
    print("after_close dxy fingers-block:", float(np.linalg.norm(lfpos[:2] - bpos[:2])), float(np.linalg.norm(rfpos[:2] - bpos[:2])))

    # Stage 5: lift well above the threshold.
    lift_target = sim.block_position().copy() + xy_backoff + np.array([0.00, 0.00, 0.34], dtype=float)
    lift_target[2] = max(lift_target[2], 0.72)
    q_des = drive_ik(sim, lift_target, q_des, steps=1200, cfg=cfg, drawer_ctrl=drawer_hold, grip_ctrl=0.0)

    # Stage 6: hold steady to make the final state stable and keep contact.
    for _ in range(1200):
        sim.data.ctrl[:7] = q_des
        sim.data.ctrl[7] = 0.0
        sim.data.ctrl[8] = drawer_hold
        sim.step(1)

    sim.save_final_state("/work/final_state.npz")

    print("final drawer_open:", sim.drawer_open_amount())
    print("final block_pos:", sim.block_position())
    print("final block_contact:", sim.has_gripper_block_contact())
    print("ctrl_trace_len:", len(sim._ctrl_trace))


if __name__ == "__main__":
    run()
