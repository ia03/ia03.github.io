"""Heuristic controller for: open drawer then pick and lift block.

Saves /work/final_state.npz containing ctrl_trace for grading.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import mujoco

from sim import Sim


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, mat.reshape(9))
    # Ensure a consistent sign (positive w) for smoother error.
    if q[0] < 0:
        q *= -1
    return q


def _body_xmat(m: mujoco.MjModel, d: mujoco.MjData, body_id: int) -> np.ndarray:
    return d.xmat[body_id].reshape(3, 3).copy()


def _body_xpos(m: mujoco.MjModel, d: mujoco.MjData, body_id: int) -> np.ndarray:
    return d.xpos[body_id].copy()


@dataclass
class IKSolver:
    model: mujoco.MjModel
    data: mujoco.MjData
    body_id: int
    dof_ids: np.ndarray  # indices into qpos for the dofs we move (here 0..6)
    qpos_ids: np.ndarray
    lam: float = 0.06

    def solve_step(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        step_scale: float = 0.7,
        rot_weight: float = 0.35,
        q_nominal: np.ndarray | None = None,
        null_weight: float = 0.08,
    ) -> np.ndarray:
        """One damped-least-squares IK step returning dq for qpos_ids."""
        m, d = self.model, self.data
        cur_pos = _body_xpos(m, d, self.body_id)
        cur_quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(cur_quat, d.xmat[self.body_id])
        if cur_quat[0] < 0:
            cur_quat *= -1

        pos_err = target_pos - cur_pos

        # Quaternion error -> axis-angle approx (small-angle): 2 * q_xyz when w ~ 1.
        q_err = _quat_mul(target_quat, _quat_conj(cur_quat))
        if q_err[0] < 0:
            q_err *= -1
        rot_err = 2.0 * q_err[1:4]

        err6 = np.concatenate([pos_err, rot_weight * rot_err], axis=0)

        jacp = np.zeros((3, m.nv), dtype=float)
        jacr = np.zeros((3, m.nv), dtype=float)
        mujoco.mj_jacBody(m, d, jacp, jacr, self.body_id)
        # Select columns for the arm dofs (0..6).
        J = np.vstack([jacp[:, self.dof_ids], rot_weight * jacr[:, self.dof_ids]])  # 6x7

        # DLS: dq = J^T (J J^T + lam^2 I)^-1 err
        A = J @ J.T + (self.lam**2) * np.eye(6)
        J_pinv = J.T @ np.linalg.solve(A, np.eye(6))
        dq = J_pinv @ err6

        if q_nominal is not None and null_weight > 0:
            # Bias toward a nominal posture in the Jacobian nullspace.
            # N = I - J^T (J J^T + lam^2 I)^-1 J
            N = np.eye(J.shape[1]) - J_pinv @ J
            q_cur = self.data.qpos[self.qpos_ids].copy()
            dq = dq + null_weight * (N @ (q_nominal - q_cur))
        dq = np.clip(dq, -0.12, 0.12)
        return step_scale * dq


def _set_arm_ctrl(sim: Sim, q: np.ndarray):
    sim.data.ctrl[0:7] = q


def _set_gripper_open(sim: Sim, open_amount: float):
    # actuator8 is tendon, ctrlrange [0,255]. 255 is open, 0 is closed.
    sim.data.ctrl[7] = float(np.clip(open_amount, 0.0, 255.0))


def _set_drawer_force(sim: Sim, u: float):
    sim.data.ctrl[8] = float(np.clip(u, -1.0, 1.0))


def run_episode(sim: Sim, render_every: int = 0) -> dict:
    m, d = sim.model, sim.data
    hand_id = m.body("hand").id
    dof_ids = np.arange(0, 7, dtype=int)  # arm dofs in qvel
    qpos_ids = np.arange(0, 7, dtype=int)
    ik = IKSolver(m, d, hand_id, dof_ids=dof_ids, qpos_ids=qpos_ids)

    # Nominal downward-ish orientation at reset.
    R0 = _body_xmat(m, d, hand_id)
    q_des = _mat_to_quat(R0)

    # A more forward-reaching seed (found via random search) helps avoid "over-the-roof" IK solutions.
    q_seed = np.array(
        [-1.29, -0.96, 1.20, -2.54, 1.14, 2.96, 1.40],
        dtype=float,
    )
    q_seed = np.clip(q_seed, m.actuator_ctrlrange[0:7, 0], m.actuator_ctrlrange[0:7, 1])

    q_home = d.qpos[0:7].copy()
    _set_arm_ctrl(sim, q_home)
    _set_gripper_open(sim, 255.0)
    _set_drawer_force(sim, 0.0)

    frames = []

    def maybe_render():
        if render_every and (len(sim._ctrl_trace) % render_every == 0):
            frames.append(sim.render(width=320, height=240))

    # Phase 1: move to seed posture while opening drawer.
    for t in range(900):
        alpha = min(1.0, (t + 1) / 650.0)
        q_cmd = (1.0 - alpha) * q_home + alpha * q_seed
        _set_arm_ctrl(sim, q_cmd)
        _set_gripper_open(sim, 255.0)
        _set_drawer_force(sim, 1.0 if sim.drawer_open_amount() < 0.12 else 0.18)
        sim.step(1)
        maybe_render()
        if sim.drawer_open_amount() >= 0.12 and alpha >= 1.0:
            break

    # Hold drawer open with a small positive force.
    hold_drawer_u = 0.22

    # Phase 2: open drawer fully and wait until the block rides out past the cabinet roof.
    for _ in range(1200):
        _set_arm_ctrl(sim, q_seed)
        _set_gripper_open(sim, 255.0)
        _set_drawer_force(sim, 1.0)
        sim.step(1)
        maybe_render()
        if sim.drawer_open_amount() >= 0.155 and sim.block_position()[0] >= 0.71:
            break

    # From here, the block should be outside the roofed region, so a top-down grasp is feasible.

    # Phase 3: move above block.
    for _ in range(1100):
        block = sim.block_position()
        target = block + np.array([0.00, 0.00, 0.22])

        q = d.qpos[0:7].copy()
        dq = ik.solve_step(target, q_des, step_scale=0.75, rot_weight=0.30, q_nominal=q_seed)
        q = q + dq
        # Respect joint limits via actuator ctrlrange.
        q = np.clip(q, m.actuator_ctrlrange[0:7, 0], m.actuator_ctrlrange[0:7, 1])

        _set_arm_ctrl(sim, q)
        _set_gripper_open(sim, 255.0)
        _set_drawer_force(sim, hold_drawer_u)
        sim.step(1)
        maybe_render()

        if np.linalg.norm(_body_xpos(m, d, hand_id) - target) < 0.02:
            break

    # Phase 4: descend to grasp height.
    for _ in range(1300):
        block = sim.block_position()
        target = block + np.array([0.00, 0.00, 0.075])

        q = d.qpos[0:7].copy()
        dq = ik.solve_step(target, q_des, step_scale=0.65, rot_weight=0.28, q_nominal=q_seed)
        q = q + dq
        q = np.clip(q, m.actuator_ctrlrange[0:7, 0], m.actuator_ctrlrange[0:7, 1])

        _set_arm_ctrl(sim, q)
        _set_gripper_open(sim, 255.0)
        _set_drawer_force(sim, hold_drawer_u)
        sim.step(1)
        maybe_render()

        if np.linalg.norm(_body_xpos(m, d, hand_id) - target) < 0.014:
            break

    # Phase 5: close gripper while nudging down for contact.
    for t in range(750):
        block = sim.block_position()
        y_wiggle = 0.015 * math.sin(2.0 * math.pi * (t / 80.0))
        target = block + np.array([0.00, y_wiggle, 0.045])

        q = d.qpos[0:7].copy()
        dq = ik.solve_step(target, q_des, step_scale=0.55, rot_weight=0.25, q_nominal=q_seed)
        q = q + dq
        q = np.clip(q, m.actuator_ctrlrange[0:7, 0], m.actuator_ctrlrange[0:7, 1])

        _set_arm_ctrl(sim, q)
        # Ramp close (255 open -> 0 closed).
        _set_gripper_open(sim, 255.0 * (1.0 - t / 750.0) ** 2)
        _set_drawer_force(sim, hold_drawer_u)
        sim.step(1)
        maybe_render()

    # Phase 6: lift block.
    lift_target_z = 0.66
    for _ in range(2600):
        block = sim.block_position()
        target = block.copy()
        target[2] = max(block[2], lift_target_z)
        target += np.array([0.00, 0.00, 0.12])

        q = d.qpos[0:7].copy()
        dq = ik.solve_step(target, q_des, step_scale=0.75, rot_weight=0.28, q_nominal=q_seed)
        q = q + dq
        q = np.clip(q, m.actuator_ctrlrange[0:7, 0], m.actuator_ctrlrange[0:7, 1])

        _set_arm_ctrl(sim, q)
        _set_gripper_open(sim, 0.0)
        _set_drawer_force(sim, hold_drawer_u)
        sim.step(1)
        maybe_render()

        if sim.block_position()[2] >= 0.61:
            break

    # Phase 7: hold steady for a bit to let things settle (drawer still open, keep grip).
    q_hold = d.qpos[0:7].copy()
    for _ in range(1000):
        _set_arm_ctrl(sim, q_hold)
        _set_gripper_open(sim, 0.0)
        _set_drawer_force(sim, hold_drawer_u)
        sim.step(1)
        maybe_render()

    info = {
        "steps": len(sim._ctrl_trace),
        "drawer_open": sim.drawer_open_amount(),
        "block_pos": sim.block_position().copy(),
        "has_contact": sim.has_gripper_block_contact(),
        "frames": frames,
    }
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="/work/final_state.npz")
    ap.add_argument("--render-every", type=int, default=0)
    args = ap.parse_args()

    sim = Sim()
    info = run_episode(sim, render_every=args.render_every)
    sim.save_final_state(args.save)
    print("saved:", args.save)
    print("steps:", info["steps"])
    print("drawer_open:", info["drawer_open"])
    print("block_pos:", info["block_pos"])
    print("contact:", info["has_contact"])


if __name__ == "__main__":
    main()
