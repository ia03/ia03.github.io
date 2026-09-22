import itertools
import math
import time

import mujoco
import numpy as np

from sim import Sim


ARM_DOF = 7
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0
DRAWER_OPEN_CTRL = 1.0
DRAWER_HOLD_CTRL = 0.35


def orientation_error(current, target):
    return 0.5 * sum(np.cross(current[:, i], target[:, i]) for i in range(3))


class HandController:
    def __init__(self, sim):
        self.sim = sim
        self.hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.home_q = sim.data.qpos[:ARM_DOF].copy()
        self.home_rot = sim.data.xmat[self.hand_id].reshape(3, 3).copy()
        self.jacp = np.zeros((3, sim.model.nv))
        self.jacr = np.zeros((3, sim.model.nv))
        self.q_ranges = sim.model.actuator_ctrlrange[:ARM_DOF].copy()

    def step_toward(self, target_pos, grip_ctrl, drawer_ctrl, target_rot=None, pos_gain=8.0, rot_gain=2.5):
        if target_rot is None:
            target_rot = self.home_rot
        hand_pos = self.sim.data.xpos[self.hand_id].copy()
        hand_rot = self.sim.data.xmat[self.hand_id].reshape(3, 3).copy()
        pos_err = target_pos - hand_pos
        rot_err = orientation_error(hand_rot, target_rot)
        mujoco.mj_jacBody(self.sim.model, self.sim.data, self.jacp, self.jacr, self.hand_id)
        J = np.vstack([self.jacp[:, :ARM_DOF], self.jacr[:, :ARM_DOF]])
        task = np.concatenate([pos_gain * pos_err, rot_gain * rot_err])
        damp = 1e-3
        jj = J @ J.T + damp * np.eye(6)
        dq = J.T @ np.linalg.solve(jj, task)
        q = self.sim.data.qpos[:ARM_DOF].copy()
        q_target = q + 0.12 * dq + 0.01 * (self.home_q - q)
        self.sim.data.ctrl[:ARM_DOF] = np.clip(q_target, self.q_ranges[:, 0], self.q_ranges[:, 1])
        self.sim.data.ctrl[7] = float(np.clip(grip_ctrl, 0.0, 255.0))
        self.sim.data.ctrl[8] = float(np.clip(drawer_ctrl, -1.0, 1.0))
        self.sim.step(1)
        return np.linalg.norm(pos_err), np.linalg.norm(rot_err)

    def move_for_steps(self, steps, target_pos, grip_ctrl, drawer_ctrl, target_rot=None):
        for _ in range(steps):
            self.step_toward(target_pos, grip_ctrl, drawer_ctrl, target_rot=target_rot)


def replay_metrics(ctrl_trace):
    sim = Sim()
    max_drawer_open = 0.0
    max_block_z = -1e9
    first_open = None
    first_retrieval = None
    contacts = []
    for i, ctrl in enumerate(ctrl_trace):
        sim.data.ctrl[:] = ctrl
        sim.step(1)
        drawer = sim.drawer_open_amount()
        block_z = sim.block_position()[2]
        max_drawer_open = max(max_drawer_open, drawer)
        max_block_z = max(max_block_z, block_z)
        if first_open is None and drawer >= 0.05:
            first_open = i
        if first_retrieval is None and block_z >= 0.50:
            first_retrieval = i
    for _ in range(500):
        sim.step(1)
        contacts.append(1.0 if sim.has_gripper_block_contact() else 0.0)
    final_drawer = sim.drawer_open_amount()
    final_block_z = sim.block_position()[2]
    contact_fraction = float(np.mean(contacts)) if contacts else 0.0
    order_ok = first_open is not None and (first_retrieval is None or first_open < first_retrieval)
    success = (
        len(ctrl_trace) >= 20
        and order_ok
        and final_drawer >= 0.05
        and final_block_z >= 0.595
        and contact_fraction >= 0.4
    )
    drawer_progress = np.clip(max_drawer_open / 0.05, 0.0, 1.0)
    retrieval_progress = np.clip((max_block_z - 0.435) / (0.50 - 0.435), 0.0, 1.0)
    final_height_progress = np.clip((final_block_z - 0.43) / (0.595 - 0.43), 0.0, 1.0)
    hold_progress = np.clip(contact_fraction / 0.4, 0.0, 1.0)
    final_drawer_progress = np.clip(final_drawer / 0.05, 0.0, 1.0)
    progress_score = (
        0.22 * drawer_progress
        + 0.28 * (1.0 if order_ok else 0.0) * retrieval_progress
        + 0.22 * final_height_progress
        + 0.18 * hold_progress
        + 0.10 * final_drawer_progress
    )
    return {
        "success": success,
        "progress_score": float(progress_score),
        "max_drawer_open": float(max_drawer_open),
        "max_block_z": float(max_block_z),
        "final_drawer": float(final_drawer),
        "final_block_z": float(final_block_z),
        "contact_fraction": float(contact_fraction),
        "order_ok": bool(order_ok),
        "steps": len(ctrl_trace),
    }


def run_trial(params):
    sim = Sim()
    ctl = HandController(sim)

    block0 = sim.block_position().copy()
    waypoints = [
        np.array([0.35, 0.0, 0.72]),
        np.array([block0[0] + 0.02, 0.0, 0.62]),
    ]
    for target in waypoints:
        ctl.move_for_steps(90, target, GRIPPER_OPEN, DRAWER_OPEN_CTRL)

    for _ in range(120):
        sim.data.ctrl[:ARM_DOF] = sim.data.qpos[:ARM_DOF]
        sim.data.ctrl[7] = GRIPPER_OPEN
        sim.data.ctrl[8] = DRAWER_OPEN_CTRL
        sim.step(1)

    block = sim.block_position().copy()
    above = block + np.array([params["x_offset"], params["y_offset"], params["z_above"]])
    grasp = block + np.array([params["x_offset"], params["y_offset"], params["z_grasp"]])
    prelift = block + np.array([params["x_offset"], params["y_offset"], params["z_prelift"]])
    lift = block + np.array([params["x_offset"] + params["x_lift_bias"], params["y_offset"], params["z_lift"]])

    ctl.move_for_steps(180, above, GRIPPER_OPEN, DRAWER_HOLD_CTRL)
    ctl.move_for_steps(220, grasp, GRIPPER_OPEN, DRAWER_HOLD_CTRL)

    for i in range(params["close_steps"]):
        alpha = (i + 1) / params["close_steps"]
        target = grasp * (1.0 - alpha) + prelift * alpha
        grip = GRIPPER_OPEN * (1.0 - alpha)
        ctl.step_toward(target, grip, DRAWER_HOLD_CTRL)

    ctl.move_for_steps(80, prelift, GRIPPER_CLOSED, DRAWER_HOLD_CTRL)
    ctl.move_for_steps(220, lift, GRIPPER_CLOSED, DRAWER_HOLD_CTRL)
    ctl.move_for_steps(200, lift + np.array([0.0, 0.0, 0.03]), GRIPPER_CLOSED, DRAWER_HOLD_CTRL)
    metrics = replay_metrics(np.array(sim._ctrl_trace, dtype=float))
    return sim, metrics


def main():
    start = time.time()
    best = None
    trials = list(
        itertools.product(
            [-0.015, 0.0, 0.015],
            [0.0, 0.01, -0.01],
            [0.12, 0.14],
            [0.09, 0.10, 0.11],
            [0.11, 0.13],
            [0.23, 0.27],
            [0.0, -0.03],
            [90, 130],
        )
    )
    for idx, values in enumerate(trials, start=1):
        params = {
            "x_offset": values[0],
            "y_offset": values[1],
            "z_above": values[2],
            "z_grasp": values[3],
            "z_prelift": values[4],
            "z_lift": values[5],
            "x_lift_bias": values[6],
            "close_steps": values[7],
        }
        sim, metrics = run_trial(params)
        improved = best is None or metrics["progress_score"] > best[0]["progress_score"]
        print(f"trial {idx}/{len(trials)} params={params} metrics={metrics}")
        if improved:
            best = (metrics, params)
            sim.save_final_state("/work/final_state.npz")
            print("saved improved final_state")
        if metrics["success"]:
            print("success found")
            break
        if time.time() - start > 600:
            print("stopping search after 600s of compute")
            break
    if best is not None:
        print("best", best[0], best[1])


if __name__ == "__main__":
    main()
