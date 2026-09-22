import math
import os
import numpy as np
import mujoco

from sim import Sim


HAND_BODY = "hand"


class Controller:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    def hand_pos(self):
        return np.array(self.data.xpos[self.hand_id])

    def pinch_pos(self):
        return 0.5 * (
            np.array(self.data.xpos[self.left_finger_id]) + np.array(self.data.xpos[self.right_finger_id])
        )

    def solve_ik(self, target, q_init=None, iterations=20):
        q = self.data.qpos[:7].copy() if q_init is None else q_init.copy()
        for _ in range(iterations):
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)
            err = target - self.hand_pos()
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_id)
            j = jacp[:, :7]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-3 * np.eye(3), err)
            q = np.clip(q + 0.8 * dq, self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])
            if np.linalg.norm(err) < 1e-4:
                break
        return q

    def drive_hand(self, target, steps, grip, q_hint=None):
        q_target = self.solve_ik(np.array(target, dtype=float), q_init=q_hint, iterations=60)
        start = self.data.ctrl[:7].copy()
        for i in range(steps):
            alpha = (i + 1) / steps
            self.data.ctrl[:7] = (1 - alpha) * start + alpha * q_target
            self.data.ctrl[7] = grip
            self.sim.step()
        return q_target

    def hold(self, steps, grip):
        q_target = self.data.ctrl[:7].copy()
        for _ in range(steps):
            self.data.ctrl[:7] = q_target
            self.data.ctrl[7] = grip
            self.sim.step()


def replay_metrics(ctrl_trace):
    sim = Sim()
    best_x = math.inf
    best_z = -math.inf
    touched = False
    for ctrl in ctrl_trace:
        sim.data.ctrl[:] = ctrl
        sim.step()
        cup = sim.cup_position()
        best_x = min(best_x, float(cup[0]))
        best_z = max(best_z, float(cup[2]))
        touched = touched or sim.has_gripper_cup_contact()
    settle_contacts = 0
    for _ in range(500):
        sim.step()
        settle_contacts += float(sim.has_gripper_cup_contact())
    final_cup = sim.cup_position().copy()
    return {
        "replay_steps": len(ctrl_trace),
        "best_x": best_x,
        "best_z": best_z,
        "touched": touched,
        "final_cup": final_cup,
        "settle_contact_fraction": settle_contacts / 500.0,
    }


def run_attempt(save_path="/work/final_state.npz"):
    sim = Sim()
    ctl = Controller(sim)

    cup = sim.cup_position().copy()
    pre_above = np.array([0.58, 0.15, 0.62])
    over_cup = np.array([0.60, 0.15, 0.50])
    grasp_pose = np.array([0.60, 0.15, 0.46])
    lift_pose = np.array([0.56, 0.12, 0.68])
    return_pose = np.array([0.38, 0.02, 0.72])
    settle_pose = np.array([0.34, 0.00, 0.74])

    q = ctl.drive_hand(pre_above, steps=180, grip=255)
    q = ctl.drive_hand(over_cup, steps=120, grip=255, q_hint=q)
    q = ctl.drive_hand(grasp_pose, steps=140, grip=255, q_hint=q)
    ctl.hold(40, grip=255)
    ctl.hold(80, grip=10)
    ctl.hold(80, grip=0)
    q = ctl.drive_hand(lift_pose, steps=220, grip=0, q_hint=q)
    q = ctl.drive_hand(return_pose, steps=260, grip=0, q_hint=q)
    q = ctl.drive_hand(settle_pose, steps=180, grip=0, q_hint=q)
    ctl.hold(300, grip=0)

    sim.save_final_state(save_path)
    metrics = replay_metrics(np.array(sim._ctrl_trace))
    metrics["cup_end"] = sim.cup_position().copy()
    metrics["cup_start"] = cup
    return metrics


if __name__ == "__main__":
    metrics = run_attempt()
    print(metrics)
