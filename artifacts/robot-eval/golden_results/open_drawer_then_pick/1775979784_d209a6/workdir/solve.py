import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


ARM_DOF = 7
GRIP_OPEN = 255.0
GRIP_CLOSED = 0.0


@dataclass
class Metrics:
    replay_steps: int
    max_drawer_before_lift: float
    final_drawer: float
    max_block_z: float
    final_block_z: float
    settle_contact_fraction: float

    def score(self) -> float:
        return (
            4.0 * min(self.max_drawer_before_lift, 0.16) / 0.16
            + 4.0 * min(self.final_block_z, 0.70) / 0.70
            + 2.0 * self.settle_contact_fraction
        )

    def success(self) -> bool:
        return (
            self.replay_steps >= 20
            and self.max_drawer_before_lift >= 0.10
            and self.final_drawer >= 0.10
            and self.final_block_z >= 0.60
            and self.settle_contact_fraction >= 0.4
        )


class Controller:
    def __init__(self):
        self.sim = Sim()
        self.q_limits = self.sim.model.jnt_range[:ARM_DOF].copy()
        self.best_score = -1e9

    def pinch_center(self):
        return (self.sim.data.xpos[10].copy() + self.sim.data.xpos[11].copy()) / 2.0

    def pinch_jacobian(self):
        jac = np.zeros((3, self.sim.model.nv))
        for body_id in (10, 11):
            jacp = np.zeros((3, self.sim.model.nv))
            jacr = np.zeros((3, self.sim.model.nv))
            mujoco.mj_jacBodyCom(self.sim.model, self.sim.data, jacp, jacr, body_id)
            jac += jacp
        return jac[:, :ARM_DOF] / 2.0

    def ik_step(self, target, gain=0.55, damping=1e-3):
        pos = self.pinch_center()
        err = np.asarray(target, dtype=float) - pos
        J = self.pinch_jacobian()
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), err)
        q = self.sim.data.qpos[:ARM_DOF].copy()
        q = np.clip(q + gain * dq, self.q_limits[:, 0], self.q_limits[:, 1])
        return q, err

    def hold(self, steps, grip, target=None, gain=0.55):
        for _ in range(steps):
            if target is None:
                q = self.sim.data.qpos[:ARM_DOF].copy()
            else:
                q, _ = self.ik_step(target, gain=gain)
            self.sim.data.ctrl[:ARM_DOF] = q
            self.sim.data.ctrl[7] = grip
            self.sim.step(1)

    def move_to(self, target, grip, steps, gain=0.55):
        for _ in range(steps):
            q, _ = self.ik_step(target, gain=gain)
            self.sim.data.ctrl[:ARM_DOF] = q
            self.sim.data.ctrl[7] = grip
            self.sim.step(1)

    def move_line(self, start, end, grip, steps, gain=0.55):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        for i in range(steps):
            alpha = (i + 1) / steps
            target = (1.0 - alpha) * start + alpha * end
            q, _ = self.ik_step(target, gain=gain)
            self.sim.data.ctrl[:ARM_DOF] = q
            self.sim.data.ctrl[7] = grip
            self.sim.step(1)

    def evaluate(self, settle_steps=500):
        trace = self.sim._trace
        trace_drawer = np.array([item["drawer_open"] for item in trace], dtype=float)
        trace_block = np.array([item["block_pos"] for item in trace], dtype=float)
        before_lift = trace_drawer[trace_block[:, 2] < 0.50]
        max_drawer_before_lift = float(before_lift.max()) if len(before_lift) else 0.0

        settle_contacts = []
        for _ in range(settle_steps):
            self.sim.step(1)
            settle_contacts.append(float(self.sim.has_gripper_block_contact()))

        metrics = Metrics(
            replay_steps=len(self.sim._ctrl_trace),
            max_drawer_before_lift=max_drawer_before_lift,
            final_drawer=self.sim.drawer_open_amount(),
            max_block_z=float(trace_block[:, 2].max()),
            final_block_z=float(self.sim.block_position()[2]),
            settle_contact_fraction=float(np.mean(settle_contacts) if settle_contacts else 0.0),
        )
        return metrics

    def maybe_save(self, metrics):
        score = metrics.score()
        if score > self.best_score:
            self.best_score = score
            self.sim.save_final_state("/work/final_state.npz")
            print(
                "saved",
                {
                    "score": round(score, 3),
                    "drawer_before_lift": round(metrics.max_drawer_before_lift, 3),
                    "final_drawer": round(metrics.final_drawer, 3),
                    "max_block_z": round(metrics.max_block_z, 3),
                    "final_block_z": round(metrics.final_block_z, 3),
                    "contact": round(metrics.settle_contact_fraction, 3),
                    "success": metrics.success(),
                },
            )


def run_attempt(params):
    ctl = Controller()
    sim = ctl.sim

    handle = np.array([0.753, -0.020, 0.440])
    block = np.array([0.615, -0.020, 0.435])

    pre_handle = handle + np.array(params["pre_handle_offset"])
    grasp_handle = handle + np.array(params["grasp_handle_offset"])
    pull_end = handle + np.array(params["pull_offset"])

    pre_block = block + np.array(params["pre_block_offset"])
    grasp_block = block + np.array(params["grasp_block_offset"])
    lift_mid = block + np.array(params["lift_mid_offset"])
    lift_high = block + np.array(params["lift_high_offset"])
    settle_pose = block + np.array(params["settle_offset"])

    ctl.hold(40, GRIP_OPEN)
    ctl.move_to(pre_handle, GRIP_OPEN, params["pre_handle_steps"], gain=0.50)
    ctl.move_line(pre_handle, grasp_handle, params["handle_grip"], params["grasp_handle_steps"], gain=0.45)
    ctl.hold(params["close_handle_steps"], params["handle_grip"], target=grasp_handle, gain=0.40)
    ctl.move_line(grasp_handle, pull_end, params["handle_grip"], params["pull_steps"], gain=0.38)
    ctl.hold(params["hold_open_steps"], params["handle_grip"], target=pull_end, gain=0.35)

    partial = ctl.evaluate(settle_steps=50)
    ctl.maybe_save(partial)

    ctl.move_to(pre_block, GRIP_OPEN, params["pre_block_steps"], gain=0.50)
    ctl.move_line(pre_block, grasp_block, GRIP_OPEN, params["grasp_block_steps"], gain=0.42)
    ctl.hold(params["preclose_block_steps"], params["block_open_grip"], target=grasp_block, gain=0.35)
    ctl.hold(params["close_block_steps"], params["block_close_grip"], target=grasp_block, gain=0.30)
    ctl.move_line(grasp_block, lift_mid, params["block_close_grip"], params["lift1_steps"], gain=0.32)
    ctl.move_line(lift_mid, lift_high, params["block_close_grip"], params["lift2_steps"], gain=0.28)
    ctl.hold(params["stabilize_steps"], params["block_close_grip"], target=settle_pose, gain=0.22)

    metrics = ctl.evaluate(settle_steps=500)
    ctl.maybe_save(metrics)
    return metrics


PARAM_SETS = [
    {
        "name": "baseline",
        "pre_handle_offset": [0.00, 0.00, 0.10],
        "grasp_handle_offset": [0.00, 0.00, 0.005],
        "pull_offset": [0.15, 0.00, 0.005],
        "pre_block_offset": [0.02, 0.00, 0.12],
        "grasp_block_offset": [0.01, 0.00, 0.03],
        "lift_mid_offset": [0.03, 0.00, 0.16],
        "lift_high_offset": [0.06, 0.00, 0.28],
        "settle_offset": [0.08, 0.00, 0.30],
        "pre_handle_steps": 150,
        "grasp_handle_steps": 140,
        "close_handle_steps": 80,
        "pull_steps": 220,
        "hold_open_steps": 70,
        "pre_block_steps": 200,
        "grasp_block_steps": 140,
        "preclose_block_steps": 40,
        "close_block_steps": 120,
        "lift1_steps": 140,
        "lift2_steps": 180,
        "stabilize_steps": 160,
        "handle_grip": 25.0,
        "block_open_grip": 115.0,
        "block_close_grip": 8.0,
    },
    {
        "name": "deeper_grasp",
        "pre_handle_offset": [-0.01, 0.00, 0.10],
        "grasp_handle_offset": [-0.008, 0.00, 0.003],
        "pull_offset": [0.14, 0.00, -0.004],
        "pre_block_offset": [0.00, 0.00, 0.12],
        "grasp_block_offset": [0.00, 0.00, 0.025],
        "lift_mid_offset": [0.01, 0.00, 0.17],
        "lift_high_offset": [0.03, 0.00, 0.30],
        "settle_offset": [0.04, 0.00, 0.33],
        "pre_handle_steps": 150,
        "grasp_handle_steps": 150,
        "close_handle_steps": 90,
        "pull_steps": 250,
        "hold_open_steps": 80,
        "pre_block_steps": 170,
        "grasp_block_steps": 170,
        "preclose_block_steps": 50,
        "close_block_steps": 130,
        "lift1_steps": 150,
        "lift2_steps": 200,
        "stabilize_steps": 180,
        "handle_grip": 15.0,
        "block_open_grip": 95.0,
        "block_close_grip": 2.0,
    },
    {
        "name": "higher_lift",
        "pre_handle_offset": [0.00, 0.00, 0.11],
        "grasp_handle_offset": [0.002, 0.00, 0.010],
        "pull_offset": [0.16, 0.00, 0.010],
        "pre_block_offset": [0.03, 0.00, 0.13],
        "grasp_block_offset": [0.01, 0.00, 0.035],
        "lift_mid_offset": [0.04, 0.00, 0.19],
        "lift_high_offset": [0.07, 0.00, 0.34],
        "settle_offset": [0.08, 0.00, 0.36],
        "pre_handle_steps": 130,
        "grasp_handle_steps": 130,
        "close_handle_steps": 70,
        "pull_steps": 220,
        "hold_open_steps": 80,
        "pre_block_steps": 180,
        "grasp_block_steps": 150,
        "preclose_block_steps": 40,
        "close_block_steps": 120,
        "lift1_steps": 160,
        "lift2_steps": 220,
        "stabilize_steps": 200,
        "handle_grip": 30.0,
        "block_open_grip": 120.0,
        "block_close_grip": 6.0,
    },
]


def main():
    best = None
    for params in PARAM_SETS:
        print("attempt", params["name"])
        metrics = run_attempt(params)
        print(
            "metrics",
            params["name"],
            {
                "replay_steps": metrics.replay_steps,
                "drawer_before_lift": round(metrics.max_drawer_before_lift, 3),
                "final_drawer": round(metrics.final_drawer, 3),
                "max_block_z": round(metrics.max_block_z, 3),
                "final_block_z": round(metrics.final_block_z, 3),
                "contact": round(metrics.settle_contact_fraction, 3),
                "success": metrics.success(),
                "score": round(metrics.score(), 3),
            },
        )
        if best is None or metrics.score() > best.score():
            best = metrics
        if metrics.success():
            break
    print("best", best)


if __name__ == "__main__":
    main()
