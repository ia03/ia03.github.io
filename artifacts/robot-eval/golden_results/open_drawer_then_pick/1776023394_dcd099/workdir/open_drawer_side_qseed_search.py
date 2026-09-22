import json
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


HAND_BODY = "hand"
LEFT_FINGER = "left_finger"
RIGHT_FINGER = "right_finger"
DRAWER_BODY = "drawer"
BLOCK_BODY = "block"


def clamp01(x):
    return float(max(0.0, min(1.0, x)))


@dataclass
class Candidate:
    q_grasp: np.ndarray
    q_lift: np.ndarray
    pre_mix: float
    pre_steps: int
    close_steps: int
    lift_steps: int
    settle_steps: int
    grip_close: float


class Controller:
    def __init__(self):
        self.sim = Sim()
        self.model = self.sim.model
        self.data = self.sim.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, LEFT_FINGER)
        self.right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_FINGER)
        self.drawer_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, DRAWER_BODY)
        self.block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BLOCK_BODY)

    def block_pos(self):
        return self.sim.block_position().copy()

    def drawer_open(self):
        return self.sim.drawer_open_amount()

    def move_q(self, q_target, grip, steps):
        q_start = self.data.qpos[:7].copy()
        g_start = float(self.data.ctrl[7])
        d_start = float(self.data.ctrl[8])
        for i in range(steps):
            a = (i + 1) / steps
            self.data.ctrl[:7] = (1 - a) * q_start + a * q_target
            self.data.ctrl[7] = (1 - a) * g_start + a * grip
            self.data.ctrl[8] = (1 - a) * d_start + a * 1.0
            self.sim.step(1)

    def hold(self, q_target, grip, steps):
        for _ in range(steps):
            self.data.ctrl[:7] = q_target
            self.data.ctrl[7] = grip
            self.data.ctrl[8] = 1.0
            self.sim.step(1)

    def replay_metrics(self, ctrl_trace):
        replay = Sim()
        drawer_qadr = replay.model.jnt_qposadr[replay.drawer_joint_id]
        block_bid = mujoco.mj_name2id(replay.model, mujoco.mjtObj.mjOBJ_BODY, BLOCK_BODY)

        trace_drawer = []
        trace_block_z = []
        for ctrl in ctrl_trace:
            replay.data.ctrl[:] = ctrl
            replay.step(1)
            trace_drawer.append(float(replay.data.qpos[drawer_qadr]))
            trace_block_z.append(float(replay.data.xpos[block_bid][2]))

        contact = 0
        for _ in range(500):
            replay.step(1)
            contact += int(replay.has_gripper_block_contact())

        trace_drawer = np.array(trace_drawer, dtype=float)
        trace_block_z = np.array(trace_block_z, dtype=float)
        max_drawer = float(trace_drawer.max()) if trace_drawer.size else 0.0
        max_z = float(trace_block_z.max()) if trace_block_z.size else 0.0
        first_drawer_idx = int(np.flatnonzero(trace_drawer >= 0.05)[0]) if np.any(trace_drawer >= 0.05) else None
        first_high_idx = int(np.flatnonzero(trace_block_z >= 0.50)[0]) if np.any(trace_block_z >= 0.50) else None
        opened_before_lift = first_drawer_idx is not None and first_high_idx is not None and first_drawer_idx <= first_high_idx
        final_drawer = replay.drawer_open_amount()
        final_block_z = float(replay.block_position()[2])
        contact_fraction = contact / 500.0
        success = (
            len(ctrl_trace) >= 20
            and opened_before_lift
            and max_drawer >= 0.05
            and final_drawer >= 0.05
            and final_block_z >= 0.60
            and contact_fraction >= 0.4
        )
        score = (
            0.25 * clamp01(max_drawer / 0.05)
            + 0.20 * clamp01((max_z - 0.435) / (0.50 - 0.435))
            + 0.20 * clamp01((final_block_z - 0.435) / (0.60 - 0.435))
            + 0.20 * clamp01(contact_fraction / 0.4)
            + 0.10 * (1.0 if opened_before_lift else 0.0)
            + 0.05 * clamp01(1.0 - len(ctrl_trace) / 2600.0)
        )
        return {
            "success": success,
            "score": score,
            "drawer_open_after": final_drawer,
            "block_z_after": final_block_z,
            "max_drawer_open": max_drawer,
            "max_block_z": max_z,
            "contact_fraction": contact_fraction,
            "opened_before_lift": opened_before_lift,
        }


def snapshot(sim: Controller):
    return {
        "qpos": sim.data.qpos.copy(),
        "qvel": sim.data.qvel.copy(),
        "ctrl": sim.data.ctrl.copy(),
        "trace": list(sim.sim._trace),
        "ctrl_trace": [item.copy() for item in sim.sim._ctrl_trace],
        "step_counter": int(sim.sim._step_counter),
    }


def restore(sim: Controller, state):
    sim.data.qpos[:] = state["qpos"]
    sim.data.qvel[:] = state["qvel"]
    sim.data.ctrl[:] = state["ctrl"]
    mujoco.mj_forward(sim.model, sim.data)
    sim.sim._trace = []
    sim.sim._ctrl_trace = []
    sim.sim._step_counter = 0
    sim.sim._record_trace()


def drawer_open_stage(ctrl: Controller):
    ctrl.data.ctrl[:7] = ctrl.data.qpos[:7]
    ctrl.data.ctrl[7] = 255.0
    ctrl.data.ctrl[8] = 1.0
    ctrl.sim.step(220)
    ctrl.data.ctrl[8] = 1.0
    ctrl.sim.step(160)


def run_candidate(ctrl: Controller, cand: Candidate):
    q_pre = (1.0 - cand.pre_mix) * np.array([0.0, -0.35, 0.0, -1.90, 0.0, 1.95, 0.785], dtype=float) + cand.pre_mix * cand.q_grasp
    ctrl.move_q(q_pre, 255.0, cand.pre_steps)
    ctrl.move_q(cand.q_grasp, 255.0, max(12, cand.pre_steps // 2))
    ctrl.move_q(cand.q_grasp, cand.grip_close, cand.close_steps)
    ctrl.hold(cand.q_grasp, cand.grip_close, 60)
    ctrl.move_q(cand.q_lift, cand.grip_close, cand.lift_steps)
    ctrl.hold(cand.q_lift, cand.grip_close, cand.settle_steps)


def search():
    base = Controller()
    drawer_open_stage(base)
    base_state = snapshot(base)
    print("drawer_open", base.drawer_open(), "block", np.round(base.block_pos(), 4))

    q_grasp_seeds = [
        np.array([1.359, 0.0771, -1.3438, -1.9034, -2.7852, 2.9761, -2.6228], dtype=float),
        np.array([1.5944, -0.6717, -1.1520, -2.1909, 1.3621, 2.7958, -1.2075], dtype=float),
        np.array([-0.0790, -0.1252, 0.2302, -2.1471, -0.8928, 1.7791, 2.1784], dtype=float),
        np.array([-2.4170, 0.1466, 2.6195, -2.2760, 0.7943, 2.8952, -0.0998], dtype=float),
    ]
    q_lift_seeds = [
        np.array([-2.5867, -1.4570, 1.3223, -0.9166, -1.0098, 0.6809, -1.6676], dtype=float),
        np.array([0.2601, 1.3605, 2.7199, -0.4998, 1.4958, 0.2920, 0.6292], dtype=float),
        np.array([0.3998, 0.7431, -0.9482, -1.2572, 1.1148, 0.0437, -2.5250], dtype=float),
    ]

    candidates = []
    for q_grasp in q_grasp_seeds:
        for q_lift in q_lift_seeds:
            for pre_mix in [0.70, 0.82]:
                for grip_close in [0.0, 5.0, 10.0]:
                    for pre_steps, close_steps, lift_steps, settle_steps in [
                        (120, 80, 140, 160),
                        (160, 100, 180, 200),
                    ]:
                        candidates.append(
                            Candidate(
                                q_grasp=q_grasp,
                                q_lift=q_lift,
                                pre_mix=pre_mix,
                                pre_steps=pre_steps,
                                close_steps=close_steps,
                                lift_steps=lift_steps,
                                settle_steps=settle_steps,
                                grip_close=grip_close,
                            )
                        )

    scored = []
    for idx, cand in enumerate(candidates, start=1):
        ctrl = Controller()
        restore(ctrl, base_state)
        run_candidate(ctrl, cand)
        full_ctrl_trace = base_state["ctrl_trace"] + [item.copy() for item in ctrl.sim._ctrl_trace]
        metrics = ctrl.replay_metrics(full_ctrl_trace)
        scored.append((metrics["score"], cand, metrics, full_ctrl_trace))
        print(
            "cand",
            idx,
            np.round(cand.q_grasp, 4),
            np.round(cand.q_lift, 4),
            cand.pre_mix,
            cand.grip_close,
            cand.pre_steps,
            cand.close_steps,
            cand.lift_steps,
            cand.settle_steps,
            "score",
            round(metrics["score"], 4),
            json.dumps(metrics, default=float),
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_cand, best_metrics, best_trace = scored[0]
    best = Controller()
    restore(best, base_state)
    run_candidate(best, best_cand)
    best.sim._ctrl_trace = [arr.copy() for arr in best_trace]
    if best_metrics["success"]:
        best.sim.save_final_state("/work/final_state.npz")
        print("PASS", best_score, best_cand, json.dumps(best_metrics, default=float))
    else:
        best.sim.save_final_state("/work/final_state.npz")
        print("best_overall", best_score, best_cand, json.dumps(best_metrics, default=float))


if __name__ == "__main__":
    search()
