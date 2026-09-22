import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


DEADLINE = 1744510239.0  # 2026-04-13 02:10:39 UTC


@dataclass
class TrialResult:
    score: float
    passed: bool
    final_block_z: float
    final_drawer: float
    settle_contact_fraction: float
    max_replay_block_z: float
    max_replay_drawer: float
    sim: Sim
    summary: dict


class Planner:
    def __init__(self):
        self.sim = Sim()
        self.model = self.sim.model
        self.data = self.sim.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    def solve_hand_ik(self, target, q_init=None, steps=100):
        q = self.data.qpos[:7].copy() if q_init is None else np.array(q_init, dtype=float).copy()
        for _ in range(steps):
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)
            err = np.array(target, dtype=float) - self.data.xpos[self.hand_id]
            if np.linalg.norm(err) < 1e-4:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_id)
            j = jacp[:, :7]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-3 * np.eye(3), err)
            q = np.clip(q + 0.8 * dq, self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])
        self.data.qpos[:7] = q
        mujoco.mj_forward(self.model, self.data)
        return q


def unix_time():
    return time.time()


def progress_score(max_drawer, max_block_z, opened_before_lift, final_drawer, final_block_z, contact_fraction):
    drawer_progress = np.clip(max_drawer / 0.05, 0.0, 1.0)
    retrieval_progress = np.clip((max_block_z - 0.435) / (0.50 - 0.435), 0.0, 1.0)
    ordered_retrieval_progress = retrieval_progress if opened_before_lift else 0.0
    final_height_progress = np.clip((final_block_z - 0.43) / (0.595 - 0.43), 0.0, 1.0)
    hold_progress = np.clip(contact_fraction / 0.4, 0.0, 1.0)
    final_drawer_progress = np.clip(final_drawer / 0.05, 0.0, 1.0)
    return (
        0.22 * drawer_progress
        + 0.28 * ordered_retrieval_progress
        + 0.22 * final_height_progress
        + 0.18 * hold_progress
        + 0.10 * final_drawer_progress
    )


def evaluate_trial(sim: Sim):
    max_drawer = 0.0
    max_block_z = -1e9
    first_high_block_time = math.inf
    first_open_time = math.inf
    trace = sim._trace
    for entry in trace:
        drawer = float(entry["drawer_open"])
        block_z = float(entry["block_pos"][2])
        if drawer >= 0.05 and first_open_time == math.inf:
            first_open_time = float(entry["time"])
        if block_z >= 0.50 and first_high_block_time == math.inf:
            first_high_block_time = float(entry["time"])
        max_drawer = max(max_drawer, drawer)
        max_block_z = max(max_block_z, block_z)
    opened_before_lift = first_open_time < first_high_block_time

    contact_steps = 0
    settle_steps = 500
    for _ in range(settle_steps):
        sim.step(1)
        if sim.has_gripper_block_contact():
            contact_steps += 1
    contact_fraction = contact_steps / settle_steps
    final_drawer = sim.drawer_open_amount()
    final_block_z = sim.block_position()[2]
    passed = (
        len(sim._ctrl_trace) >= 20
        and max_drawer >= 0.05
        and opened_before_lift
        and final_drawer >= 0.05
        and final_block_z >= 0.595
        and contact_fraction >= 0.4
    )
    score = progress_score(max_drawer, max_block_z, opened_before_lift, final_drawer, final_block_z, contact_fraction)
    return TrialResult(
        score=score,
        passed=passed,
        final_block_z=final_block_z,
        final_drawer=final_drawer,
        settle_contact_fraction=contact_fraction,
        max_replay_block_z=max_block_z,
        max_replay_drawer=max_drawer,
        sim=sim,
        summary={
            "ctrl_steps": len(sim._ctrl_trace),
            "opened_before_lift": opened_before_lift,
        },
    )


def goto_ctrl(sim: Sim, target_ctrl, steps):
    target_ctrl = np.array(target_ctrl, dtype=float)
    start = sim.data.ctrl.copy()
    for i in range(steps):
        alpha = (i + 1) / steps
        sim.data.ctrl[:] = (1 - alpha) * start + alpha * target_ctrl
        sim.step(1)


def hold_ctrl(sim: Sim, ctrl, steps):
    sim.data.ctrl[:] = ctrl
    sim.step(steps)


def run_trial(params):
    planner = Planner()
    sim = planner.sim
    open_grip = 255.0
    close_grip = params["close_grip"]

    high_q = planner.solve_hand_ik([0.45, 0.0, 0.72])
    pre_q = planner.solve_hand_ik(params["pre_pos"], q_init=high_q)
    grasp_q = planner.solve_hand_ik(params["grasp_pos"], q_init=pre_q)
    lift_q = planner.solve_hand_ik(params["lift_pos"], q_init=grasp_q)
    retreat_q = planner.solve_hand_ik(params["retreat_pos"], q_init=lift_q)

    ctrl = np.zeros(sim.model.nu)
    ctrl[:7] = sim.data.qpos[:7]
    ctrl[7] = open_grip
    ctrl[8] = 1.0
    sim.data.ctrl[:] = ctrl

    goto_ctrl(sim, np.r_[high_q, open_grip, 1.0], 80)
    goto_ctrl(sim, np.r_[pre_q, open_grip, 1.0], params["pre_steps"])
    goto_ctrl(sim, np.r_[grasp_q, open_grip, 1.0], params["down_steps"])
    hold_ctrl(sim, np.r_[grasp_q, open_grip, 1.0], params["settle_before_close"])
    goto_ctrl(sim, np.r_[grasp_q, close_grip, 1.0], params["close_steps"])
    hold_ctrl(sim, np.r_[grasp_q, close_grip, 1.0], params["squeeze_steps"])
    goto_ctrl(sim, np.r_[lift_q, close_grip, 1.0], params["lift_steps"])
    hold_ctrl(sim, np.r_[lift_q, close_grip, 1.0], params["hold_after_lift"])
    goto_ctrl(sim, np.r_[retreat_q, close_grip, 1.0], params["retreat_steps"])
    hold_ctrl(sim, np.r_[retreat_q, close_grip, 1.0], params["final_hold"])

    return evaluate_trial(sim)


def candidate_params():
    for y in [0.0, 0.012, -0.012, 0.02, -0.02]:
        for pre_z in [0.55, 0.53]:
            for grasp_z in [0.49, 0.48, 0.47]:
                for grasp_x in [0.775, 0.785, 0.795]:
                    for lift_z in [0.68, 0.72, 0.76]:
                        for lift_x in [0.73, 0.78]:
                            for close_grip in [0.0, 5.0, 10.0, 20.0]:
                                yield {
                                    "pre_pos": [grasp_x, y, pre_z],
                                    "grasp_pos": [grasp_x, y, grasp_z],
                                    "lift_pos": [lift_x, y, lift_z],
                                    "retreat_pos": [lift_x - 0.03, y, lift_z + 0.02],
                                    "pre_steps": 90,
                                    "down_steps": 70,
                                    "settle_before_close": 8,
                                    "close_steps": 35,
                                    "squeeze_steps": 20,
                                    "lift_steps": 90,
                                    "hold_after_lift": 30,
                                    "retreat_steps": 40,
                                    "final_hold": 30,
                                    "close_grip": close_grip,
                                }


def main():
    best = None
    start = unix_time()
    for idx, params in enumerate(candidate_params(), start=1):
        now = unix_time()
        if now > DEADLINE - 30:
            break
        result = run_trial(params)
        if best is None or result.score > best.score:
            best = result
            best.sim.save_final_state("/work/final_state.npz")
            print(
                f"trial={idx} score={result.score:.3f} pass={result.passed} "
                f"final_z={result.final_block_z:.3f} contact={result.settle_contact_fraction:.3f} "
                f"drawer={result.final_drawer:.3f} max_z={result.max_replay_block_z:.3f} params={params}",
                flush=True,
            )
        if idx % 10 == 0:
            remain = DEADLINE - unix_time()
            print(
                f"checkpoint trial={idx} elapsed={unix_time()-start:.1f}s remain={remain:.1f}s "
                f"best={best.score if best else -1:.3f}",
                flush=True,
            )
        if best and best.passed:
            break
    if best is not None:
        print("best_summary", best.score, best.passed, best.summary, flush=True)


if __name__ == "__main__":
    main()
