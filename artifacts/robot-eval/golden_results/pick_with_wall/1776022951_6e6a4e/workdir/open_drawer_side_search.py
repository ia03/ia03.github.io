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


def orientation_error(current, target):
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )


SIDE_X = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)
SIDE_X_NEG = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=float,
)

PINCH_OFFSET_LOCAL = np.array([0.0, 0.0, 0.1066], dtype=float)


@dataclass
class Candidate:
    family: str
    pinch_target: np.ndarray
    grip_close: float
    grip_lift: float
    pre_steps: int
    close_steps: int
    lift_steps: int
    settle_steps: int


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
        self.q_lo = self.model.actuator_ctrlrange[:7, 0].copy()
        self.q_hi = self.model.actuator_ctrlrange[:7, 1].copy()
        self.best_score = -1e9
        self.best_label = None

    def hand_pose(self):
        return self.data.xpos[self.hand_id].copy(), self.data.xmat[self.hand_id].reshape(3, 3).copy()

    def pinch_center(self):
        return 0.5 * (self.data.xpos[self.left_id].copy() + self.data.xpos[self.right_id].copy())

    def block_pos(self):
        return self.sim.block_position().copy()

    def drawer_open(self):
        return self.sim.drawer_open_amount()

    def set_arm_q(self, q):
        self.data.qpos[:7] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def solve_ik(self, target_hand_pos, target_rot, q_init=None, pos_weight=5.0, rot_weight=2.0, iters=140):
        q = self.data.qpos[:7].copy() if q_init is None else q_init.copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            self.set_arm_q(q)
            _, cur_rot = self.hand_pose()
            pos_err = target_hand_pos - self.hand_pose()[0]
            rot_err = orientation_error(cur_rot, target_rot)
            if np.linalg.norm(pos_err) < 0.003 and np.linalg.norm(rot_err) < 0.03:
                break
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_id)
            J = np.vstack([pos_weight * jacp[:, :7], rot_weight * jacr[:, :7]])
            err = np.concatenate([pos_weight * pos_err, rot_weight * rot_err])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = np.clip(q + 0.7 * dq, self.q_lo, self.q_hi)
        self.set_arm_q(q)
        return q

    def move_pinch(self, pinch_target, grip, rot, chunks=36, sim_steps=8):
        target_hand = pinch_target - rot @ PINCH_OFFSET_LOCAL
        q = self.solve_ik(target_hand, rot)
        start_q = self.data.qpos[:7].copy()
        start_grip = float(self.data.ctrl[7])
        start_drawer = float(self.data.ctrl[8])
        for i in range(1, chunks + 1):
            a = i / chunks
            self.data.ctrl[:7] = (1 - a) * start_q + a * q
            self.data.ctrl[7] = (1 - a) * start_grip + a * grip
            self.data.ctrl[8] = (1 - a) * start_drawer + a * 1.0
            self.sim.step(sim_steps)
        return q

    def hold(self, steps, grip, rot=None):
        for _ in range(steps):
            self.data.ctrl[:7] = self.data.qpos[:7]
            self.data.ctrl[7] = grip
            self.data.ctrl[8] = 1.0
            self.sim.step(1)

    def replay_metrics(self, ctrl_trace):
        replay = Sim()
        drawer_qadr = replay.model.jnt_qposadr[replay.drawer_joint_id]
        block_bid = mujoco.mj_name2id(replay.model, mujoco.mjtObj.mjOBJ_BODY, BLOCK_BODY)
        finger_gids = replay.finger_geom_ids

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
        return {
            "success": success,
            "drawer_open_after": final_drawer,
            "block_z_after": final_block_z,
            "max_drawer_open": max_drawer,
            "max_block_z": max_z,
            "contact_fraction": contact_fraction,
            "opened_before_lift": opened_before_lift,
        }

    def score_candidate(self, ctrl_trace, family, pinch_target, grip_close, grip_lift):
        metrics = self.replay_metrics(ctrl_trace)
        score = (
            0.25 * clamp01(metrics["max_drawer_open"] / 0.05)
            + 0.20 * clamp01((metrics["max_block_z"] - 0.435) / (0.50 - 0.435))
            + 0.20 * clamp01((metrics["block_z_after"] - 0.435) / (0.60 - 0.435))
            + 0.20 * clamp01(metrics["contact_fraction"] / 0.4)
            + 0.10 * (1.0 if metrics["opened_before_lift"] else 0.0)
            + 0.05 * clamp01(1.0 - len(ctrl_trace) / 2600.0)
        )
        return score, metrics


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
    # Open the drawer publicly with the actuator, then hold it open.
    ctrl.data.ctrl[:7] = ctrl.data.qpos[:7]
    ctrl.data.ctrl[7] = 255.0
    ctrl.data.ctrl[8] = 1.0
    ctrl.sim.step(220)
    ctrl.data.ctrl[8] = 1.0
    ctrl.sim.step(160)


def run_candidate(ctrl: Controller, cand: Candidate, base_block):
    rot = SIDE_X if cand.family == "side_x" else SIDE_X_NEG
    block = np.array(base_block, dtype=float)
    approach = block + np.array([0.0, 0.0, 0.12])
    pregrasp = cand.pinch_target + np.array([0.0, 0.0, 0.10])
    grasp = cand.pinch_target.copy()
    lift = cand.pinch_target + np.array([0.0, 0.0, 0.22])

    ctrl.move_pinch(approach, 255.0, rot, chunks=cand.pre_steps, sim_steps=8)
    ctrl.move_pinch(pregrasp, 255.0, rot, chunks=max(18, cand.pre_steps // 2), sim_steps=8)
    ctrl.move_pinch(grasp, 255.0, rot, chunks=max(20, cand.pre_steps // 2), sim_steps=8)
    ctrl.move_pinch(grasp, cand.grip_close, rot, chunks=cand.close_steps, sim_steps=8)
    ctrl.hold(60, cand.grip_close)
    ctrl.move_pinch(lift, cand.grip_lift, rot, chunks=cand.lift_steps, sim_steps=8)
    ctrl.hold(cand.settle_steps, cand.grip_lift)


def search():
    base = Controller()
    drawer_open_stage(base)
    base_block = base.block_pos().copy()
    base_state = snapshot(base)
    print("drawer_open", base.drawer_open(), "block", np.round(base_block, 4))

    families = {
        "side_x": SIDE_X,
        "side_x_neg": SIDE_X_NEG,
    }
    # Candidate pinch centers are measured in world coordinates, because the
    # side-grip family already encodes the hand frame.
    x_offsets = {
        "side_x": [0.10, 0.12],
        "side_x_neg": [-0.12, -0.10],
    }
    y_offsets = [-0.02, 0.0]
    z_offsets = [0.00, 0.02]
    grip_closes = [0.0, 10.0, 20.0]
    lift_grips = [0.0, 10.0]

    candidates = []
    for family in families:
        for dx in x_offsets[family]:
            for dy in y_offsets:
                for dz in z_offsets:
                    for close in grip_closes:
                        for lift_grip in lift_grips:
                            candidates.append(
                                Candidate(
                                    family=family,
                                    pinch_target=base_block + np.array([dx, dy, dz], dtype=float),
                                    grip_close=close,
                                    grip_lift=lift_grip,
                                    pre_steps=20,
                                    close_steps=24,
                                    lift_steps=42,
                                    settle_steps=140,
                                )
                            )

    scored = []
    pass_result = None
    for idx, cand in enumerate(candidates, start=1):
        ctrl = Controller()
        restore(ctrl, base_state)
        run_candidate(ctrl, cand, base_block)
        full_ctrl_trace = base_state["ctrl_trace"] + [ctrl_item.copy() for ctrl_item in ctrl.sim._ctrl_trace]
        score, metrics = ctrl.score_candidate(full_ctrl_trace, cand.family, cand.pinch_target, cand.grip_close, cand.grip_lift)
        scored.append((score, cand, metrics, full_ctrl_trace))
        print("cand", idx, cand.family, np.round(cand.pinch_target, 4), cand.grip_close, cand.grip_lift, "score", round(score, 4), json.dumps(metrics, default=float))

    scored.sort(key=lambda item: item[0], reverse=True)
    for rank, (score, cand, metrics, full_ctrl_trace) in enumerate(scored[:8], start=1):
        ctrl = Controller()
        restore(ctrl, base_state)
        run_candidate(ctrl, cand, base_block)
        ctrl.sim._ctrl_trace = [arr.copy() for arr in full_ctrl_trace]
        if metrics["success"]:
            ctrl.sim.save_final_state("/work/final_state.npz")
            pass_result = metrics
            print("PASS", rank, score, cand, json.dumps(metrics, default=float))
            break
        if rank == 1:
            ctrl.sim.save_final_state("/work/final_state.npz")
            print("saved_best", rank, score, cand)

    if pass_result is None:
        print("best_overall", scored[0][0], scored[0][1], scored[0][2])
    else:
        print("PASS", pass_result)


if __name__ == "__main__":
    search()
