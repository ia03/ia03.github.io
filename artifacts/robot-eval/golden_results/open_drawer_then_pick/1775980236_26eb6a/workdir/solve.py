import math
from pathlib import Path

import mujoco
import numpy as np

from sim import Sim


HAND_BODY = "hand"
LEFT_FINGER = "left_finger"
RIGHT_FINGER = "right_finger"
DRAWER_BODY = "drawer"
BLOCK_BODY = "block"

TOPDOWN_R = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)
PINCH_OFFSET_LOCAL = np.array([0.0, 0.0, 0.1066])


def orientation_error(current, target):
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )


class Controller:
    def __init__(self):
        self.sim = Sim()
        self.model = self.sim.model
        self.data = self.sim.data
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, LEFT_FINGER)
        self.right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_FINGER)
        self.drawer_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, DRAWER_BODY)
        self.block_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BLOCK_BODY)
        self.q_lo = self.model.jnt_range[:7, 0].copy()
        self.q_hi = self.model.jnt_range[:7, 1].copy()
        self.best_score = -1e9
        self.best_info = None

    def hand_pose(self):
        pos = self.data.xpos[self.hand_body_id].copy()
        rot = self.data.xmat[self.hand_body_id].reshape(3, 3).copy()
        return pos, rot

    def pinch_center(self):
        return 0.5 * (
            self.data.xpos[self.left_finger_id].copy() + self.data.xpos[self.right_finger_id].copy()
        )

    def block_pos(self):
        return self.sim.block_position()

    def drawer_open(self):
        return self.sim.drawer_open_amount()

    def set_arm_qpos(self, q):
        self.data.qpos[:7] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def solve_ik(self, target_hand_pos, target_rot, q_init=None, pos_weight=4.0, rot_weight=1.5, iters=120):
        q = self.data.qpos[:7].copy() if q_init is None else q_init.copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            self.set_arm_qpos(q)
            cur_pos, cur_rot = self.hand_pose()
            pos_err = target_hand_pos - cur_pos
            rot_err = orientation_error(cur_rot, target_rot)
            err = np.concatenate([pos_weight * pos_err, rot_weight * rot_err])
            if np.linalg.norm(pos_err) < 0.003 and np.linalg.norm(rot_err) < 0.03:
                break
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body_id)
            J = np.vstack([pos_weight * jacp[:, :7], rot_weight * jacr[:, :7]])
            JT = J.T
            dq = JT @ np.linalg.solve(J @ JT + 1e-4 * np.eye(6), err)
            q = np.clip(q + 0.7 * dq, self.q_lo, self.q_hi)
        self.set_arm_qpos(q)
        return q

    def move_pinch(self, pinch_target, grip, rot=TOPDOWN_R, chunks=40, sim_steps=10):
        target_hand = pinch_target - rot @ PINCH_OFFSET_LOCAL
        q = self.solve_ik(target_hand, rot)
        start_q = self.data.qpos[:7].copy()
        grip0 = float(self.data.ctrl[7])
        for i in range(1, chunks + 1):
            alpha = i / chunks
            self.data.ctrl[:7] = (1 - alpha) * start_q + alpha * q
            self.data.ctrl[7] = (1 - alpha) * grip0 + alpha * grip
            self.sim.step(sim_steps)
        return q

    def hold(self, steps, grip=None):
        if grip is not None:
            self.data.ctrl[7] = grip
        self.sim.step(steps)

    def save_if_best(self, label):
        score = 3.0 * self.drawer_open() + 4.0 * self.block_pos()[2] + 1.5 * float(self.sim.has_gripper_block_contact())
        if score > self.best_score:
            self.best_score = score
            self.best_info = (label, self.drawer_open(), self.block_pos().copy(), float(self.data.time))
            self.sim.save_final_state("/work/final_state.npz")
            print("saved", label, "drawer", round(self.drawer_open(), 4), "block", np.round(self.block_pos(), 4), "t", round(float(self.data.time), 3))

    def replay_eval(self, path="/work/final_state.npz"):
        data = np.load(path)
        ctrl_trace = data["ctrl_trace"]
        replay = Sim()
        opened_before_lift = False
        max_open = 0.0
        max_z = -1e9
        first_high = False
        for ctrl in ctrl_trace:
            replay.data.ctrl[:] = ctrl
            replay.step(1)
            op = replay.drawer_open_amount()
            z = replay.block_position()[2]
            max_open = max(max_open, op)
            max_z = max(max_z, z)
            if op >= 0.10:
                opened_before_lift = True
            if z >= 0.50 and not opened_before_lift:
                first_high = True
        contact = 0
        for _ in range(500):
            replay.step(1)
            contact += int(replay.has_gripper_block_contact())
        settle_contact = contact / 500.0
        result = {
            "ctrl_steps": len(ctrl_trace),
            "max_open": max_open,
            "max_z": max_z,
            "final_open": replay.drawer_open_amount(),
            "final_z": replay.block_position()[2],
            "contact_fraction": settle_contact,
            "opened_before_high": not first_high,
        }
        result["pass"] = (
            result["ctrl_steps"] >= 20
            and result["opened_before_high"]
            and result["max_open"] >= 0.10
            and result["final_open"] >= 0.10
            and result["final_z"] >= 0.60
            and result["contact_fraction"] >= 0.4
        )
        return result


def run_attempt():
    c = Controller()
    c.data.ctrl[:7] = c.data.qpos[:7]
    c.data.ctrl[7] = 80.0
    c.hold(50)

    drawer = c.data.xpos[c.drawer_body_id].copy()
    handle = drawer + np.array([0.093, 0.0, 0.0])
    block = c.block_pos().copy()

    waypoints = [
        ("pre_handle_high", handle + np.array([0.00, 0.0, 0.18]), 120.0, 35),
        ("pre_handle", handle + np.array([0.005, 0.0, 0.065]), 120.0, 30),
        ("handle_grasp", handle + np.array([0.005, 0.0, 0.020]), 60.0, 25),
        ("handle_close", handle + np.array([0.010, 0.0, 0.018]), 0.0, 25),
        ("pull1", handle + np.array([0.055, 0.0, 0.020]), 0.0, 35),
        ("pull2", handle + np.array([0.115, 0.0, 0.020]), 0.0, 40),
        ("pull3", handle + np.array([0.145, 0.0, 0.030]), 0.0, 30),
    ]
    for label, pos, grip, chunks in waypoints:
        c.move_pinch(pos, grip=grip, chunks=chunks, sim_steps=10)
        c.save_if_best(label)

    c.hold(120, grip=0.0)
    c.save_if_best("drawer_hold")

    opened = c.drawer_open()
    block = c.block_pos().copy()
    print("after pull drawer", round(opened, 4), "block", np.round(block, 4))

    pickup_points = [
        ("release_handle", block + np.array([0.18, 0.0, 0.17]), 120.0, 35),
        ("block_above_far", block + np.array([0.06, 0.0, 0.17]), 120.0, 30),
        ("block_above", block + np.array([0.01, 0.0, 0.14]), 120.0, 30),
        ("block_descend", block + np.array([0.005, 0.0, 0.055]), 90.0, 35),
        ("block_grasp", block + np.array([0.005, 0.0, 0.035]), 0.0, 40),
        ("lift1", block + np.array([0.02, 0.0, 0.18]), 0.0, 45),
        ("lift2", np.array([0.62, -0.02, 0.63]), 0.0, 50),
        ("lift3", np.array([0.58, -0.02, 0.68]), 0.0, 40),
    ]
    for label, pos, grip, chunks in pickup_points:
        c.move_pinch(pos, grip=grip, chunks=chunks, sim_steps=10)
        c.save_if_best(label)

    c.hold(250, grip=0.0)
    c.save_if_best("final_hold")

    result = c.replay_eval()
    print("eval", result)
    return result


if __name__ == "__main__":
    result = run_attempt()
    print("PASS" if result["pass"] else "FAIL")
