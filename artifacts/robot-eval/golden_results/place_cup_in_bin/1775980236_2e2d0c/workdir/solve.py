import math
import numpy as np
import mujoco

from sim import Sim


HAND_BODY = "hand"
LEFT_FINGER = "left_finger"
RIGHT_FINGER = "right_finger"


class Controller:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, LEFT_FINGER)
        self.right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_FINGER)
        self.q_target = self.data.qpos[:7].copy()
        self.r_des = self.data.xmat[self.hand_id].reshape(3, 3).copy()
        hand_pos = self.data.xpos[self.hand_id].copy()
        pinch_pos = self.pinch_pos()
        self.pinch_offset_local = self.r_des.T @ (pinch_pos - hand_pos)

    def pinch_pos(self):
        return 0.5 * (self.data.xpos[self.left_id] + self.data.xpos[self.right_id])

    def set_gripper(self, value):
        self.data.ctrl[7] = float(np.clip(value, *self.model.actuator_ctrlrange[7]))

    def step_to(self, pinch_target, gripper, steps, pos_gain=2.5, rot_gain=1.2):
        pinch_target = np.array(pinch_target, dtype=float)
        for _ in range(steps):
            self.set_gripper(gripper)
            self._ik_update(pinch_target, pos_gain=pos_gain, rot_gain=rot_gain)
            self.sim.step(1)

    def _ik_update(self, pinch_target, pos_gain=2.5, rot_gain=1.2):
        desired_hand = pinch_target - self.r_des @ self.pinch_offset_local
        current_hand = self.data.xpos[self.hand_id].copy()
        pos_err = desired_hand - current_hand

        current_r = self.data.xmat[self.hand_id].reshape(3, 3).copy()
        rot_err = 0.5 * (
            np.cross(current_r[:, 0], self.r_des[:, 0])
            + np.cross(current_r[:, 1], self.r_des[:, 1])
            + np.cross(current_r[:, 2], self.r_des[:, 2])
        )

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_id)
        j = np.vstack([jacp[:, :7], jacr[:, :7]])
        err = np.concatenate([pos_gain * pos_err, rot_gain * rot_err])

        damping = 1e-3
        lhs = j.T @ j + damping * np.eye(7)
        dq = np.linalg.solve(lhs, j.T @ err)
        dq = np.clip(dq, -0.05, 0.05)
        self.q_target += dq
        lo = self.model.actuator_ctrlrange[:7, 0]
        hi = self.model.actuator_ctrlrange[:7, 1]
        self.q_target = np.clip(self.q_target, lo, hi)
        self.data.ctrl[:7] = self.q_target


def cup_uprightness(sim: Sim):
    return float(sim.data.xmat[sim.cup_body_id].reshape(3, 3)[2, 2])


def evaluate(sim: Sim):
    trace = sim._trace
    if not trace:
        return {}
    replay_contacts = np.array([entry["cup_contact"] for entry in trace], dtype=float)
    replay_cup = np.array([entry["cup_pos"] for entry in trace], dtype=float)

    settle = Sim()
    ctrl_trace = np.array(sim._ctrl_trace, dtype=float).reshape(-1, sim.model.nu)
    for ctrl in ctrl_trace:
        settle.data.ctrl[:] = ctrl
        settle.step(1)
    replay_ok = len(ctrl_trace) >= 20
    replay_contact = np.any(replay_contacts > 0.5)
    replay_lift = np.max(replay_cup[:, 2]) >= 0.50
    last10_contact = replay_contacts[-10:].mean() if len(replay_contacts) >= 10 else 1.0

    settle_contacts = []
    for _ in range(500):
        settle_contacts.append(float(settle.has_gripper_cup_contact()))
        settle.step(1)
    settle_contacts = np.array(settle_contacts, dtype=float)
    cup = settle.cup_position()
    inside = abs(cup[0] - 0.70) <= 0.046 and abs(cup[1] - 0.14) <= 0.046
    high = cup[2] >= 0.42
    upright = cup_uprightness(settle) >= 0.85
    release_replay = last10_contact < 0.5
    release_settle = settle_contacts.mean() < 0.1
    success = all([replay_ok, replay_contact, replay_lift, release_replay, release_settle, inside, high, upright])
    return {
        "success": success,
        "steps": len(ctrl_trace),
        "max_replay_z": float(np.max(replay_cup[:, 2])),
        "final_cup": cup,
        "uprightness": cup_uprightness(settle),
        "inside": inside,
        "high": high,
        "replay_contact": replay_contact,
        "release_replay": release_replay,
        "release_settle": release_settle,
    }


def run_attempt():
    sim = Sim()
    ctrl = Controller(sim)

    cup = sim.cup_position().copy()
    above_cup = cup + np.array([0.0, 0.0, 0.14])
    at_cup = cup + np.array([0.0, 0.0, 0.045])
    lift = np.array([cup[0], cup[1], 0.56])
    pre_bin = np.array([0.70, 0.14, 0.58])
    place = np.array([0.70, 0.14, 0.485])
    retreat = np.array([0.70, 0.14, 0.60])

    ctrl.step_to(above_cup, gripper=230, steps=220)
    sim.save_final_state("/work/final_state.npz")
    ctrl.step_to(at_cup, gripper=230, steps=180)
    ctrl.step_to(at_cup, gripper=0, steps=140)
    ctrl.step_to(lift, gripper=0, steps=180)
    ctrl.step_to(pre_bin, gripper=0, steps=220)
    ctrl.step_to(place, gripper=0, steps=160)
    ctrl.step_to(place, gripper=255, steps=160)
    ctrl.step_to(retreat, gripper=255, steps=160)
    sim.step(220)
    sim.save_final_state("/work/final_state.npz")
    return sim


if __name__ == "__main__":
    sim = run_attempt()
    result = evaluate(sim)
    print(result)
