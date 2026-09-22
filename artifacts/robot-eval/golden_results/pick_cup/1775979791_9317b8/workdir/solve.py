import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import CUP_HALF_HEIGHT, Sim


ARM_QPOS = slice(0, 7)
FINGER_QPOS = slice(7, 9)


def body_pose(data, body_id):
    pos = data.xpos[body_id].copy()
    mat = data.xmat[body_id].reshape(3, 3).copy()
    return pos, mat


def pose_error(current_pos, current_mat, target_pos, target_mat):
    pos_err = target_pos - current_pos
    rot_err = 0.5 * (
        np.cross(current_mat[:, 0], target_mat[:, 0])
        + np.cross(current_mat[:, 1], target_mat[:, 1])
        + np.cross(current_mat[:, 2], target_mat[:, 2])
    )
    return np.concatenate([pos_err, rot_err])


def make_grasp_rot(y_axis):
    y_axis = np.asarray(y_axis, dtype=float)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def quat_from_mat(mat):
    quat = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quat, mat.reshape(-1))
    return quat


@dataclass
class Phase:
    arm_q: np.ndarray
    grip: float
    steps: int


class Planner:
    def __init__(self):
        self.sim = Sim()
        self.model = self.sim.model
        self.data = self.sim.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.cup_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        self.hand_offset = np.array([0.0, 0.0, 0.1029])

    def set_arm_q(self, q):
        self.data.qpos[ARM_QPOS] = q
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def pinch_pose(self):
        hand_pos, hand_mat = body_pose(self.data, self.hand_id)
        pinch = hand_pos + hand_mat @ self.hand_offset
        return pinch, hand_mat

    def ik_contact(self, target_contact, q_init=None, target_z=np.array([0.0, 0.0, -1.0]), z_weight=0.1):
        q = np.array(
            q_init if q_init is not None else [0.0, -0.4, 0.0, -2.0, 0.0, 1.8, 0.7],
            dtype=float,
        ).copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        target_contact = np.asarray(target_contact, dtype=float)
        target_z = np.asarray(target_z, dtype=float)
        for _ in range(300):
            self.set_arm_q(q)
            pinch, hand_mat = self.pinch_pose()
            pos_err = target_contact - pinch
            z_err = np.cross(hand_mat[:, 2], target_z)
            err = np.concatenate([pos_err, z_weight * z_err])
            if np.linalg.norm(pos_err) < 1.5e-3 and np.linalg.norm(z_err) < 2e-2:
                break
            mujoco.mj_jac(self.model, self.data, jacp, jacr, pinch, self.hand_id)
            J = np.vstack([jacp[:, :7], z_weight * jacr[:, :7]])
            damp = 3e-3
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
            q += np.clip(dq, -0.06, 0.06)
            q = np.clip(q, self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])
        self.set_arm_q(q)
        pinch, hand_mat = self.pinch_pose()
        return q, np.linalg.norm(target_contact - pinch), np.linalg.norm(np.cross(hand_mat[:, 2], target_z))


def contact_fraction(sim, steps=500):
    cup_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
    left_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    hits = 0
    for _ in range(steps):
        touched = False
        for i in range(sim.data.ncon):
            con = sim.data.contact[i]
            b1 = sim.model.geom_bodyid[con.geom1]
            b2 = sim.model.geom_bodyid[con.geom2]
            bodies = {int(b1), int(b2)}
            if cup_bid in bodies and (left_bid in bodies or right_bid in bodies):
                touched = True
                break
        hits += int(touched)
        sim.step(1)
    return hits / steps


def rollout(phases):
    sim = Sim()
    sim.data.ctrl[7] = 255.0
    sim.step(5)
    for phase in phases:
        start = sim.data.ctrl[:7].copy()
        for i in range(phase.steps):
            alpha = (i + 1) / phase.steps
            sim.data.ctrl[:7] = start * (1 - alpha) + phase.arm_q * alpha
            sim.data.ctrl[7] = phase.grip
            sim.step(1)
    return sim


def evaluate(phases):
    sim = rollout(phases)
    end_z = float(sim.cup_position()[2])
    settle_contact = contact_fraction(sim, steps=500)
    final_z = float(sim.cup_position()[2])
    progress = max(0.0, min(1.0, (final_z - 0.435) / (0.52 - 0.435)))
    contact_prog = max(0.0, min(1.0, settle_contact / 0.5))
    score = 0.5 * (progress + contact_prog)
    return {
        "sim": sim,
        "end_z": end_z,
        "final_z": final_z,
        "contact_fraction": settle_contact,
        "progress_score": score,
    }


def build_plan():
    planner = Planner()
    cup = planner.sim.cup_position()
    q_above, e1, r1 = planner.ik_contact(cup + np.array([0.0, 0.0, 0.09]))
    q_pre, e2, r2 = planner.ik_contact(cup + np.array([0.0, 0.0, 0.045]), q_init=q_above)
    q_grasp, e3, r3 = planner.ik_contact(cup + np.array([0.0, 0.0, 0.015]), q_init=q_pre)
    q_lift, e4, r4 = planner.ik_contact(cup + np.array([0.0, 0.0, 0.16]), q_init=q_grasp)
    print("ik errors", (e1, r1), (e2, r2), (e3, r3), (e4, r4))
    return [
        Phase(q_above, 255.0, 220),
        Phase(q_pre, 255.0, 180),
        Phase(q_grasp, 255.0, 120),
        Phase(q_grasp, 0.0, 220),
        Phase(q_lift, 0.0, 320),
        Phase(q_lift, 0.0, 260),
    ]


def main():
    phases = build_plan()
    result = evaluate(phases)
    print({k: v for k, v in result.items() if k != "sim"})
    result["sim"].save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")


if __name__ == "__main__":
    main()
