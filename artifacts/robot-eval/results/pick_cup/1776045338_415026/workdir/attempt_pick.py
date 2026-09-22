import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim, SUCCESS_CUP_Z


ARM_JOINTS = 7
DT = 0.002


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def geom_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def quat_from_mat(mat):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.reshape(9))
    return quat


def orientation_error(current_mat, target_mat):
    qc = quat_from_mat(current_mat)
    qt = quat_from_mat(target_mat)
    q_conj = np.array([qc[0], -qc[1], -qc[2], -qc[3]])
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, qt, q_conj)
    if q_err[0] < 0:
        q_err *= -1
    return 2.0 * q_err[1:]


def finger_contact_fraction(sim, steps=500):
    m, d = sim.model, sim.data
    cup_gid = geom_id(m, "cup_geom")
    left_bid = body_id(m, "left_finger")
    right_bid = body_id(m, "right_finger")
    touched = 0
    for _ in range(steps):
        sim.step(1)
        has_contact = False
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = con.geom1, con.geom2
            if g1 == cup_gid or g2 == cup_gid:
                other = g2 if g1 == cup_gid else g1
                other_body = m.geom_bodyid[other]
                if other_body == left_bid or other_body == right_bid:
                    has_contact = True
                    break
        touched += float(has_contact)
    return touched / steps


def replay_score(ctrl_trace):
    sim = Sim()
    if len(ctrl_trace):
        for ctrl in ctrl_trace:
            sim.data.ctrl[:] = ctrl
            sim.step(1)
    contact_fraction = finger_contact_fraction(sim, 500)
    final_z = sim.cup_position()[2]
    height_progress = np.clip((final_z - 0.435) / (SUCCESS_CUP_Z - 0.435), 0.0, 1.0)
    contact_progress = np.clip(contact_fraction / 0.5, 0.0, 1.0)
    progress = 0.5 * height_progress + 0.5 * contact_progress
    return {
        "final_z": float(final_z),
        "contact_fraction": float(contact_fraction),
        "progress": float(progress),
    }


@dataclass
class Candidate:
    grasp_x: float
    grasp_y: float
    pregrasp_z: float
    grasp_z: float
    lift_z: float
    open_ctrl: float
    close_ctrl: float
    descend_steps: int
    close_steps: int
    lift_steps: int
    settle_steps: int


class Controller:
    def __init__(self):
        self.sim = Sim()
        self.m = self.sim.model
        self.d = self.sim.data
        self.hand_bid = body_id(self.m, "hand")
        self.home_q = self.d.qpos[:ARM_JOINTS].copy()
        self.home_hand_mat = self.d.xmat[self.hand_bid].reshape(3, 3).copy()
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))

    def solve_ik(self, target_pos, target_mat, q_init=None, steps=120, tol=1e-4):
        q = self.d.qpos[:ARM_JOINTS].copy() if q_init is None else q_init.copy()
        limits = self.m.jnt_range[:ARM_JOINTS].copy()
        mid = limits.mean(axis=1)
        span = limits[:, 1] - limits[:, 0]
        for _ in range(steps):
            self.d.qpos[:ARM_JOINTS] = q
            self.d.qvel[:] = 0
            mujoco.mj_forward(self.m, self.d)
            pos = self.d.xpos[self.hand_bid].copy()
            mat = self.d.xmat[self.hand_bid].reshape(3, 3).copy()
            pos_err = target_pos - pos
            rot_err = orientation_error(mat, target_mat)
            err = np.concatenate([pos_err, 0.25 * rot_err])
            if np.linalg.norm(err[:3]) < tol and np.linalg.norm(err[3:]) < 2e-3:
                break
            mujoco.mj_jacBody(self.m, self.d, self.jacp, self.jacr, self.hand_bid)
            jac = np.vstack([self.jacp[:, :ARM_JOINTS], 0.25 * self.jacr[:, :ARM_JOINTS]])
            reg = 2e-3 * np.eye(jac.shape[0])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + reg, err)
            dq += -2e-3 * (q - mid) / np.maximum(span, 1e-3)
            q = np.clip(q + 0.8 * dq, limits[:, 0], limits[:, 1])
        return q

    def goto_q(self, q_target, gripper_ctrl, steps):
        q_start = self.d.ctrl[:ARM_JOINTS].copy()
        g_start = float(self.d.ctrl[7])
        for i in range(steps):
            a = (i + 1) / steps
            self.d.ctrl[:ARM_JOINTS] = (1 - a) * q_start + a * q_target
            self.d.ctrl[7] = (1 - a) * g_start + a * gripper_ctrl
            self.sim.step(1)

    def hold(self, arm_target, gripper_ctrl, steps):
        self.d.ctrl[:ARM_JOINTS] = arm_target
        self.d.ctrl[7] = gripper_ctrl
        self.sim.step(steps)

    def run_candidate(self, cand: Candidate):
        self.sim.reset()
        self.d.ctrl[:ARM_JOINTS] = self.home_q
        self.d.ctrl[7] = 0.0
        self.sim.step(5)

        cup = self.sim.cup_position().copy()
        pre_pos = np.array([cup[0] + cand.grasp_x, cup[1] + cand.grasp_y, cand.pregrasp_z])
        grasp_pos = np.array([cup[0] + cand.grasp_x, cup[1] + cand.grasp_y, cand.grasp_z])
        lift_pos = np.array([cup[0] + cand.grasp_x, cup[1] + cand.grasp_y, cand.lift_z])

        q_open = self.solve_ik(pre_pos, self.home_hand_mat, q_init=self.home_q)
        q_grasp = self.solve_ik(grasp_pos, self.home_hand_mat, q_init=q_open)
        q_lift = self.solve_ik(lift_pos, self.home_hand_mat, q_init=q_grasp)

        self.goto_q(q_open, cand.open_ctrl, 300)
        self.goto_q(q_grasp, cand.open_ctrl, cand.descend_steps)
        self.hold(q_grasp, cand.open_ctrl, 40)
        self.hold(q_grasp, cand.close_ctrl, cand.close_steps)
        self.hold(q_grasp, cand.close_ctrl, 100)
        self.goto_q(q_lift, cand.close_ctrl, cand.lift_steps)
        self.hold(q_lift, cand.close_ctrl, cand.settle_steps)

        ctrl_trace = np.array(self.sim._ctrl_trace, dtype=float)
        score = replay_score(ctrl_trace)
        return score, ctrl_trace, self.sim.cup_position().copy()


def save_trace(ctrl_trace):
    sim = Sim()
    for ctrl in ctrl_trace:
        sim.data.ctrl[:] = ctrl
        sim.step(1)
    sim.save_final_state("/work/final_state.npz")


def main():
    controller = Controller()
    candidates = [
        Candidate(0.0, 0.0, 0.58, 0.505, 0.66, 180.0, 0.0, 500, 250, 500, 700),
        Candidate(0.0, 0.0, 0.56, 0.495, 0.62, 180.0, 0.0, 450, 300, 450, 700),
        Candidate(0.0, 0.0, 0.55, 0.490, 0.60, 150.0, 0.0, 450, 350, 500, 700),
        Candidate(0.004, 0.0, 0.56, 0.495, 0.63, 180.0, 0.0, 450, 350, 500, 700),
        Candidate(-0.004, 0.0, 0.56, 0.495, 0.63, 180.0, 0.0, 450, 350, 500, 700),
        Candidate(0.0, 0.004, 0.56, 0.495, 0.63, 180.0, 0.0, 450, 350, 500, 700),
        Candidate(0.0, -0.004, 0.56, 0.495, 0.63, 180.0, 0.0, 450, 350, 500, 700),
        Candidate(0.0, 0.0, 0.545, 0.485, 0.59, 135.0, 0.0, 450, 450, 550, 800),
    ]

    best = None
    for idx, cand in enumerate(candidates):
        score, ctrl_trace, cup_pos = controller.run_candidate(cand)
        print(
            f"candidate {idx}: progress={score['progress']:.3f} "
            f"final_z={score['final_z']:.3f} contact={score['contact_fraction']:.3f} "
            f"cup_now={cup_pos}"
        )
        if best is None or score["progress"] > best[0]["progress"]:
            best = (score, ctrl_trace, cand)
            save_trace(ctrl_trace)
            print(f"saved best candidate {idx} to /work/final_state.npz")
        if score["final_z"] >= SUCCESS_CUP_Z and score["contact_fraction"] >= 0.5:
            break

    if best is not None:
        print("best", best[0], best[2])
        assert os.path.exists("/work/final_state.npz")


if __name__ == "__main__":
    main()
