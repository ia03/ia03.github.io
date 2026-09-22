import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim, BIN_CENTER, CUP_INIT_POS


def _mat_to_quat(mat33: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, mat33.reshape(9))
    return quat


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    vel = np.zeros(3, dtype=float)
    mujoco.mju_quat2Vel(vel, q, 1.0)
    return vel


@dataclass
class IKConfig:
    pos_gain: float = 4.0
    rot_gain: float = 0.15
    damping: float = 1e-3
    max_qvel: float = 2.5


class Controller:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.dt = float(self.model.opt.timestep)
        self.hand_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.left_finger_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        self.q_target = self.data.qpos[:7].copy()
        self.q_nominal = self.q_target.copy()
        self.hand_quat_des = _mat_to_quat(self.data.xmat[self.hand_body].reshape(3, 3))
        self.ik = IKConfig()

        self.gripper_open = 255.0
        self.gripper_closed = 0.0

    def hand_pose(self):
        pos = self.data.xpos[self.hand_body].copy()
        mat = self.data.xmat[self.hand_body].reshape(3, 3).copy()
        return pos, mat

    def ee_position(self):
        lp = self.data.xpos[self.left_finger_body]
        rp = self.data.xpos[self.right_finger_body]
        return 0.5 * (lp + rp)

    def _solve_ik(self, pos_des: np.ndarray, quat_des: np.ndarray | None = None):
        _, mat_cur = self.hand_pose()
        ee_cur = self.ee_position()
        pos_err = (pos_des - ee_cur) * self.ik.pos_gain

        jacp_l = np.zeros((3, self.model.nv), dtype=float)
        jacr_l = np.zeros((3, self.model.nv), dtype=float)
        jacp_r = np.zeros((3, self.model.nv), dtype=float)
        jacr_r = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacBody(self.model, self.data, jacp_l, jacr_l, self.left_finger_body)
        mujoco.mj_jacBody(self.model, self.data, jacp_r, jacr_r, self.right_finger_body)
        jacp = 0.5 * (jacp_l + jacp_r)
        jacr = 0.5 * (jacr_l + jacr_r)

        Jp = jacp[:, :7]
        Jr = jacr[:, :7]

        if quat_des is None:
            J = Jp
            err = pos_err
        else:
            quat_cur = _mat_to_quat(mat_cur)
            quat_err = _quat_mul(quat_des, _quat_conj(quat_cur))
            rot_err = _quat_to_rotvec(quat_err) * self.ik.rot_gain
            J = np.vstack([Jp, Jr])
            err = np.concatenate([pos_err, rot_err])

        JJt = J @ J.T
        JJt.flat[:: JJt.shape[0] + 1] += self.ik.damping
        sol = np.linalg.solve(JJt, err)
        dq = J.T @ sol

        dq = np.clip(dq, -self.ik.max_qvel, self.ik.max_qvel)
        self.q_target = self.q_target + dq * self.dt

        # Respect actuator control ranges (same as joint limits for arm).
        ctrlrange = self.model.actuator_ctrlrange[:7]
        self.q_target = np.clip(self.q_target, ctrlrange[:, 0], ctrlrange[:, 1])

    def set_gripper(self, cmd: float):
        self.data.ctrl[7] = float(np.clip(cmd, 0.0, 255.0))

    def set_arm_ctrl(self):
        self.data.ctrl[:7] = self.q_target

    def goto(self, pos: np.ndarray, gripper: float, quat: np.ndarray | None = None):
        self._solve_ik(pos_des=pos, quat_des=quat)
        self.set_arm_ctrl()
        self.set_gripper(gripper)


def run_episode(sim: Sim, max_steps: int = 2500):
    ctrl = Controller(sim)
    sim.reset()

    cup_start = sim.cup_position()
    # Waypoints are for the fingertip-midpoint ("EE") position.
    pregrasp = cup_start + np.array([0.0, 0.0, 0.20])
    grasp = cup_start + np.array([0.0, 0.0, 0.00])
    lift = cup_start + np.array([0.0, 0.0, 0.25])
    bin_center = np.array(BIN_CENTER, dtype=float)
    above_bin = bin_center + np.array([0.0, 0.0, 0.25])
    in_bin = bin_center + np.array([0.0, 0.0, 0.05])
    retreat = bin_center + np.array([0.0, 0.0, 0.30])

    phase = 0
    phase_steps = 0
    had_contact = False
    close_steps = 0
    release_steps = 0

    def dist_to(target):
        hp, _ = ctrl.hand_pose()
        return float(np.linalg.norm(hp - target))

    for _ in range(max_steps):
        cup_pos = sim.cup_position()
        if sim.has_gripper_cup_contact():
            had_contact = True

        if phase == 0:
            ctrl.goto(pregrasp, gripper=ctrl.gripper_open, quat=ctrl.hand_quat_des)
            if dist_to(pregrasp) < 0.03:
                phase = 1
                phase_steps = 0
        elif phase == 1:
            grasp_xy = np.array([cup_start[0], cup_start[1], grasp[2]], dtype=float)
            ctrl.goto(grasp_xy, gripper=ctrl.gripper_open, quat=None)
            if dist_to(grasp_xy) < 0.02 and phase_steps > 40:
                phase = 2
                phase_steps = 0
        elif phase == 2:
            grasp_xy = np.array([cup_start[0], cup_start[1], grasp[2]], dtype=float)
            ctrl.goto(grasp_xy, gripper=ctrl.gripper_closed, quat=None)
            close_steps += 1
            if close_steps > 120 and had_contact:
                phase = 3
                phase_steps = 0
        elif phase == 3:
            lift_xy = np.array([cup_start[0], cup_start[1], lift[2]], dtype=float)
            ctrl.goto(lift_xy, gripper=ctrl.gripper_closed, quat=None)
            if cup_pos[2] >= 0.52 or phase_steps > 400:
                phase = 4
                phase_steps = 0
        elif phase == 4:
            ctrl.goto(above_bin, gripper=ctrl.gripper_closed, quat=ctrl.hand_quat_des)
            if dist_to(above_bin) < 0.04 and phase_steps > 80:
                phase = 5
                phase_steps = 0
        elif phase == 5:
            ctrl.goto(in_bin, gripper=ctrl.gripper_closed, quat=ctrl.hand_quat_des)
            if dist_to(in_bin) < 0.03 and phase_steps > 80:
                phase = 6
                phase_steps = 0
        elif phase == 6:
            ctrl.goto(in_bin, gripper=ctrl.gripper_open, quat=ctrl.hand_quat_des)
            release_steps += 1
            if release_steps > 120:
                phase = 7
                phase_steps = 0
        elif phase == 7:
            ctrl.goto(retreat, gripper=ctrl.gripper_open, quat=ctrl.hand_quat_des)
            if phase_steps > 300:
                break

        sim.step(1)
        phase_steps += 1

    # Let things settle a bit under the final command before saving.
    sim.step(600)


def quick_metrics(sim: Sim):
    m = sim.model
    d = sim.data

    # Extra settle like grader.
    for _ in range(500):
        mujoco.mj_step(m, d)

    cup_pos = sim.cup_position()
    # Cup body xmat: last column is local z-axis in world.
    cup_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cup")
    cup_z_axis = d.xmat[cup_id].reshape(3, 3)[:, 2]
    upright = float(abs(cup_z_axis[2]))
    inside = (abs(cup_pos[0] - BIN_CENTER[0]) <= 0.046) and (abs(cup_pos[1] - BIN_CENTER[1]) <= 0.046)
    return {
        "cup_pos": cup_pos,
        "upright": upright,
        "inside_bin": bool(inside),
        "contact_now": bool(sim.has_gripper_cup_contact()),
        "time": float(d.time),
        "ctrl_steps": int(len(sim._ctrl_trace)),
    }


if __name__ == "__main__":
    t0 = time.time()
    sim = Sim()
    run_episode(sim)
    metrics = quick_metrics(sim)
    print("metrics:", metrics)
    sim.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")
    print("elapsed_s:", round(time.time() - t0, 2))
