import math
import os
import time

import mujoco
import numpy as np

import sim


ARM_JOINTS = 7
HAND_BODY = "hand"


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


class WaypointController:
    def __init__(self, s: sim.Sim):
        self.s = s
        self.hand_bid = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.arm_qmin = s.model.jnt_range[:ARM_JOINTS, 0].copy()
        self.arm_qmax = s.model.jnt_range[:ARM_JOINTS, 1].copy()
        self.q_home = s.data.qpos[:ARM_JOINTS].copy()
        self.R_target = s.data.xmat[self.hand_bid].reshape(3, 3).copy()

    @staticmethod
    def _rotvec_from_matrix(R):
        tr = float(np.trace(R))
        cos_theta = clamp((tr - 1.0) * 0.5, -1.0, 1.0)
        theta = math.acos(cos_theta)
        if theta < 1e-6:
            return np.array(
                [
                    0.5 * (R[2, 1] - R[1, 2]),
                    0.5 * (R[0, 2] - R[2, 0]),
                    0.5 * (R[1, 0] - R[0, 1]),
                ]
            )
        denom = 2.0 * math.sin(theta)
        axis = np.array(
            [
                (R[2, 1] - R[1, 2]) / denom,
                (R[0, 2] - R[2, 0]) / denom,
                (R[1, 0] - R[0, 1]) / denom,
            ]
        )
        return axis * theta

    def solve_ik(self, target_pos, step_size=0.18, damping=1e-3, rot_weight=0.25, posture_gain=0.03):
        q = self.s.data.qpos[:ARM_JOINTS].copy()
        self.s.data.qpos[:ARM_JOINTS] = q
        self.s.data.qpos[ARM_JOINTS:] = self.s.data.qpos[ARM_JOINTS:]
        mujoco.mj_forward(self.s.model, self.s.data)
        pos = self.s.data.xpos[self.hand_bid].copy()
        R = self.s.data.xmat[self.hand_bid].reshape(3, 3).copy()
        err = target_pos - pos
        rot_err = self._rotvec_from_matrix(self.R_target @ R.T)
        jacp = np.zeros((3, self.s.model.nv))
        jacr = np.zeros((3, self.s.model.nv))
        mujoco.mj_jacBody(self.s.model, self.s.data, jacp, jacr, self.hand_bid)
        J = np.vstack([jacp[:, :ARM_JOINTS], rot_weight * jacr[:, :ARM_JOINTS]])
        e = np.concatenate([err, rot_weight * rot_err])
        A = J @ J.T + damping * np.eye(6)
        dq = step_size * (J.T @ np.linalg.solve(A, e)) + posture_gain * (self.q_home - q)
        q = clamp(q + dq, self.arm_qmin, self.arm_qmax)
        return q

    def control_step(self, target_pos, gripper_open):
        qdes = self.solve_ik(target_pos)
        ctrl = np.zeros(self.s.model.nu)
        ctrl[:ARM_JOINTS] = qdes
        ctrl[7] = 255 if gripper_open else 0
        return ctrl


def interp(a, b, t):
    return a * (1 - t) + b * t


def main():
    s = sim.Sim()
    ctrl = WaypointController(s)

    # Waypoints chosen to stay high over the wall, then return to the robot side.
    p_pre = np.array([0.58, 0.16, 0.72])
    p_grasp = np.array([0.60, 0.16, 0.505])
    p_lift = np.array([0.60, 0.16, 0.69])
    p_retreat = np.array([0.41, 0.02, 0.68])
    p_hold = np.array([0.38, 0.00, 0.64])

    total_steps = 1100
    stage_steps = [240, 170, 130, 220, 340]
    assert sum(stage_steps) == total_steps

    stages = []
    stages += [(p_pre, True)] * stage_steps[0]
    stages += [(p_grasp, True)] * stage_steps[1]
    stages += [(p_grasp, False)] * stage_steps[2]
    stages += [(p_lift, False)] * stage_steps[3]
    stages += [(p_retreat, False)] * stage_steps[4]

    # Smooth transition from pregrasp to grasp to reduce jerk.
    for i, (target, open_flag) in enumerate(stages):
        if i < 120:
            tgt = interp(np.array([0.52, 0.12, 0.72]), p_pre, i / 120.0)
        elif i < 240:
            t = (i - 120) / 120.0
            tgt = interp(p_pre, p_grasp, t)
        elif i < 410:
            tgt = p_grasp
        elif i < 540:
            t = (i - 410) / 130.0
            tgt = interp(p_grasp, p_lift, t)
        elif i < 760:
            t = (i - 540) / 220.0
            tgt = interp(p_lift, p_retreat, t)
        else:
            t = (i - 760) / 340.0
            tgt = interp(p_retreat, p_hold, t)
        s.data.ctrl[:] = ctrl.control_step(tgt, open_flag)
        s.step(1)
        if i % 100 == 0:
            print(
                f"step {i:4d} time={s.data.time:.2f} hand={s.data.xpos[ctrl.hand_bid]} "
                f"cup={s.cup_position()} contact={s.has_gripper_cup_contact()} ctrl7={s.data.ctrl[7]:.1f}"
            )

    # Keep holding for a short settle while still recording control history.
    for i in range(100):
        s.data.ctrl[:] = ctrl.control_step(p_hold, False)
        s.step(1)

    print("final hand", s.data.xpos[ctrl.hand_bid])
    print("final cup", s.cup_position())
    print("contact", s.has_gripper_cup_contact())
    s.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
