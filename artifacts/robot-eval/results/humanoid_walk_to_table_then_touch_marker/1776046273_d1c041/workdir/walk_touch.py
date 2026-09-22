import math
import numpy as np

import sim


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Controller:
    def __init__(self, sim_obj):
        self.sim = sim_obj
        self.model = sim_obj.model
        self.data = sim_obj.data
        self.joint_names = sim_obj.joint_names()
        self.name_to_act = {name: i for i, name in enumerate(sim_obj.actuator_names())}
        self.name_to_jnt = {name: i for i, name in enumerate(self.joint_names)}
        self.qpos_adr = {
            name: int(self.model.jnt_qposadr[jid])
            for name, jid in self.name_to_jnt.items()
            if jid > 0
        }
        self.qvel_adr = {
            name: int(self.model.jnt_dofadr[jid])
            for name, jid in self.name_to_jnt.items()
            if jid > 0
        }
        self.base = self.data.qpos.copy()

        self.kp = np.array([
            80, 80, 110, 140, 35,
            80, 80, 110, 140, 35,
            90,
            45, 35, 20, 25,
            45, 35, 20, 25,
        ], dtype=float)
        self.kd = np.array([
            6, 6, 8, 10, 3,
            6, 6, 8, 10, 3,
            7,
            4, 3, 2, 2,
            4, 3, 2, 2,
        ], dtype=float)
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

    def joint_pos(self, name):
        return float(self.data.qpos[self.qpos_adr[name]])

    def joint_vel(self, name):
        return float(self.data.qvel[self.qvel_adr[name]])

    def set_pose(self, t):
        q = self.base.copy()

        # Phase schedule: stabilize, walk, then reach and hold.
        if t < 120:
            walk = 0.0
            reach = 0.0
        elif t < 1150:
            walk = 1.0
            reach = 0.0
        elif t < 1550:
            walk = 0.45
            reach = 1.0
        else:
            walk = 0.15
            reach = 1.0

        # Walking cycle.
        freq = 1.7
        phase = 2.0 * math.pi * freq * (t * 0.002)
        s = math.sin(phase)
        c = math.cos(phase)
        left = 0.5 * (s + 1.0)
        right = 0.5 * (-s + 1.0)

        # Base crouch / lean.
        hip_pitch_base = -0.48
        knee_base = 0.95
        ankle_base = -0.46
        torso_base = 0.18

        # Mild side shift to make stepping asymmetric and easier to balance.
        roll_bias = 0.08 * c

        # Legs.
        q[self.qpos_adr["left_hip_yaw"]] = 0.0
        q[self.qpos_adr["right_hip_yaw"]] = 0.0
        q[self.qpos_adr["left_hip_roll"]] = 0.02 + roll_bias
        q[self.qpos_adr["right_hip_roll"]] = -0.02 - roll_bias
        q[self.qpos_adr["left_hip_pitch"]] = hip_pitch_base + 0.28 * math.sin(phase)
        q[self.qpos_adr["right_hip_pitch"]] = hip_pitch_base - 0.28 * math.sin(phase)
        q[self.qpos_adr["left_knee"]] = knee_base + 0.22 * right
        q[self.qpos_adr["right_knee"]] = knee_base + 0.22 * left
        q[self.qpos_adr["left_ankle"]] = ankle_base - 0.10 * right + 0.05 * s
        q[self.qpos_adr["right_ankle"]] = ankle_base - 0.10 * left - 0.05 * s

        # Torso stays slightly forward to encourage locomotion.
        q[self.qpos_adr["torso"]] = torso_base + 0.08 * walk

        # Arms: keep left arm a bit more forward and then fully reach near the table.
        left_sp = -0.55 - 0.70 * reach
        right_sp = -0.30 - 0.15 * reach
        left_el = 0.10 - 0.15 * reach
        right_el = 0.05 - 0.10 * reach
        q[self.qpos_adr["left_shoulder_pitch"]] = left_sp
        q[self.qpos_adr["left_shoulder_roll"]] = 0.12
        q[self.qpos_adr["left_shoulder_yaw"]] = -0.06
        q[self.qpos_adr["left_elbow"]] = left_el
        q[self.qpos_adr["right_shoulder_pitch"]] = right_sp
        q[self.qpos_adr["right_shoulder_roll"]] = -0.10
        q[self.qpos_adr["right_shoulder_yaw"]] = 0.06
        q[self.qpos_adr["right_elbow"]] = right_el

        return q

    def control(self, t):
        target = self.set_pose(t)
        ctrl = np.zeros(self.model.nu, dtype=float)
        for i, name in enumerate(self.joint_names[1:]):
            q = self.joint_pos(name)
            v = self.joint_vel(name)
            target_q = float(target[self.qpos_adr[name]])
            u = self.kp[i] * (target_q - q) - self.kd[i] * v
            ctrl[i] = clamp(u, self.ctrl_lo[i], self.ctrl_hi[i])
        return ctrl


def run():
    s = sim.Sim()
    ctrlr = Controller(s)
    steps = 1700
    for t in range(steps):
        s.data.ctrl[:] = ctrlr.control(t)
        s.step()
    s.save_final_state("/work/final_state.npz")

    print("time", s.data.time)
    print("pelvis", s.pelvis_position())
    print("torso_up", s.torso_up())
    print("distance_to_target", s.distance_to_target())
    print("distance_to_touch", s.distance_to_touch())
    print("final qpos", s.data.qpos[:10])


if __name__ == "__main__":
    run()
