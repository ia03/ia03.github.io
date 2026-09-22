"""
Heuristic controller for the Humanoid Cradle Box task.

Writes /work/final_state.npz via Sim.save_final_state().
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


@dataclass
class JointMap:
    qpos_adr: int
    qvel_adr: int


class PDMotorController:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data

        self.actuator_names = sim.actuator_names()
        self._maps: list[JointMap] = []
        for act_id, name in enumerate(self.actuator_names):
            # For joint motors, actuator_trnid[:,0] is the joint id.
            j_id = int(self.model.actuator_trnid[act_id, 0])
            if j_id < 0:
                j_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
            qpos_adr = int(self.model.jnt_qposadr[j_id])
            qvel_adr = int(self.model.jnt_dofadr[j_id])
            self._maps.append(JointMap(qpos_adr=qpos_adr, qvel_adr=qvel_adr))

        # Desired joint angles in actuator order (matches sim.home_ctrl()).
        self.q_des = sim.home_ctrl().copy()
        self.qd_des = np.zeros_like(self.q_des)

        # Per-joint PD gains (roughly tuned).
        self.kp = np.array(
            [
                120, 120, 160, 220, 120,  # left leg
                120, 120, 160, 220, 120,  # right leg
                180,  # torso
                60, 60, 40, 40,  # left arm
                60, 60, 40, 40,  # right arm
            ],
            dtype=float,
        )
        self.kd = np.array(
            [
                6, 6, 8, 10, 6,
                6, 6, 8, 10, 6,
                10,
                2, 2, 2, 2,
                2, 2, 2, 2,
            ],
            dtype=float,
        )

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

    def set_q_des(self, q_des: np.ndarray):
        self.q_des[:] = q_des

    def _pelvis_roll_pitch(self) -> tuple[float, float]:
        # Pelvis free joint quaternion is qpos[3:7] in (w, x, y, z).
        quat = self.data.qpos[3:7].copy()
        mat = np.zeros(9, dtype=float)
        mujoco.mju_quat2Mat(mat, quat)
        R = mat.reshape(3, 3)
        # ZYX yaw-pitch-roll extraction; pitch about world Y, roll about world X.
        pitch = math.asin(float(-R[2, 0]))
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        return roll, pitch

    def step(self):
        q = np.array([self.data.qpos[m.qpos_adr] for m in self._maps], dtype=float)
        qd = np.array([self.data.qvel[m.qvel_adr] for m in self._maps], dtype=float)
        # Gravity/Coriolis compensation around current state.
        bias = np.array([self.data.qfrc_bias[m.qvel_adr] for m in self._maps], dtype=float)
        torque = bias + self.kp * (self.q_des - q) + self.kd * (self.qd_des - qd)

        # Simple balancing torques based on pelvis roll/pitch.
        roll, pitch = self._pelvis_roll_pitch()
        angvel = self.data.qvel[3:6].copy()
        roll_rate = float(angvel[0])
        pitch_rate = float(angvel[1])

        # Tuned heuristically for this model; acts like an ankle/hip stabilizer.
        kpr, kdr = 120.0, 10.0
        kpp, kdp = 140.0, 12.0
        roll_cmd = -(kpr * roll + kdr * roll_rate)
        # Sign convention: negative pitch here corresponds to tipping backward in this scene.
        # Apply corrective torques in the same sign as pitch to counteract.
        pitch_cmd = (kpp * pitch + kdp * pitch_rate)

        # Indices in actuator order.
        L_HIP_ROLL, R_HIP_ROLL = 1, 6
        L_HIP_PITCH, R_HIP_PITCH = 2, 7
        L_ANKLE, R_ANKLE = 4, 9

        torque[L_HIP_ROLL] += roll_cmd
        torque[R_HIP_ROLL] -= roll_cmd
        torque[L_HIP_PITCH] += 0.6 * pitch_cmd
        torque[R_HIP_PITCH] += 0.6 * pitch_cmd
        torque[L_ANKLE] += 0.8 * pitch_cmd
        torque[R_ANKLE] += 0.8 * pitch_cmd

        torque = np.clip(torque, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[:] = torque


def make_cradle_pose(home_q: np.ndarray) -> np.ndarray:
    q = home_q.copy()
    name_to_idx = {
        "left_shoulder_pitch": 11,
        "left_shoulder_roll": 12,
        "left_shoulder_yaw": 13,
        "left_elbow": 14,
        "right_shoulder_pitch": 15,
        "right_shoulder_roll": 16,
        "right_shoulder_yaw": 17,
        "right_elbow": 18,
        "torso": 10,
    }
    # Bring arms forward and slightly inward to contact the box.
    q[name_to_idx["left_shoulder_pitch"]] = 0.55
    q[name_to_idx["right_shoulder_pitch"]] = 0.55
    q[name_to_idx["left_shoulder_roll"]] = 0.25
    q[name_to_idx["right_shoulder_roll"]] = -0.25
    q[name_to_idx["left_elbow"]] = 1.05
    q[name_to_idx["right_elbow"]] = 1.05
    # Slight forward torso pitch to encourage forward motion.
    q[name_to_idx["torso"]] = 0.15
    return q


def gait_delta(t: float) -> np.ndarray:
    """Open-loop small-amplitude gait in joint-angle space (19-dim)."""
    period = 0.70
    phase = 2.0 * math.pi * (t / period)
    s = math.sin(phase)
    c = math.cos(phase)

    delta = np.zeros(19, dtype=float)

    # Joint indices in actuator order.
    L_HIP_YAW, L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE = 0, 1, 2, 3, 4
    R_HIP_YAW, R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_ANKLE = 5, 6, 7, 8, 9

    # Side-to-side weight shift.
    roll = 0.12 * s
    delta[L_HIP_ROLL] += -roll
    delta[R_HIP_ROLL] += roll

    # Alternating swing: hip pitch and knee flex. Sign chosen to match common convention
    # (positive hip_pitch flexes forward).
    hip_amp = 0.35
    knee_amp = 0.55
    ankle_amp = 0.25

    delta[L_HIP_PITCH] += hip_amp * s
    delta[R_HIP_PITCH] += -hip_amp * s

    # Flex knee when leg swings forward (use cosine to lead/lag).
    swing_left = max(0.0, c)
    swing_right = max(0.0, -c)
    delta[L_KNEE] += knee_amp * swing_left
    delta[R_KNEE] += knee_amp * swing_right

    # Ankle compensation to keep swing foot from toe-dragging too much.
    delta[L_ANKLE] += ankle_amp * swing_left
    delta[R_ANKLE] += ankle_amp * swing_right

    # Slight yaw to damp symmetric drift.
    delta[L_HIP_YAW] += 0.02 * s
    delta[R_HIP_YAW] += -0.02 * s

    return delta


def run(seed_final_state_path: str = "/work/final_state.npz", total_steps: int = 1800):
    sim = Sim()
    ctrl = PDMotorController(sim)

    # Stable forward-leaning base posture (joint angles), in actuator order.
    base = ctrl.q_des.copy()
    for idx in (2, 7):  # left/right hip_pitch
        base[idx] += -0.45

    dt = float(sim.model.opt.timestep)
    stable_steps = 200          # settle from reset
    gait_steps = 600            # brief walk/shuffle phase
    gait_scale = 0.33           # tuned to stay stable while reaching x checkpoints
    total_steps = int(total_steps)

    for i in range(total_steps):
        t = i * dt
        q_des = base.copy()
        if stable_steps <= i < stable_steps + gait_steps:
            q_des += gait_scale * gait_delta(t - stable_steps * dt)
        ctrl.set_q_des(q_des)
        ctrl.step()
        sim.step(1)

    sim.save_final_state(seed_final_state_path)
    return sim


if __name__ == "__main__":
    t0 = time.time()
    sim = run()
    print("time_s:", time.time() - t0)
    print("steps:", len(sim._ctrl_trace))
    print("pelvis:", sim.pelvis_position(), "torso_up:", sim.torso_up(), "dist_target:", sim.distance_to_target())
    print("box:", sim.box_position(), "box_dist_target:", sim.box_distance_to_target(), "box_to_pelvis:", sim.box_distance_to_pelvis())
