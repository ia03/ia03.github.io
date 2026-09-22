"""Open-loop gait + reach controller for the H1 walk-and-touch task.

Writes an evaluation artifact via sim.save_final_state("/work/final_state.npz").
"""

from __future__ import annotations

import math
import numpy as np
import mujoco

from sim import Sim


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * a + t * b


def _actuated_joint_state(sim: Sim) -> tuple[np.ndarray, np.ndarray]:
    """Return joint position/velocity vectors in actuator order."""
    m = sim.model
    d = sim.data
    q = np.zeros(m.nu, dtype=float)
    qd = np.zeros(m.nu, dtype=float)
    for i in range(m.nu):
        jnt_id = int(m.actuator_trnid[i, 0])
        qadr = int(m.jnt_qposadr[jnt_id])
        dadr = int(m.jnt_dofadr[jnt_id])
        q[i] = float(d.qpos[qadr])
        qd[i] = float(d.qvel[dadr])
    return q, qd


def bias_torque(sim: Sim) -> np.ndarray:
    """Return qfrc_bias projected into actuator order (hinge dof)."""
    m = sim.model
    d = sim.data
    tau = np.zeros(m.nu, dtype=float)
    # Ensure bias is up to date for the current state.
    mujoco.mj_forward(m, d)
    for i in range(m.nu):
        jnt_id = int(m.actuator_trnid[i, 0])
        dadr = int(m.jnt_dofadr[jnt_id])
        tau[i] = float(d.qfrc_bias[dadr])
    return tau


def pd_torque(sim: Sim, q_des: np.ndarray, kp: np.ndarray, kd: np.ndarray, ff: np.ndarray) -> np.ndarray:
    q, qd = _actuated_joint_state(sim)
    tau = ff + kp * (q_des - q) - kd * qd
    ctrl_lo = sim.model.actuator_ctrlrange[:, 0]
    ctrl_hi = sim.model.actuator_ctrlrange[:, 1]
    return np.clip(tau, ctrl_lo, ctrl_hi)


def torso_pitch_roll(sim: Sim) -> tuple[float, float]:
    """Approximate torso pitch/roll from torso rotation matrix.

    Assumes sim.data.xmat is a body-to-world rotation in row-major form, and
    extracts angles from the torso z-axis direction.
    """
    R = sim.data.xmat[sim.torso_body_id].reshape(3, 3)
    # Torso z-axis expressed in world.
    zx, zy, zz = float(R[0, 2]), float(R[1, 2]), float(R[2, 2])
    pitch = math.atan2(zx, max(1e-9, zz))
    roll = math.atan2(-zy, max(1e-9, zz))
    return pitch, roll


def main() -> None:
    sim = Sim()
    dt = float(sim.model.opt.timestep)

    home = sim.home_ctrl()
    # Empirically stable standing setpoint (offset from home).
    stand_pose = home.copy()

    # Control schedule (in seconds).
    t_stand = 0.8
    t_walk = 6.0
    t_reach_hold = 1.2
    total_time = t_stand + t_walk + t_reach_hold
    total_steps = int(total_time / dt)

    # Indices (Sim actuator order matches printed names).
    (
        L_HIP_YAW,
        L_HIP_ROLL,
        L_HIP_PITCH,
        L_KNEE,
        L_ANKLE,
        R_HIP_YAW,
        R_HIP_ROLL,
        R_HIP_PITCH,
        R_KNEE,
        R_ANKLE,
        TORSO,
        L_SHOULDER_PITCH,
        L_SHOULDER_ROLL,
        L_SHOULDER_YAW,
        L_ELBOW,
        R_SHOULDER_PITCH,
        R_SHOULDER_ROLL,
        R_SHOULDER_YAW,
        R_ELBOW,
    ) = range(sim.model.nu)

    # Stable posture offsets (empirically found).
    stand_pose[L_HIP_PITCH] = home[L_HIP_PITCH] - 0.10
    stand_pose[R_HIP_PITCH] = home[R_HIP_PITCH] - 0.10
    stand_pose[L_KNEE] = home[L_KNEE] + 0.00
    stand_pose[R_KNEE] = home[R_KNEE] + 0.00
    stand_pose[L_ANKLE] = home[L_ANKLE] + 0.00
    stand_pose[R_ANKLE] = home[R_ANKLE] + 0.00
    stand_pose[TORSO] = 0.02

    # Walking parameters.
    freq = 0.95  # Hz
    w = 2.0 * math.pi * freq
    hip_swing = 0.34
    hip_stance = 0.20
    knee_swing = 0.75
    knee_stance = -0.10
    ankle_swing = -0.25
    ankle_stance = 0.10
    roll_shift = 0.10
    torso_lean = 0.06

    # Reach parameters (right arm).
    reach_pose = stand_pose.copy()
    reach_pose[TORSO] = torso_lean + 0.10
    reach_pose[R_SHOULDER_PITCH] = 1.35
    reach_pose[R_SHOULDER_ROLL] = -0.15
    reach_pose[R_SHOULDER_YAW] = 0.10
    reach_pose[R_ELBOW] = 0.90

    walk_pose = stand_pose.copy()
    walk_pose[TORSO] = torso_lean

    # Joint-space PD gains (torque actuators).
    kp = np.array(
        [
            900, 900, 1200, 2000, 1300,  # left leg
            900, 900, 1200, 2000, 1300,  # right leg
            900,  # torso
            70, 70, 50, 45,  # left arm
            70, 70, 50, 45,  # right arm
        ],
        dtype=float,
    )
    kd = 0.08 * kp

    prev_pitch, prev_roll = torso_pitch_roll(sim)

    for step in range(total_steps):
        t = step * dt
        pelvis_x = float(sim.pelvis_position()[0])
        pitch, roll = torso_pitch_roll(sim)
        pitch_rate = (pitch - prev_pitch) / dt
        roll_rate = (roll - prev_roll) / dt
        prev_pitch, prev_roll = pitch, roll

        if t < t_stand:
            alpha = t / max(1e-6, t_stand)
            q_des = lerp(home, stand_pose, alpha)
        else:
            gait_t = t - t_stand
            phase = w * gait_t
            s = math.sin(phase)

            left_swing = s > 0.0
            swing = abs(s)
            stance = 1.0 - swing

            q_des = walk_pose.copy()

            # Shift weight toward stance leg.
            shift = roll_shift * (1.0 if left_swing else -1.0)
            q_des[L_HIP_ROLL] = -shift
            q_des[R_HIP_ROLL] = -shift

            if left_swing:
                q_des[L_HIP_PITCH] = stand_pose[L_HIP_PITCH] - hip_swing * swing
                q_des[L_KNEE] = stand_pose[L_KNEE] + knee_swing * swing
                q_des[L_ANKLE] = stand_pose[L_ANKLE] + ankle_swing * swing

                q_des[R_HIP_PITCH] = stand_pose[R_HIP_PITCH] + hip_stance * swing
                q_des[R_KNEE] = stand_pose[R_KNEE] + knee_stance * swing
                q_des[R_ANKLE] = stand_pose[R_ANKLE] + ankle_stance * swing
            else:
                q_des[R_HIP_PITCH] = stand_pose[R_HIP_PITCH] - hip_swing * swing
                q_des[R_KNEE] = stand_pose[R_KNEE] + knee_swing * swing
                q_des[R_ANKLE] = stand_pose[R_ANKLE] + ankle_swing * swing

                q_des[L_HIP_PITCH] = stand_pose[L_HIP_PITCH] + hip_stance * swing
                q_des[L_KNEE] = stand_pose[L_KNEE] + knee_stance * swing
                q_des[L_ANKLE] = stand_pose[L_ANKLE] + ankle_stance * swing

            # Start reaching once we're near the table; stop stepping at the end.
            reach_gate = float(np.clip((pelvis_x - 0.85) / 0.25, 0.0, 1.0))
            hold_gate = float(np.clip((gait_t - (t_walk - 1.0)) / 1.0, 0.0, 1.0))
            q_des = lerp(q_des, stand_pose, hold_gate)
            q_des = lerp(q_des, reach_pose, reach_gate * (1.0 - hold_gate))

        # Mild torso stabilization to keep torso_up high.
        pitch_u = 0.35 * (0.06 - pitch) - 0.03 * pitch_rate
        roll_u = 0.35 * (0.0 - roll) - 0.03 * roll_rate
        q_des[TORSO] = float(np.clip(q_des[TORSO] + 0.25 * pitch_u, -0.6, 0.6))
        q_des[L_HIP_ROLL] = q_des[L_HIP_ROLL] + 0.15 * roll_u
        q_des[R_HIP_ROLL] = q_des[R_HIP_ROLL] + 0.15 * roll_u

        ff = bias_torque(sim)
        sim.data.ctrl[:] = pd_torque(sim, q_des, kp, kd, ff)
        sim.step(1)

    # Report end diagnostics and save artifact.
    print("final time", float(sim.data.time))
    print("final pelvis", sim.pelvis_position())
    print("final torso_up", sim.torso_up())
    print("final dist_target", sim.distance_to_target())
    print("best-touch (final) dist_touch", sim.distance_to_touch())
    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
