"""First-pass controller for the H1 walk-and-touch task.

Writes a replayable control history via sim.save_final_state("/work/final_state.npz").
"""

from __future__ import annotations

import numpy as np

from sim import Sim, TOUCH_POINT


def clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def run(seed: int = 0, steps: int = 5200) -> None:
    rng = np.random.default_rng(seed)
    sim = Sim()

    # Control limits are stored on actuators; default to +/- inf if unspecified.
    ctrlrange = np.asarray(sim.model.actuator_ctrlrange, dtype=float)
    lo = ctrlrange[:, 0].copy()
    hi = ctrlrange[:, 1].copy()
    lo[~np.isfinite(lo)] = -1e9
    hi[~np.isfinite(hi)] = 1e9

    # Note: this task uses torque motors (ctrl ~= torque). We'll do our own PD.
    home_q = sim.home_ctrl()
    ctrl = np.zeros_like(home_q)
    sim.data.ctrl[:] = ctrl

    # A slightly more upright nominal pose than the provided crouched "home".
    nominal_q = home_q.copy()
    nominal_q[2] = -0.22
    nominal_q[7] = -0.22
    nominal_q[3] = 0.48
    nominal_q[8] = 0.48
    nominal_q[4] = -0.26
    nominal_q[9] = -0.26

    # Simple open-loop phases:
    # 1) Stand and settle.
    # 2) Walk-ish gait with hip/knee/roll oscillations + slight forward lean.
    # 3) Hold near target while reaching right arm forward toward the touch marker.
    stand_steps = 1600
    walk_steps = 2800
    reach_steps = steps - stand_steps - walk_steps
    reach_steps = max(reach_steps, 0)

    # Joint-space PD gains (Nm/rad-ish); tuned for stability over aggressiveness.
    kp = np.array(
        [
            120,  # left_hip_yaw
            160,  # left_hip_roll
            200,  # left_hip_pitch
            240,  # left_knee
            80,   # left_ankle
            120,  # right_hip_yaw
            160,  # right_hip_roll
            200,  # right_hip_pitch
            240,  # right_knee
            80,   # right_ankle
            140,  # torso
            60,   # left_shoulder_pitch
            40,   # left_shoulder_roll
            40,   # left_shoulder_yaw
            30,   # left_elbow
            60,   # right_shoulder_pitch
            40,   # right_shoulder_roll
            40,   # right_shoulder_yaw
            30,   # right_elbow
        ],
        dtype=float,
    )
    kd = 2.0 * np.sqrt(kp) * 0.35

    # Gait parameters tuned to be conservative (avoid immediate falls).
    gait_hz = 1.2
    hip_pitch_amp = 0.28
    knee_lift_amp = 0.28
    hip_roll_amp = 0.10
    lean_extra = -0.12

    # Reaching posture (right arm); keep left arm near neutral for balance.
    reach_right_shoulder_pitch = 0.9
    reach_right_shoulder_roll = 0.15
    reach_right_shoulder_yaw = -0.2
    reach_right_elbow = 1.0
    reach_torso = -0.15

    # Add a tiny asymmetry so we don't end up in a perfect in-place oscillation.
    phase_bias = float(rng.uniform(-0.15, 0.15))

    for k in range(steps):
        t = float(sim.data.time)

        q = np.asarray(sim.data.qpos[7 : 7 + sim.model.nu], dtype=float)
        qd = np.asarray(sim.data.qvel[6 : 6 + sim.model.nu], dtype=float)

        # Start from the keyframed crouch and ease toward nominal to avoid an initial "kick".
        alpha = float(np.clip(k / 800.0, 0.0, 1.0))
        q_des = (1.0 - alpha) * home_q + alpha * nominal_q

        if k < stand_steps:
            # Slightly bend knees/ankles to settle contacts.
            q_des[3] = q_des[3] + 0.03  # left_knee
            q_des[8] = q_des[8] + 0.03  # right_knee
        elif k < stand_steps + walk_steps:
            tau = 2.0 * np.pi * gait_hz * (t - stand_steps * sim.model.opt.timestep)
            s = np.sin(tau + phase_bias)
            c = np.cos(tau + phase_bias)

            # Weight shift via hip roll: shift opposite swing leg.
            q_des[1] = home_q[1] + hip_roll_amp * c  # left_hip_roll
            q_des[6] = home_q[6] - hip_roll_amp * c  # right_hip_roll

            # Hip pitch: stance leg extends (more negative), swing leg comes forward (less negative).
            q_des[2] = nominal_q[2] + lean_extra - hip_pitch_amp * s  # left_hip_pitch
            q_des[7] = nominal_q[7] + lean_extra + hip_pitch_amp * s  # right_hip_pitch

            # Knee lift on swing leg only (half-wave rectification).
            left_swing = max(0.0, s)
            right_swing = max(0.0, -s)
            q_des[3] = nominal_q[3] - knee_lift_amp * left_swing  # left_knee
            q_des[8] = nominal_q[8] - knee_lift_amp * right_swing  # right_knee

            # Ankles follow to keep feet closer to flat.
            q_des[4] = nominal_q[4] + 0.6 * knee_lift_amp * left_swing  # left_ankle
            q_des[9] = nominal_q[9] + 0.6 * knee_lift_amp * right_swing  # right_ankle

            # Tiny torso pitch forward to encourage progress.
            q_des[10] = -0.08
        else:
            # Reach phase: keep a crouched, slightly leaned-forward stance.
            q_des[2] = nominal_q[2] + lean_extra - 0.10  # left_hip_pitch
            q_des[7] = nominal_q[7] + lean_extra - 0.10  # right_hip_pitch
            q_des[3] = nominal_q[3] + 0.10
            q_des[8] = nominal_q[8] + 0.10
            q_des[4] = nominal_q[4] - 0.05
            q_des[9] = nominal_q[9] - 0.05

            # Bring the right elbow toward the marker.
            q_des[10] = reach_torso
            q_des[15] = reach_right_shoulder_pitch
            q_des[16] = reach_right_shoulder_roll
            q_des[17] = reach_right_shoulder_yaw
            q_des[18] = reach_right_elbow

        # Simple balance assist in the sagittal plane based on torso pitch + velocity.
        torso_mat = np.asarray(sim.data.xmat[sim.torso_body_id], dtype=float).reshape(3, 3)
        torso_fwd = torso_mat[:, 0]
        pitch = float(np.arctan2(-torso_fwd[2], torso_fwd[0]))  # + => leaning forward
        vx = float(sim.data.qvel[0])
        wy = float(sim.data.qvel[4])
        pitch_target = 0.03 if k < stand_steps else 0.10
        pitch_err = pitch - pitch_target
        pitch_correction = 0.9 * pitch_err + 0.18 * wy + 0.25 * vx
        q_des[4] += pitch_correction
        q_des[9] += pitch_correction
        q_des[2] -= 0.55 * pitch_correction
        q_des[7] -= 0.55 * pitch_correction

        # PD in joint space + gravity compensation -> torque motor command.
        bias = np.asarray(sim.data.qfrc_bias[6 : 6 + sim.model.nu], dtype=float)
        ctrl[:] = bias + kp * (q_des - q) - kd * qd
        # Damp out hip yaw a bit (it can excite spins).
        ctrl[0] *= 0.4
        ctrl[5] *= 0.4

        sim.data.ctrl[:] = clamp(ctrl, lo, hi)
        sim.step(1)

        # Lightweight progress prints every ~0.5s.
        if k % 250 == 0 or k == steps - 1:
            pelvis = sim.pelvis_position()
            print(
                f"step={k:4d} t={sim.data.time:6.2f} pelvis=({pelvis[0]:+.2f},{pelvis[1]:+.2f},{pelvis[2]:.2f}) "
                f"up={sim.torso_up():.2f} d_target={sim.distance_to_target():.2f} d_touch={sim.distance_to_touch():.2f}"
            )

    # Save best-so-far regardless of success; grader replays from canonical start using ctrl_trace.
    sim.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")
    print("TOUCH_POINT", TOUCH_POINT)


if __name__ == "__main__":
    run()
