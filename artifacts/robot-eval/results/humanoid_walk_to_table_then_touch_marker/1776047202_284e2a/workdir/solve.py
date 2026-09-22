#!/usr/bin/env python3
import math
import numpy as np

from sim import Sim, TOUCH_POINT


def make_index(sim: Sim):
    names = sim.actuator_names()
    idx = {n: i for i, n in enumerate(names)}
    m = sim.model
    qposadr = np.empty(m.nu, dtype=int)
    dofadr = np.empty(m.nu, dtype=int)
    for i in range(m.nu):
        joint_id = int(m.actuator_trnid[i, 0])
        qposadr[i] = int(m.jnt_qposadr[joint_id])
        dofadr[i] = int(m.jnt_dofadr[joint_id])
    return names, idx, qposadr, dofadr


def run(seed: int = 0, steps: int = 3200, save_path: str = "/work/final_state.npz"):
    rng = np.random.default_rng(seed)
    sim = Sim()
    m = sim.model
    dt = float(m.opt.timestep)

    names, idx, qposadr, dofadr = make_index(sim)

    q_nom = sim.data.qpos[7 : 7 + m.nu].copy()

    kp = np.ones(m.nu) * 20.0
    kd = np.ones(m.nu) * 1.5
    for i, n in enumerate(names):
        if "hip" in n or "knee" in n:
            kp[i] = 170.0
            kd[i] = 10.0
        if "ankle" in n:
            kp[i] = 90.0
            kd[i] = 6.0
        if "shoulder" in n or "elbow" in n:
            kp[i] = 40.0
            kd[i] = 2.5

    prev_pitch = 0.0
    prev_roll = 0.0

    def torso_angles():
        R = sim.data.xmat[sim.torso_body_id].reshape(3, 3)
        up = R[:, 2]
        pitch = float(math.atan2(float(up[0]), float(up[2])))
        roll = float(math.atan2(float(up[1]), float(up[2])))
        return pitch, roll

    def control(step: int):
        nonlocal prev_pitch, prev_roll

        pelvis = sim.pelvis_position()
        pelvis_x = float(pelvis[0])
        pelvis_y = float(pelvis[1])

        pitch, roll = torso_angles()
        pitch_rate = (pitch - prev_pitch) / dt
        roll_rate = (roll - prev_roll) / dt
        prev_pitch, prev_roll = pitch, roll

        # Phase schedule: stabilize -> walk -> reach/hold.
        if step < 300:
            walk = 0.0
            pitch_ref = 0.0
        elif pelvis_x < 0.90:
            walk = 1.0
            pitch_ref = 0.14
        else:
            walk = 0.0
            pitch_ref = 0.10

        q_des = q_nom.copy()

        # Gentle crouch so elbows can reach the marker height.
        crouch = 0.20 if pelvis_x > 0.70 else 0.10
        for side in ("left", "right"):
            q_des[idx[f"{side}_hip_pitch"]] = q_nom[idx[f"{side}_hip_pitch"]] - 0.10 * crouch / 0.10
            q_des[idx[f"{side}_knee"]] = q_nom[idx[f"{side}_knee"]] + 0.25 * crouch / 0.10
            q_des[idx[f"{side}_ankle"]] = q_nom[idx[f"{side}_ankle"]] - 0.12 * crouch / 0.10

        if walk > 0.5:
            # Open-loop gait: alternating hip/knee with a lateral weight shift.
            omega = 2.0 * math.pi * 0.85
            phase = omega * (step * dt - 0.6)
            s = math.sin(phase)
            c = math.cos(phase)

            hip_amp = 0.32
            knee_amp = 0.40
            ankle_amp = 0.18
            roll_amp = 0.10

            q_des[idx["left_hip_pitch"]] += hip_amp * s
            q_des[idx["right_hip_pitch"]] -= hip_amp * s
            q_des[idx["left_knee"]] += knee_amp * max(0.0, c)
            q_des[idx["right_knee"]] += knee_amp * max(0.0, -c)
            q_des[idx["left_ankle"]] -= ankle_amp * s
            q_des[idx["right_ankle"]] += ankle_amp * s

            # Shift weight side-to-side and lightly regulate pelvis y.
            y_reg = float(np.clip(-1.5 * pelvis_y, -0.12, 0.12))
            q_des[idx["left_hip_roll"]] = q_nom[idx["left_hip_roll"]] + roll_amp * s + y_reg
            q_des[idx["right_hip_roll"]] = q_nom[idx["right_hip_roll"]] - roll_amp * s - y_reg
        else:
            # When not walking, try to center lateral drift.
            y_reg = float(np.clip(-2.0 * pelvis_y, -0.18, 0.18))
            q_des[idx["left_hip_roll"]] = q_nom[idx["left_hip_roll"]] + y_reg
            q_des[idx["right_hip_roll"]] = q_nom[idx["right_hip_roll"]] - y_reg

        # Reaching configuration once close enough to the table.
        if pelvis_x > 0.85:
            q_des[idx["right_shoulder_pitch"]] = -0.45
            q_des[idx["right_shoulder_roll"]] = 0.55
            q_des[idx["right_shoulder_yaw"]] = 0.05
            q_des[idx["right_elbow"]] = 0.85

        q = sim.data.qpos[qposadr]
        qd = sim.data.qvel[dofadr]
        tau = kp * (q_des - q) - kd * qd

        # Balance torques (ankle + hip_pitch) based on torso tilt.
        pitch_err = pitch - pitch_ref
        bal_ankle = 230.0 * pitch_err + 26.0 * pitch_rate
        bal_hip = 120.0 * pitch_err + 12.0 * pitch_rate
        tau[idx["left_ankle"]] += bal_ankle
        tau[idx["right_ankle"]] += bal_ankle
        tau[idx["left_hip_pitch"]] += bal_hip
        tau[idx["right_hip_pitch"]] += bal_hip

        # Roll stabilization via hip_roll differential torques.
        roll_err = roll
        bal_roll = 110.0 * roll_err + 10.0 * roll_rate
        tau[idx["left_hip_roll"]] += -bal_roll
        tau[idx["right_hip_roll"]] += +bal_roll

        # Small random dither to break symmetry early.
        if step < 600:
            tau += rng.normal(scale=0.5, size=tau.shape)

        sim.data.ctrl[:] = np.clip(tau, -200.0, 200.0)

    for step in range(int(steps)):
        control(step)
        sim.step(1)

    sim.save_final_state(save_path)

    pelvis = sim.pelvis_position()
    touch = sim.distance_to_touch()
    print(
        f"saved t={sim.data.time:.3f}s pelvis=({pelvis[0]:.3f},{pelvis[1]:.3f},{pelvis[2]:.3f}) "
        f"torso_up={sim.torso_up():.3f} target_d={sim.distance_to_target():.3f} touch_d={touch:.3f} "
        f"touch_point={tuple(float(x) for x in TOUCH_POINT)}"
    )
    return sim


if __name__ == "__main__":
    run()

