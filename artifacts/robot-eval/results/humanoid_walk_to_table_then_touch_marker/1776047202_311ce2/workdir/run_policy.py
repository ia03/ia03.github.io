import numpy as np

from sim import Sim


def torso_pitch_roll(sim: Sim):
    # xmat is row-major 3x3 rotation from body->world.
    # Body z-axis in world coordinates is the 3rd column.
    xmat = sim.data.xmat[sim.torso_body_id].reshape(3, 3)
    up = xmat[:, 2]
    pitch = float(np.arctan2(up[0], up[2]))
    roll = float(-np.arctan2(up[1], up[2]))
    return pitch, roll


def run(total_steps: int = 5000, save_path: str = "/work/final_state.npz"):
    sim = Sim()
    names = sim.actuator_names()
    idx = {name: i for i, name in enumerate(names)}

    def set_tau(name, value):
        sim.data.ctrl[idx[name]] = float(value)

    # Basic balance targets.
    pitch_des = 0.10  # lean slightly forward
    z_des = 0.95

    # Gains (hand-tuned quickly).
    kp_pitch = 120.0
    kd_pitch = 8.0
    kp_roll = 80.0
    kd_roll = 6.0
    kp_z = 600.0
    kd_z = 80.0

    # Gait / reach parameters.
    step_period = 0.70
    hip_swing = 55.0
    knee_swing = 60.0
    hip_roll_shift = 25.0

    reach_start_x = 0.85
    reach_torque = 35.0

    prev_pitch = 0.0
    prev_roll = 0.0

    dt = float(sim.model.opt.timestep)

    # Give MuJoCo a moment to settle into contact.
    sim.data.ctrl[:] = 0.0
    sim.step(50)

    # Save an early, minimal (but replayable) attempt as soon as we have >=30 steps.
    # This is a hard requirement for scoring.
    for t in range(total_steps):
        sim.data.ctrl[:] = 0.0

        pelvis = sim.pelvis_position()
        pitch, roll = torso_pitch_roll(sim)

        pitch_rate = (pitch - prev_pitch) / dt
        roll_rate = (roll - prev_roll) / dt
        prev_pitch, prev_roll = pitch, roll

        # Vertical support using knee extension + ankle pitch.
        z_err = z_des - float(pelvis[2])
        z_vel = float(sim.data.qvel[2])  # freejoint linear z velocity
        support = kp_z * z_err - kd_z * z_vel
        # Split support across both legs.
        set_tau("left_knee", -0.5 * support)
        set_tau("right_knee", -0.5 * support)
        set_tau("left_ankle", 0.18 * support)
        set_tau("right_ankle", 0.18 * support)

        # Pitch balance (ankle strategy) + slight forward lean.
        pitch_err = pitch_des - pitch
        pitch_u = kp_pitch * pitch_err - kd_pitch * pitch_rate
        set_tau("left_ankle", sim.data.ctrl[idx["left_ankle"]] + pitch_u)
        set_tau("right_ankle", sim.data.ctrl[idx["right_ankle"]] + pitch_u)
        set_tau("torso", -0.25 * pitch_u)

        # Roll balance via hip roll.
        roll_u = kp_roll * (0.0 - roll) - kd_roll * roll_rate
        set_tau("left_hip_roll", -roll_u)
        set_tau("right_hip_roll", roll_u)

        # Very simple alternating hip/knee swing to generate forward steps.
        phase = 2.0 * np.pi * (float(sim.data.time) / step_period)
        s = float(np.sin(phase))
        c = float(np.cos(phase))
        set_tau("left_hip_pitch", hip_swing * s - 0.15 * support)
        set_tau("right_hip_pitch", -hip_swing * s - 0.15 * support)
        set_tau("left_knee", sim.data.ctrl[idx["left_knee"]] + knee_swing * c)
        set_tau("right_knee", sim.data.ctrl[idx["right_knee"]] - knee_swing * c)
        set_tau("left_hip_yaw", 4.0 * s)
        set_tau("right_hip_yaw", -4.0 * s)
        set_tau("left_hip_roll", sim.data.ctrl[idx["left_hip_roll"]] + hip_roll_shift * s)
        set_tau("right_hip_roll", sim.data.ctrl[idx["right_hip_roll"]] - hip_roll_shift * s)

        # Arm reach once we're near the table.
        if float(pelvis[0]) > reach_start_x:
            set_tau("right_shoulder_pitch", -reach_torque)
            set_tau("right_elbow", -0.6 * reach_torque)
            set_tau("right_shoulder_roll", -0.15 * reach_torque)

        sim.step(1)

        if t == 80:
            sim.save_final_state(save_path)

    sim.save_final_state(save_path)
    return sim


if __name__ == "__main__":
    sim = run()
    print("final pelvis:", sim.pelvis_position())
    print("final torso_up:", sim.torso_up())
    print("final target dist:", sim.distance_to_target())
    best_touch = float(np.min([entry["touch_distance"] for entry in sim._trace]))
    print("best touch dist:", best_touch)
