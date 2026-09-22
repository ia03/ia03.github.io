import numpy as np

from sim import MIN_PELVIS_Z, MIN_TORSO_UP, Sim, TARGET_XY


def pd_torque(model, q_des, q, qd):
    # Tuned by joint group (actuator ordering == joint ordering after free root).
    kp = np.array(
        [
            120.0,  # left_hip_yaw
            160.0,  # left_hip_roll
            220.0,  # left_hip_pitch
            320.0,  # left_knee
            90.0,  # left_ankle
            120.0,  # right_hip_yaw
            160.0,  # right_hip_roll
            220.0,  # right_hip_pitch
            320.0,  # right_knee
            90.0,  # right_ankle
            220.0,  # torso
            60.0,  # left_shoulder_pitch
            40.0,  # left_shoulder_roll
            30.0,  # left_shoulder_yaw
            25.0,  # left_elbow
            60.0,  # right_shoulder_pitch
            40.0,  # right_shoulder_roll
            30.0,  # right_shoulder_yaw
            25.0,  # right_elbow
        ],
        dtype=float,
    )
    kd = 2.0 * np.sqrt(kp) * 0.55
    tau = kp * (q_des - q) - kd * qd
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    return np.clip(tau, lo, hi)


def torso_roll_pitch(sim):
    R = sim.data.xmat[sim.torso_body_id].reshape(3, 3)
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    pitch = float(np.arctan2(-R[2, 0], sy))
    return roll, pitch


def add_balance_terms(sim, tau, roll_target=0.0, pitch_target=0.06):
    roll, pitch = torso_roll_pitch(sim)
    w = sim.data.cvel[sim.torso_body_id][:3].copy()  # world-frame angular velocity
    roll_rate = float(w[0])
    pitch_rate = float(w[1])

    # Pitch balance primarily through hip pitch (sign tested: negative hip torque -> pitch positive).
    pitch_err = pitch - pitch_target
    kp_pitch = 900.0
    kd_pitch = 120.0
    corr_pitch = kp_pitch * pitch_err + kd_pitch * pitch_rate
    tau[2] += corr_pitch
    tau[7] += corr_pitch

    # Roll balance through hip roll (sign tested: tau positive -> roll negative).
    roll_err = roll - roll_target
    kp_roll = 700.0
    kd_roll = 90.0
    corr_roll = kp_roll * roll_err + kd_roll * roll_rate
    tau[1] += corr_roll
    tau[6] += corr_roll

    return tau


def make_qdes(home_q, phase, step_amp=0.22, lift_amp=0.08, knee_lift=0.30):
    """
    Open-loop desired joint angles around the home crouch.
    A PD controller converts these into actuator torques.
    """
    qdes = home_q.copy()

    # Indices by actuator order (see sim.py main).
    (
        l_hy,
        l_hr,
        l_hp,
        l_k,
        l_a,
        r_hy,
        r_hr,
        r_hp,
        r_k,
        r_a,
        torso,
        l_sp,
        l_sr,
        l_sy,
        l_e,
        r_sp,
        r_sr,
        r_sy,
        r_e,
    ) = range(qdes.shape[0])

    s = np.sin(phase)
    c = np.cos(phase)

    # Keep torso actuator near zero; balance controller handles global pitch/roll.
    qdes[torso] = 0.0

    # Weight shift: roll hips opposite directions.
    qdes[l_hr] += -lift_amp * s
    qdes[r_hr] += +lift_amp * s

    # Swing/stance hip pitch: asymmetric forward/back motion.
    qdes[l_hp] += +step_amp * s
    qdes[r_hp] += -step_amp * s

    # Knee lift during swing (both, but phase-dependent lift).
    qdes[l_k] += knee_lift * max(0.0, s)
    qdes[r_k] += knee_lift * max(0.0, -s)
    qdes[l_a] += -0.55 * knee_lift * max(0.0, s)
    qdes[r_a] += -0.55 * knee_lift * max(0.0, -s)

    # Keep feet under body via small ankle compensation.
    qdes[l_a] += -0.06 * s
    qdes[r_a] += +0.06 * s

    # Tiny yaw stabilization.
    qdes[l_hy] += 0.05 * c
    qdes[r_hy] += -0.05 * c

    # Arm swing for balance.
    qdes[l_sp] += -0.55 * s
    qdes[r_sp] += +0.55 * s
    qdes[l_e] += +0.25 * max(0.0, s)
    qdes[r_e] += +0.25 * max(0.0, -s)

    return qdes


def run_episode(
    replay_steps=2400,
    warmup_steps=250,
    gait_period_steps=460,
    step_amp=0.22,
    lift_amp=0.08,
    stop_steps=450,
    seed=0,
):
    rng = np.random.default_rng(seed)
    sim = Sim()
    home_q = sim._initial_qpos[7 : 7 + sim.model.nu].copy()

    # Small randomization to avoid perfectly symmetric deadlocks.
    home_q = home_q + rng.normal(scale=0.001, size=home_q.shape)

    # Warmup: PD hold home pose.
    for _ in range(warmup_steps):
        q = sim.data.qpos[7 : 7 + sim.model.nu].copy()
        qd = sim.data.qvel[6 : 6 + sim.model.nu].copy()
        tau = pd_torque(sim.model, home_q, q, qd)
        tau = add_balance_terms(sim, tau, pitch_target=0.05)
        sim.data.ctrl[:] = tau
        sim.step(1)

    # Gait.
    w = 2 * np.pi / float(gait_period_steps)
    for t in range(replay_steps):
        phase = w * t
        q = sim.data.qpos[7 : 7 + sim.model.nu].copy()
        qd = sim.data.qvel[6 : 6 + sim.model.nu].copy()
        qdes = make_qdes(home_q, phase, step_amp=step_amp, lift_amp=lift_amp)
        tau = pd_torque(sim.model, qdes, q, qd)
        tau = add_balance_terms(sim, tau, pitch_target=0.08)
        sim.data.ctrl[:] = tau
        sim.step(1)

    # Controlled low stop near the end: keep lean, reduce stepping.
    for t in range(stop_steps):
        phase = w * t
        q = sim.data.qpos[7 : 7 + sim.model.nu].copy()
        qd = sim.data.qvel[6 : 6 + sim.model.nu].copy()
        qdes = make_qdes(home_q, phase, step_amp=0.02, lift_amp=0.05, knee_lift=0.08)
        # Slightly deeper crouch for settle stability.
        qdes[3] += 0.15  # left_knee
        qdes[8] += 0.15  # right_knee
        qdes[4] += -0.07  # left_ankle
        qdes[9] += -0.07  # right_ankle
        tau = pd_torque(sim.model, qdes, q, qd)
        tau = add_balance_terms(sim, tau, pitch_target=0.10)
        sim.data.ctrl[:] = tau
        sim.step(1)

    pelvis = sim.pelvis_position()
    torso_up = sim.torso_up()
    dist = sim.distance_to_target()

    trace_pelvis = np.array([e["pelvis_pos"] for e in sim._trace], dtype=float)
    trace_torso = np.array([e["torso_up"] for e in sim._trace], dtype=float)
    min_pz = float(np.min(trace_pelvis[:, 2]))
    min_tu = float(np.min(trace_torso))

    print(
        "final pelvis",
        np.round(pelvis, 3),
        "torso_up",
        round(torso_up, 3),
        "dist",
        round(dist, 3),
        "min_pz",
        round(min_pz, 3),
        "min_tu",
        round(min_tu, 3),
    )
    print("target_xy", TARGET_XY, "thresholds", {"pelvis_z": MIN_PELVIS_Z, "torso_up": MIN_TORSO_UP})
    return sim


if __name__ == "__main__":
    sim = run_episode()
    sim.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")
