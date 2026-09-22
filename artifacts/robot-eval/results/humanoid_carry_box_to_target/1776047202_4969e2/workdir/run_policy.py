#!/usr/bin/env python3
import numpy as np

import sim


def main():
    s = sim.Sim()
    dt = float(s.model.opt.timestep)

    q0 = s._initial_qpos[7 : 7 + s.model.nu].copy()

    # Empirically stable symmetric stance.
    q_stance = q0.copy()
    q_stance[2] = q_stance[7] = -0.4  # hip pitch
    q_stance[3] = q_stance[8] = 0.6  # knee
    q_stance[4] = q_stance[9] = -0.2  # ankle
    q_stance[10] = -0.2  # torso

    kp = 1500.0
    kd = 80.0

    def apply_pd(q_des):
        q = s.data.qpos[7 : 7 + s.model.nu]
        qd = s.data.qvel[6 : 6 + s.model.nu]
        tau = kp * (q_des - q) - kd * qd
        s.data.ctrl[:] = tau

    # Settle into stance with feedback (torques get recorded in ctrl_trace).
    settle_steps = 2200  # 4.4s
    for _ in range(settle_steps):
        apply_pd(q_stance)
        s.step()

    # Freeze the last torque vector to make the final control open-loop-stable.
    tau_const = s.data.ctrl.copy()
    freeze_steps = 800  # 1.6s
    for _ in range(freeze_steps):
        s.data.ctrl[:] = tau_const
        s.step()

    print("time", s.data.time, "pelvis", s.pelvis_position(), "box", s.box_position())
    print(
        "dist_to_target",
        s.distance_to_target(),
        "box_dist_to_target",
        s.box_distance_to_target(),
        "box_to_pelvis",
        s.box_distance_to_pelvis(),
        "torso_up",
        s.torso_up(),
    )
    s.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()

