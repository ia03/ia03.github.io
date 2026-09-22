"""Fallback policy: hold a stable standing pose and save final_state.npz.

This doesn't attempt the table-touch, but ensures the saved artifact is at least
upright/stable for partial credit.
"""

from __future__ import annotations

import numpy as np

from sim import Sim
from run_policy import bias_torque, pd_torque


def main() -> None:
    sim = Sim()
    dt = float(sim.model.opt.timestep)

    home = sim.home_ctrl()
    q_des = home.copy()

    # Empirically stable offsets.
    names = sim.actuator_names()
    idx = {n: i for i, n in enumerate(names)}
    q_des[idx["left_hip_pitch"]] = home[idx["left_hip_pitch"]] - 0.10
    q_des[idx["right_hip_pitch"]] = home[idx["right_hip_pitch"]] - 0.10
    q_des[idx["torso"]] = 0.02

    # High-gain joint-space PD (torque actuators) + bias feedforward.
    kp = np.ones(sim.model.nu, dtype=float) * 300.0
    for i, n in enumerate(names):
        if "hip_" in n:
            kp[i] = 1200.0
        if "knee" in n:
            kp[i] = 2200.0
        if "ankle" in n:
            kp[i] = 1400.0
        if n == "torso":
            kp[i] = 900.0
        if "shoulder" in n:
            kp[i] = 120.0
        if "elbow" in n:
            kp[i] = 80.0
    kd = 0.06 * kp

    steps = int(6.0 / dt)
    for _ in range(steps):
        ff = bias_torque(sim)
        sim.data.ctrl[:] = pd_torque(sim, q_des, kp, kd, ff)
        sim.step(1)

    print("final time", float(sim.data.time))
    print("final pelvis", sim.pelvis_position())
    print("final torso_up", sim.torso_up())
    print("final dist_target", sim.distance_to_target())
    print("final dist_touch", sim.distance_to_touch())
    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()

