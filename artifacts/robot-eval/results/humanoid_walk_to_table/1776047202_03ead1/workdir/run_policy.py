"""Open-loop torque replay policy for the Humanoid Walk To Table task.

This script runs a deterministic PD-torque trajectory in the provided MuJoCo
sim and saves the resulting replay payload to /work/final_state.npz.
"""

from __future__ import annotations

import argparse
import numpy as np

from sim import Sim, MIN_PELVIS_Z, MIN_TORSO_UP, TARGET_XY


def run(steps: int, kp: float, kd: float, seed: int) -> Sim:
    rng = np.random.default_rng(seed)
    _ = rng  # keep deterministic hook in case we add noise later

    sim = Sim()

    # Hand-tuned stable "forward drift" posture found via random search.
    # Params: (droll, dhip, dknee, dank, dtor, kp, kd)
    droll = -0.09
    dhip = 0.639
    dknee = -0.286
    dank = -0.579
    dtor = 0.563

    base = sim._initial_qpos[7 : 7 + sim.model.nu].copy()
    q_des = base.copy()

    # Actuator indices (match sim.actuator_names()).
    left_hip_roll = 1
    left_hip_pitch = 2
    left_knee = 3
    left_ankle = 4
    right_hip_roll = 6
    right_hip_pitch = 7
    right_knee = 8
    right_ankle = 9
    torso = 10

    q_des[left_hip_roll] += droll
    q_des[right_hip_roll] -= droll
    for idx in (left_hip_pitch, right_hip_pitch):
        q_des[idx] += dhip
    for idx in (left_knee, right_knee):
        q_des[idx] += dknee
    for idx in (left_ankle, right_ankle):
        q_des[idx] += dank
    q_des[torso] += dtor

    stable_steps = 0
    checkpoints = (0.18, 0.34, 0.45)
    hit = [False, False, False]

    for _ in range(steps):
        q = sim.data.qpos[7 : 7 + sim.model.nu]
        qd = sim.data.qvel[6 : 6 + sim.model.nu]
        u = kp * (q_des - q) - kd * qd
        sim.data.ctrl[:] = np.clip(u, -200.0, 200.0)
        sim.step(1)

        pelvis = sim.pelvis_position()
        for i, cpt in enumerate(checkpoints):
            if pelvis[0] >= cpt:
                hit[i] = True
        if pelvis[2] >= MIN_PELVIS_Z and sim.torso_up() >= MIN_TORSO_UP:
            stable_steps += 1

    pelvis = sim.pelvis_position()
    dist = float(np.linalg.norm(pelvis[:2] - TARGET_XY))
    stable_frac = stable_steps / max(1, steps)
    print(
        "done:",
        f"steps={steps}",
        f"pelvis={pelvis}",
        f"dist={dist:.3f}",
        f"torso_up={sim.torso_up():.3f}",
        f"stable_frac={stable_frac:.3f}",
        f"hit={hit}",
    )
    return sim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--kp", type=float, default=106.109)
    ap.add_argument("--kd", type=float, default=12.228)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="/work/final_state.npz")
    args = ap.parse_args()

    sim = run(args.steps, args.kp, args.kd, args.seed)
    sim.save_final_state(args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()

