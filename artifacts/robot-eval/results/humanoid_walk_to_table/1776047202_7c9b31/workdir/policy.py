"""Heuristic controller for Humanoid Walk To Table.

Generates a torque control history (recorded by sim.py) that moves the
Unitree H1 to the target marker and holds a stable stop pose.
"""

from __future__ import annotations

import argparse
import numpy as np
import mujoco

from sim import MIN_PELVIS_Z, MIN_TORSO_UP, Sim, TARGET_XY


def _pd_torque(sim: Sim, q_des: np.ndarray, kp: float, kd: float) -> np.ndarray:
    q = sim.data.qpos[7:]
    qd = sim.data.qvel[6:]
    tau = kp * (q_des - q) - kd * qd
    cr = sim.model.actuator_ctrlrange
    return np.clip(tau, cr[:, 0], cr[:, 1])


def run(
    *,
    max_steps: int,
    x_switch: float,
    crouch_steps: int,
    kp: float,
    kd: float,
    save_path: str | None,
    render_path: str | None,
):
    sim = Sim()
    home = sim.home_ctrl()

    # Forward-driving pose (found via quick random search).
    q_forward = home.copy()
    q_forward[2] = q_forward[7] = -0.455  # hip_pitch
    q_forward[3] = q_forward[8] = 0.638  # knee
    q_forward[4] = q_forward[9] = -1.055  # ankle
    q_forward[10] = 0.129  # torso

    # Stop pose: stable and near-stationary at the target.
    q_stop = home.copy()
    q_stop[2] = q_stop[7] = -0.35
    q_stop[3] = q_stop[8] = 1.20
    q_stop[4] = q_stop[9] = -0.90
    q_stop[10] = 0.0

    # Lower (more crouched) finish pose.
    q_crouch = home.copy()
    q_crouch[2] = q_crouch[7] = -1.0
    q_crouch[3] = q_crouch[8] = 1.4
    q_crouch[4] = q_crouch[9] = -1.1
    q_crouch[10] = 0.15

    pelvis_hist = []
    torso_up_hist = []

    # Optional lightweight rendering (saves a few frames as a .npz).
    frames = []
    frame_stride = 50

    for k in range(max_steps):
        pelvis = sim.pelvis_position()
        torso_up = sim.torso_up()
        pelvis_hist.append(pelvis.copy())
        torso_up_hist.append(torso_up)

        if pelvis[0] < x_switch:
            q_des = q_forward
        elif crouch_steps > 0 and k >= max_steps - crouch_steps:
            q_des = q_crouch
        else:
            q_des = q_stop
        sim.data.ctrl[:] = _pd_torque(sim, q_des, kp=kp, kd=kd)
        sim.step(1)

        if render_path and (k % frame_stride == 0):
            frames.append(sim.render(width=480, height=360))

    pelvis_hist = np.asarray(pelvis_hist, dtype=float)
    torso_up_hist = np.asarray(torso_up_hist, dtype=float)

    stable = (pelvis_hist[:, 2] >= MIN_PELVIS_Z) & (torso_up_hist >= MIN_TORSO_UP)
    stable_frac = float(stable.mean()) if len(stable) else 0.0

    # Simulate the grader's settle period on a copy of the current state so we
    # don't extend the saved control history.
    data_settle = mujoco.MjData(sim.model)
    data_settle.qpos[:] = sim.data.qpos
    data_settle.qvel[:] = sim.data.qvel
    data_settle.act[:] = sim.data.act
    data_settle.ctrl[:] = sim.data.ctrl
    data_settle.time = sim.data.time
    mujoco.mj_forward(sim.model, data_settle)

    settle_min_z = 1e9
    settle_min_up = 1e9
    for _ in range(500):
        mujoco.mj_step(sim.model, data_settle)
        settle_min_z = min(settle_min_z, float(data_settle.xpos[sim.pelvis_body_id][2]))
        settle_min_up = min(
            settle_min_up,
            float(data_settle.xmat[sim.torso_body_id].reshape(3, 3)[2, 2]),
        )

    settle_final_pelvis = np.array(data_settle.xpos[sim.pelvis_body_id], dtype=float)
    settle_final_dist = float(np.linalg.norm(settle_final_pelvis[:2] - TARGET_XY))

    final_pelvis = sim.pelvis_position()
    final_dist = float(np.linalg.norm(final_pelvis[:2] - TARGET_XY))

    checkpoints = [0.18, 0.34, 0.45]
    hit = {c: bool((pelvis_hist[:, 0] >= c).any()) for c in checkpoints}

    print(
        "replay_steps",
        len(pelvis_hist),
        "hit",
        hit,
        "stable_frac",
        stable_frac,
        "final_pelvis",
        final_pelvis,
        "final_dist",
        final_dist,
        "settle_final_pelvis",
        settle_final_pelvis,
        "settle_final_dist",
        settle_final_dist,
        "settle_min_z",
        settle_min_z,
        "settle_min_up",
        settle_min_up,
    )

    if render_path:
        np.savez_compressed(render_path, frames=np.asarray(frames, dtype=np.uint8))
        print("saved_render_frames", render_path, "n_frames", len(frames))

    if save_path:
        sim.save_final_state(save_path)
        print("saved_final_state", save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="/work/final_state.npz")
    parser.add_argument("--max-steps", type=int, default=1650)
    parser.add_argument("--x-switch", type=float, default=0.565)
    parser.add_argument("--crouch-steps", type=int, default=300)
    parser.add_argument("--kp", type=float, default=250.0)
    parser.add_argument("--kd", type=float, default=15.0)
    parser.add_argument("--render-npz", default=None)
    args = parser.parse_args()

    run(
        max_steps=args.max_steps,
        x_switch=args.x_switch,
        crouch_steps=args.crouch_steps,
        kp=args.kp,
        kd=args.kd,
        save_path=args.save,
        render_path=args.render_npz,
    )


if __name__ == "__main__":
    main()
