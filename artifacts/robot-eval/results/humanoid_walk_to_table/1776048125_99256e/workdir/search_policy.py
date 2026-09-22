import argparse
import math
import numpy as np

from sim import Sim, TARGET_XY

CHECKPOINTS = [0.18, 0.34, 0.45]


def evaluate(ctrls: np.ndarray):
    sim = Sim()
    xs, zs, ups = [], [], []
    for u in ctrls:
        sim.data.ctrl[:] = u
        sim.step(1)
        p = sim.pelvis_position()
        xs.append(p[0])
        zs.append(p[2])
        ups.append(sim.torso_up())
    settle_z, settle_u = [], []
    for _ in range(500):
        sim.step(1)
        settle_z.append(sim.pelvis_position()[2])
        settle_u.append(sim.torso_up())
    xs = np.array(xs)
    zs = np.array(zs)
    ups = np.array(ups)
    settle_z = np.array(settle_z)
    settle_u = np.array(settle_u)
    cp = sum(1 for c in CHECKPOINTS if np.max(xs) >= c)
    stable = float(((zs >= 0.23) & (ups >= 0.55)).mean()) if len(zs) else 0.0
    dist = float(np.linalg.norm(sim.pelvis_position()[:2] - TARGET_XY))
    minsz = float(np.min(settle_z))
    minsu = float(np.min(settle_u))
    passed = (
        len(ctrls) >= 30
        and cp == 3
        and stable >= 0.35
        and dist <= 0.20
        and minsz >= 0.23
        and minsu >= 0.55
    )
    checkpoint_progress = cp / 3
    stability_progress = min(stable / 0.35, 1.0)
    if dist <= 0.20:
        distance_progress = 1.0
    else:
        distance_progress = max(0.0, 1.0 - (dist - 0.20) / 0.20)
    settle_posture_progress = 0.5 * min(minsz / 0.23, 1.0) + 0.5 * min(minsu / 0.55, 1.0)
    efficiency_progress = min(1600 / len(ctrls), 1.0)
    score = (
        0.25 * checkpoint_progress
        + 0.25 * stability_progress
        + 0.30 * distance_progress
        + 0.15 * settle_posture_progress
        + 0.05 * efficiency_progress
    )
    return {
        "pass": passed,
        "score": score,
        "cp": cp,
        "stable": stable,
        "dist": dist,
        "min_settle_z": minsz,
        "min_settle_up": minsu,
        "xmax": float(np.max(xs)),
    }


def run(args):
    sim = Sim()
    home = sim.home_ctrl()
    ctrls = []

    # brief stabilization
    for _ in range(args.warmup):
        sim.data.ctrl[:] = home
        sim.step(1)
        ctrls.append(sim.data.ctrl.copy())

    dt = 0.002
    for i in range(args.walk_steps):
        t = i * dt
        ph = 2 * math.pi * args.freq * t
        s = math.sin(ph)
        s_opp = math.sin(ph + math.pi)
        c = home.copy()
        c[2] += args.hip_amp * s + args.hip_bias
        c[7] += args.hip_amp * s_opp + args.hip_bias
        c[3] += args.knee_amp * max(0.0, -s) + args.knee_bias
        c[8] += args.knee_amp * max(0.0, -s_opp) + args.knee_bias
        c[4] += args.ank_amp * s + args.ank_bias
        c[9] += args.ank_amp * s_opp + args.ank_bias
        c[1] += args.roll_amp * s
        c[6] -= args.roll_amp * s
        c[10] = args.torso_bias + args.torso_amp * math.sin(ph + args.torso_phase)
        c[11] += args.arm_amp * s_opp
        c[15] += args.arm_amp * s
        sim.data.ctrl[:] = c
        sim.step(1)
        ctrls.append(c.copy())

    for j in range(args.stop_steps):
        a = (j + 1) / max(1, args.stop_steps)
        c = home.copy()
        c[2] += args.stop_hip * a
        c[7] += args.stop_hip * a
        c[3] += args.stop_knee * a
        c[8] += args.stop_knee * a
        c[4] += args.stop_ank * a
        c[9] += args.stop_ank * a
        c[10] = args.stop_torso * a
        sim.data.ctrl[:] = c
        sim.step(1)
        ctrls.append(c.copy())

    for _ in range(args.hold_steps):
        c = home.copy()
        c[2] += args.stop_hip
        c[7] += args.stop_hip
        c[3] += args.stop_knee
        c[8] += args.stop_knee
        c[4] += args.stop_ank
        c[9] += args.stop_ank
        c[10] = args.stop_torso
        sim.data.ctrl[:] = c
        sim.step(1)
        ctrls.append(c.copy())

    ctrls = np.asarray(ctrls, dtype=float)
    metrics = evaluate(ctrls)
    print(metrics)
    if args.save:
        sim.save_final_state("/work/final_state.npz")
        print("saved /work/final_state.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--walk-steps", type=int, default=900)
    ap.add_argument("--stop-steps", type=int, default=250)
    ap.add_argument("--hold-steps", type=int, default=200)
    ap.add_argument("--freq", type=float, default=1.2)
    ap.add_argument("--hip-amp", type=float, default=0.28)
    ap.add_argument("--hip-bias", type=float, default=0.03)
    ap.add_argument("--knee-amp", type=float, default=0.45)
    ap.add_argument("--knee-bias", type=float, default=0.0)
    ap.add_argument("--ank-amp", type=float, default=0.10)
    ap.add_argument("--ank-bias", type=float, default=0.0)
    ap.add_argument("--roll-amp", type=float, default=0.05)
    ap.add_argument("--torso-bias", type=float, default=-0.10)
    ap.add_argument("--torso-amp", type=float, default=0.05)
    ap.add_argument("--torso-phase", type=float, default=math.pi / 2)
    ap.add_argument("--arm-amp", type=float, default=0.25)
    ap.add_argument("--stop-hip", type=float, default=0.10)
    ap.add_argument("--stop-knee", type=float, default=-0.06)
    ap.add_argument("--stop-ank", type=float, default=0.05)
    ap.add_argument("--stop-torso", type=float, default=0.10)
    ap.add_argument("--save", action="store_true")
    run(ap.parse_args())
