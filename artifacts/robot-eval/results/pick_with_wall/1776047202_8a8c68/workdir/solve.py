#!/usr/bin/env python3
import argparse
import os

import numpy as np

from sim import Sim


def rollout(sim: Sim, render_dir: str | None = None, max_steps: int = 2600, verbose: bool = False):
    # Waypoints found via kinematic solve for the finger-midpoint to reach the cup.
    q_pregrasp = np.array([0.79814238, 0.40982225, -0.37176690, -1.56631735, -2.10956263, 0.75318422, 1.14058045])
    q_grasp = np.array([0.77105995, 0.61942165, -0.32769227, -1.76547246, -1.93309847, 0.59640985, 1.14058045])
    q_lift = np.array([0.81429443, 0.42021826, -0.42176864, -1.50308152, -2.16341878, 0.76959872, 1.14058045])
    q_carry = np.array([0.96545515, -0.05857149, -0.52040654, -1.96393110, -1.70800146, 0.26930022, 1.14058045])

    def maybe_render(frame_idx: int):
        if not render_dir or frame_idx % 50 != 0:
            return
        os.makedirs(render_dir, exist_ok=True)
        img = sim.render(640, 480)
        try:
            from PIL import Image
        except Exception:
            return
        Image.fromarray(img).save(os.path.join(render_dir, f"frame_{frame_idx:05d}.png"))

    def set_arm(q_target: np.ndarray):
        sim.data.ctrl[:7] = q_target

    def open_gripper():
        sim.data.ctrl[7] = 255.0

    def close_gripper():
        sim.data.ctrl[7] = 0.0

    def arm_err(q_target: np.ndarray) -> float:
        return float(np.max(np.abs(sim.data.qpos[:7] - q_target)))

    def run_segment(q_target: np.ndarray, gripper: str, max_seg_steps: int, tol: float, min_steps: int = 1):
        if gripper == "open":
            open_gripper()
        elif gripper == "close":
            close_gripper()
        else:
            raise ValueError(gripper)
        set_arm(q_target)

        steps = 0
        while steps < max_seg_steps:
            maybe_render(len(sim._ctrl_trace))
            if gripper == "open":
                open_gripper()
            else:
                close_gripper()
            set_arm(q_target)
            sim.step(1)
            steps += 1
            if steps >= min_steps and arm_err(q_target) < tol:
                break
        return steps

    total = 0

    # Ensure ctrl is initialized.
    open_gripper()
    set_arm(sim.data.qpos[:7].copy())
    sim.step(5)
    total += 5

    total += run_segment(q_pregrasp, "open", max_seg_steps=700, tol=0.03, min_steps=200)
    total += run_segment(q_grasp, "open", max_seg_steps=700, tol=0.025, min_steps=250)

    # Close and wait for contact / settle into grasp.
    close_gripper()
    set_arm(q_grasp)
    contact_seen = False
    for _ in range(300):
        maybe_render(len(sim._ctrl_trace))
        close_gripper()
        set_arm(q_grasp)
        sim.step(1)
        total += 1
        contact_seen = contact_seen or sim.has_gripper_cup_contact()
        if contact_seen and total > 50:
            break

    total += run_segment(q_lift, "close", max_seg_steps=600, tol=0.03, min_steps=200)
    total += run_segment(q_carry, "close", max_seg_steps=900, tol=0.035, min_steps=300)

    # Hold for a while so replay ends with stable, sustained contact.
    hold_steps = min(700, max(50, max_steps - total))
    for _ in range(hold_steps):
        maybe_render(len(sim._ctrl_trace))
        close_gripper()
        set_arm(q_carry)
        sim.step(1)
        total += 1

    if verbose:
        cup = sim.cup_position()
        print("final cup", cup, "contact", sim.has_gripper_cup_contact(), "steps", len(sim._ctrl_trace))

    return sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", default=None)
    parser.add_argument("--steps", type=int, default=2600)
    parser.add_argument("--out", default="/work/final_state.npz")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sim = Sim()
    rollout(sim, render_dir=args.render_dir, max_steps=args.steps, verbose=args.verbose)
    sim.save_final_state(args.out)
    print("saved:", os.path.abspath(args.out), "steps:", len(sim._ctrl_trace))


if __name__ == "__main__":
    main()
