import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim, SUCCESS_CUP_Z


ARM_Q0 = np.array([0.0, 0.4, 0.0, -1.8, 0.0, 2.2, 0.8], dtype=float)
OPEN_GRIP = 255.0
CLOSED_GRIP = 0.0


@dataclass
class Outcome:
    label: str
    cup_z: float
    contact_fraction: float
    progress: float
    passed: bool
    sim: Sim


def hand_ik(target, q_seed, iters=200, damping=1e-3):
    sim = Sim()
    m, d = sim.model, sim.data
    hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    q = q_seed.copy()
    for _ in range(iters):
        d.qpos[:7] = q
        d.qpos[7:9] = 0.04
        mujoco.mj_forward(m, d)
        err = target - d.xpos[hand_id]
        if np.linalg.norm(err) < 1e-4:
            break
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, None, hand_id)
        j = jacp[:, :7]
        dq = np.linalg.solve(j.T @ j + damping * np.eye(7), j.T @ err)
        q = np.clip(q + 0.8 * dq, m.jnt_range[:7, 0], m.jnt_range[:7, 1])
    return q


def finger_contact(sim):
    m, d = sim.model, sim.data
    cup_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")
    finger_bodies = {
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
    }
    for i in range(d.ncon):
        con = d.contact[i]
        g1, g2 = con.geom1, con.geom2
        b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
        if g1 == cup_geom and b2 in finger_bodies:
            return True
        if g2 == cup_geom and b1 in finger_bodies:
            return True
    return False


def progress(cup_z, contact_fraction):
    height_progress = np.clip((cup_z - 0.435) / (SUCCESS_CUP_Z - 0.435), 0.0, 1.0)
    contact_progress = np.clip(contact_fraction / 0.5, 0.0, 1.0)
    return 0.5 * height_progress + 0.5 * contact_progress


def hold_eval(sim, steps=500):
    contacts = 0
    for _ in range(steps):
        sim.step()
        contacts += int(finger_contact(sim))
    frac = contacts / steps
    cup_z = float(sim.cup_position()[2])
    return cup_z, frac


def drive(sim, q_target, grip, steps, blend=1.0):
    q_start = sim.data.ctrl[:7].copy()
    g_start = float(sim.data.ctrl[7])
    for i in range(steps):
        a = (i + 1) / steps
        b = 1.0 - math.exp(-blend * a)
        sim.data.ctrl[:7] = (1 - b) * q_start + b * q_target
        sim.data.ctrl[7] = (1 - b) * g_start + b * grip
        sim.step()


def run_sequence(descend_x, descend_z, grip_close, lift_z, settle_hold, label):
    sim = Sim()
    sim.data.ctrl[:7] = ARM_Q0
    sim.data.ctrl[7] = OPEN_GRIP
    sim.step(20)

    q_above = hand_ik(np.array([0.500, 0.0, 0.600]), ARM_Q0)
    q_desc = hand_ik(np.array([descend_x, 0.0, descend_z]), q_above)
    q_lift = hand_ik(np.array([descend_x, 0.0, lift_z]), q_desc)

    drive(sim, ARM_Q0, OPEN_GRIP, 100)
    drive(sim, q_above, OPEN_GRIP, 220)
    drive(sim, q_desc, OPEN_GRIP, 200)
    drive(sim, q_desc, grip_close, 160)
    drive(sim, q_lift, grip_close, 260)
    if settle_hold:
        drive(sim, q_lift, grip_close, settle_hold)

    cup_z, contact_fraction = hold_eval(sim)
    outcome = Outcome(
        label=label,
        cup_z=cup_z,
        contact_fraction=contact_fraction,
        progress=progress(cup_z, contact_fraction),
        passed=(cup_z >= SUCCESS_CUP_Z and contact_fraction >= 0.5),
        sim=sim,
    )
    return outcome


def maybe_save(outcome, best):
    if best is None or outcome.progress > best.progress:
        outcome.sim.save_final_state("/work/final_state.npz")
        print(
            f"saved {outcome.label} progress={outcome.progress:.3f} cup_z={outcome.cup_z:.3f} contact={outcome.contact_fraction:.3f}",
            flush=True,
        )
        return outcome
    return best


def main():
    best = None

    coarse = [
        (0.500, 0.495, 0.0, 0.650, 80, "coarse_a"),
        (0.498, 0.490, 0.0, 0.650, 80, "coarse_b"),
        (0.496, 0.488, 0.0, 0.640, 120, "coarse_c"),
        (0.494, 0.486, 0.0, 0.635, 160, "coarse_d"),
        (0.492, 0.484, 8.0, 0.630, 160, "coarse_e"),
    ]

    for params in coarse:
        outcome = run_sequence(*params)
        print(
            f"{outcome.label}: pass={outcome.passed} progress={outcome.progress:.3f} cup_z={outcome.cup_z:.3f} contact={outcome.contact_fraction:.3f}",
            flush=True,
        )
        best = maybe_save(outcome, best)

    if best and best.passed:
        return

    refine = []
    for dx in np.linspace(0.492, 0.500, 5):
        for dz in np.linspace(0.482, 0.492, 6):
            for grip in (0.0, 2.0, 4.0, 6.0):
                refine.append((float(dx), float(dz), float(grip), 0.635, 180, f"dx{dx:.3f}_dz{dz:.3f}_g{grip:.1f}"))

    for params in refine:
        outcome = run_sequence(*params)
        print(
            f"{outcome.label}: pass={outcome.passed} progress={outcome.progress:.3f} cup_z={outcome.cup_z:.3f} contact={outcome.contact_fraction:.3f}",
            flush=True,
        )
        best = maybe_save(outcome, best)
        if outcome.passed:
            break

    if best is None:
        raise RuntimeError("no attempt executed")


if __name__ == "__main__":
    main()
