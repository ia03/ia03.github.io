from __future__ import annotations

import math
import numpy as np
import mujoco

from sim import Sim


FINGER_GEOM_IDS = (69, 77)


def solve_ik(sim, target, q_seed, wrist7=None, steps=120, alpha=0.18):
    q = q_seed.copy()
    model, data = sim.model, sim.data
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]
    for _ in range(steps):
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)
        pinch = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        err = target - pinch
        if np.linalg.norm(err) < 1e-4:
            break
        jp1 = np.zeros((3, model.nv))
        jr1 = np.zeros((3, model.nv))
        jp2 = np.zeros((3, model.nv))
        jr2 = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jp1, jr1, left_id)
        mujoco.mj_jacBody(model, data, jp2, jr2, right_id)
        J = 0.5 * (jp1[:, :7] + jp2[:, :7])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
        dq += 0.02 * (q_seed - q)
        q = np.clip(q + alpha * dq, qmin, qmax)
        if wrist7 is not None:
            q[6] = wrist7
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    return q


def open_drawer(sim):
    from goldens.open_drawer_then_pick.solve import open_drawer as open_drawer_impl

    open_drawer_impl(sim)


def has_contact(sim):
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        if 84 in (g1, g2) and (g1 in FINGER_GEOM_IDS or g2 in FINGER_GEOM_IDS):
            return True
    return False


def run_case(wrist7: float, grasp_z: float, lift_z: float):
    sim = Sim()
    open_drawer(sim)
    q = sim.data.qpos[:7].copy()
    cup = sim.block_position().copy()

    for target in [
        np.array([0.68, 0.04, 0.68]),
        np.array([cup[0], 0.04, 0.60]),
        np.array([cup[0], cup[1], grasp_z]),
    ]:
        q = solve_ik(sim, target, q, wrist7=wrist7, steps=140, alpha=0.16)
        for _ in range(50):
            sim.data.ctrl[:7] = q
            sim.data.ctrl[7] = 255.0
            sim.data.ctrl[8] = 1.0
            sim.step(1)

    for g in [255.0, 120.0, 40.0, 0.0]:
        for _ in range(60):
            sim.data.ctrl[:7] = q
            sim.data.ctrl[7] = g
            sim.data.ctrl[8] = 1.0
            sim.step(1)

    contact = has_contact(sim)
    block_after = sim.block_position().copy()

    for target in [
        np.array([block_after[0], 0.04, lift_z]),
        np.array([0.65, 0.04, lift_z]),
        np.array([0.55, 0.04, lift_z]),
        np.array([0.45, 0.04, lift_z]),
    ]:
        q = solve_ik(sim, target, q, wrist7=wrist7, steps=100, alpha=0.14)
        for _ in range(70):
            sim.data.ctrl[:7] = q
            sim.data.ctrl[7] = 0.0
            sim.data.ctrl[8] = 1.0
            sim.step(1)

    for _ in range(220):
        sim.data.ctrl[:7] = q
        sim.data.ctrl[7] = 0.0
        sim.data.ctrl[8] = 1.0
        sim.step(1)

    best_z = max(v["block_pos"][2] for v in sim._trace)
    final = sim.block_position().copy()
    passed_like = contact and best_z >= 0.5 and final[0] <= 0.42 and final[2] >= 0.5
    return passed_like, contact, best_z, final, sim


def main():
    best = None
    for wrist7 in [0.8, 1.0]:
        for grasp_z in [0.44, 0.45]:
            for lift_z in [0.60, 0.66, 0.72]:
                passed_like, contact, best_z, final, sim = run_case(wrist7, grasp_z, lift_z)
                row = (passed_like, contact, best_z, final.tolist(), wrist7, grasp_z, lift_z)
                print("trial", row, flush=True)
                if best is None or row > best[0]:
                    best = (row, sim)
                    sim.save_final_state("/work/final_state.npz")
                    print("saved best", row, flush=True)
                if passed_like:
                    return
    if best is not None:
        best[1].save_final_state("/work/final_state.npz")
        print("best", best[0], flush=True)


if __name__ == "__main__":
    main()
