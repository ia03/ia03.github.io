from __future__ import annotations

import math
import numpy as np
import mujoco

from sim import Sim


FINGER_GEOM_IDS = (69, 77)


def rotz(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def solve_ik(sim, target, q_seed, wrist7=None, steps=180, alpha=0.18):
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


def move(sim, target, grip, q_seed, wrist7=None, steps=180):
    q = solve_ik(sim, target, q_seed, wrist7=wrist7, steps=120, alpha=0.18)
    for _ in range(steps):
        sim.data.ctrl[:7] = q
        sim.data.ctrl[7] = grip
        sim.step(1)
    return q


def has_contact(sim):
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        if 84 in (g1, g2) and (g1 in FINGER_GEOM_IDS or g2 in FINGER_GEOM_IDS):
            return True
    return False


def run_case(yaw, wrist7, grasp_z, lift_z):
    sim = Sim()
    q = sim.data.qpos[:7].copy()
    sim.data.ctrl[7] = 255.0
    sim.step(30)
    cup = sim.cup_position().copy()

    r = rotz(yaw) @ np.diag([1.0, 1.0, -1.0])
    # Use the wrist rotation directly so the finger gap faces the wall-side approach.
    waypoints = [
        np.array([0.36, 0.24, 0.78]),
        np.array([0.50, 0.24, 0.78]),
        np.array([cup[0], 0.24, 0.78]),
        np.array([cup[0] + 0.01, cup[1], 0.60]),
        np.array([cup[0] + 0.01, cup[1], grasp_z]),
    ]
    for wp in waypoints:
        q = solve_ik(sim, wp, q, wrist7=wrist7, steps=120, alpha=0.18)
        for _ in range(50):
            sim.data.ctrl[:7] = q
            sim.data.ctrl[7] = 255.0
            sim.step(1)

    for g in [255.0, 140.0, 60.0, 0.0]:
        for _ in range(60):
            sim.data.ctrl[:7] = q
            sim.data.ctrl[7] = g
            sim.step(1)

    contact = has_contact(sim)
    cup_after = sim.cup_position().copy()
    for wp in [
        np.array([cup_after[0], 0.10, lift_z]),
        np.array([0.50, 0.10, lift_z]),
        np.array([0.42, 0.10, lift_z]),
        np.array([0.38, 0.10, lift_z]),
    ]:
        q = solve_ik(sim, wp, q, wrist7=wrist7, steps=100, alpha=0.16)
        for _ in range(80):
            sim.data.ctrl[:7] = q
            sim.data.ctrl[7] = 0.0
            sim.step(1)

    for _ in range(280):
        sim.data.ctrl[:7] = q
        sim.data.ctrl[7] = 0.0
        sim.step(1)

    best_x = min(v["cup_pos"][0] for v in sim._trace)
    best_z = max(v["cup_pos"][2] for v in sim._trace)
    final = sim.cup_position().copy()
    passed_like = contact and best_x <= 0.50 and best_z >= 0.56 and final[0] <= 0.42 and final[2] >= 0.56
    return passed_like, contact, best_x, best_z, final, sim


def main():
    best = None
    for yaw in [math.pi / 2, -math.pi / 2]:
        for wrist7 in [0.8, 1.0, 1.2]:
            for grasp_z in [0.44, 0.45, 0.46]:
                for lift_z in [0.58, 0.62, 0.66, 0.70]:
                    passed_like, contact, best_x, best_z, final, sim = run_case(yaw, wrist7, grasp_z, lift_z)
                    row = (passed_like, contact, best_x, best_z, final.tolist(), yaw, wrist7, grasp_z, lift_z)
                    print("trial", row, flush=True)
                    if best is None or (row[0], row[1], row[3]) > (best[0][0], best[0][1], best[0][3]):
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
