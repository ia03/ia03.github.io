from __future__ import annotations

import math
import numpy as np
import mujoco

from sim import Sim


FINGER_GEOM_IDS = (69, 77)


def orientation_error(current, target):
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )


def rotz(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def solve_ik(sim, target, q_seed, wrist7=None, steps=180, alpha=0.16):
    q = q_seed.copy()
    model, data = sim.model, sim.data
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]
    for _ in range(steps):
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)
        p = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        r = data.xmat[hand_id].reshape(3, 3).copy()
        err = target - p
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


def move(sim, target, grip, q_seed, wrist7=None, steps=220, alpha=0.15):
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


def main():
    sims = []
    candidates = []
    yaws = [math.pi / 2, -math.pi / 2, 2.09, -2.09]
    wrist7s = [0.6, 0.8, 1.0, 1.2]
    grasp_offsets = [
        (0.0, 0.0, 0.44),
        (0.0, 0.0, 0.45),
        (0.0, 0.0, 0.46),
        (0.01, 0.0, 0.45),
        (0.01, 0.0, 0.46),
        (0.0, 0.01, 0.45),
        (0.0, -0.01, 0.45),
    ]
    lifts = [0.58, 0.62, 0.66, 0.70]
    for yaw in yaws:
        for wrist7 in wrist7s:
            for dx, dy, gz in grasp_offsets:
                for lift_z in lifts:
                    sim = Sim()
                    q = sim.data.qpos[:7].copy()
                    sim.data.ctrl[7] = 255.0
                    sim.step(40)
                    r = rotz(yaw) @ np.diag([1.0, 1.0, -1.0])
                    cup = sim.cup_position().copy()
                    waypoints = [
                        np.array([0.36, 0.24, 0.78]),
                        np.array([0.50, 0.24, 0.78]),
                        np.array([cup[0] + 0.02, 0.24, 0.78]),
                        np.array([cup[0] + 0.01, cup[1], 0.60]),
                        np.array([cup[0] + dx, cup[1] + dy, gz]),
                    ]
                    for wp in waypoints:
                        q = solve_ik(sim, wp, q, wrist7=wrist7, steps=120, alpha=0.16)
                        for _ in range(60):
                            sim.data.ctrl[:7] = q
                            sim.data.ctrl[7] = 255.0
                            sim.step(1)
                    q = solve_ik(sim, waypoints[-1], q, wrist7=wrist7, steps=80, alpha=0.12)
                    for g in [255.0, 180.0, 120.0, 80.0, 40.0, 0.0]:
                        for _ in range(40):
                            sim.data.ctrl[:7] = q
                            sim.data.ctrl[7] = g
                            sim.step(1)
                    contact = has_contact(sim)
                    cup_after = sim.cup_position().copy()
                    if contact or cup_after[2] > 0.5:
                        print("candidate", yaw, wrist7, (dx, dy, gz), lift_z, "contact", contact, "cup", cup_after, "trace", len(sim._ctrl_trace))
                    for target in [
                        np.array([cup_after[0], 0.10, lift_z]),
                        np.array([0.50, 0.10, lift_z]),
                        np.array([0.42, 0.10, lift_z]),
                        np.array([0.38, 0.10, lift_z]),
                    ]:
                        q = solve_ik(sim, target, q, wrist7=wrist7, steps=100, alpha=0.14)
                        for _ in range(80):
                            sim.data.ctrl[:7] = q
                            sim.data.ctrl[7] = 0.0
                            sim.step(1)
                    for _ in range(260):
                        sim.data.ctrl[:7] = q
                        sim.data.ctrl[7] = 0.0
                        sim.step(1)
                    best_z = max(v["cup_pos"][2] for v in sim._trace)
                    final = sim.cup_position().copy()
                    passed_like = contact and best_z >= 0.56 and final[0] <= 0.42 and final[2] >= 0.56
                    candidates.append((passed_like, contact, best_z, final.copy(), yaw, wrist7, (dx, dy, gz), lift_z, sim))
                    print("trial", len(candidates), "contact", contact, "best_z", round(best_z, 3), "final", np.round(final, 3), "passed", passed_like, flush=True)
                    if passed_like:
                        sim.save_final_state("/work/final_state.npz")
                        print("saved final_state.npz")
                        return
    best = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    best[-1].save_final_state("/work/final_state.npz")
    print("best", best[:8], flush=True)


if __name__ == "__main__":
    main()
