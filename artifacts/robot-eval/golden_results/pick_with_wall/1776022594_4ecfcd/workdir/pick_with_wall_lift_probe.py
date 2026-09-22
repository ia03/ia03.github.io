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
    import mujoco

    HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
    SAFE = np.array([0.0, -0.35, 0.0, -1.90, 0.0, 1.95, 0.785], dtype=float)
    Q_HANDLE = np.array(
        [0.260114734942896, 0.5978935416914477, -0.2548739609310991, -1.7011568239308756, 0.0731417910040021, 3.7525, 0.8723261334629767],
        dtype=float,
    )
    Q_PULL = np.array(
        [0.12598874856351472, 1.1973462499176968, -0.34850701301211456, -0.625005200213342, -0.022487162744560703, 3.3513450345879296, 1.1390361931507589],
        dtype=float,
    )
    DRAWER_OPEN_CTRL = 1.0

    def orient_err(r_cur: np.ndarray, r_tgt: np.ndarray) -> np.ndarray:
        return 0.5 * (
            np.cross(r_cur[:, 0], r_tgt[:, 0])
            + np.cross(r_cur[:, 1], r_tgt[:, 1])
            + np.cross(r_cur[:, 2], r_tgt[:, 2])
        )

    def solve_midpoint_ik(model, data, hand_id, left_id, right_id, target_midpoint, seed, target_down):
        q = seed.copy()
        qmin = model.actuator_ctrlrange[:7, 0]
        qmax = model.actuator_ctrlrange[:7, 1]
        target_down = np.array(target_down, dtype=float)
        target_down = target_down / max(np.linalg.norm(target_down), 1e-9)
        for _ in range(160):
            data.qpos[:7] = q
            data.qpos[7:9] = 0.04
            mujoco.mj_forward(model, data)
            hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
            midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
            pos_err = target_midpoint - midpoint
            ori_err = orient_err(hand_z.reshape(3, 1), target_down.reshape(3, 1)) if False else np.cross(hand_z, target_down)
            jacp_l = np.zeros((3, model.nv))
            jacr_l = np.zeros((3, model.nv))
            jacp_r = np.zeros((3, model.nv))
            jacr_r = np.zeros((3, model.nv))
            jacp_h = np.zeros((3, model.nv))
            jacr_h = np.zeros((3, model.nv))
            mujoco.mj_jacBodyCom(model, data, jacp_l, jacr_l, left_id)
            mujoco.mj_jacBodyCom(model, data, jacp_r, jacr_r, right_id)
            mujoco.mj_jacBody(model, data, jacp_h, jacr_h, hand_id)
            jac = np.vstack([0.5 * (jacp_l[:, :7] + jacp_r[:, :7]), 0.35 * jacr_h[:, :7]])
            err = np.concatenate([pos_err, 0.35 * ori_err])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), err)
            q = np.clip(q + 0.8 * dq, qmin, qmax)
        return q

    def best_q(model, data, hand_id, left_id, right_id, target_xyz, target_down, seeds, extra_seed=None):
        target = np.array(target_xyz, dtype=float)
        candidates = []
        all_seeds = list(seeds)
        if extra_seed is not None:
            all_seeds.append(extra_seed.copy())
        for seed in all_seeds:
            q = solve_midpoint_ik(model, data, hand_id, left_id, right_id, target, seed, target_down)
            data.qpos[:7] = q
            data.qpos[7:9] = 0.04
            mujoco.mj_forward(model, data)
            midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
            hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
            err = np.linalg.norm(midpoint - target) + 0.2 * np.linalg.norm(np.cross(hand_z, target_down / max(np.linalg.norm(target_down), 1e-9)))
            candidates.append((err, q))
        return min(candidates, key=lambda item: item[0])[1]

    def move_q(sim, q_target, grip, steps, drawer_ctrl=DRAWER_OPEN_CTRL):
        q_start = sim.data.qpos[:7].copy()
        g_start = float(sim.data.ctrl[7])
        d_start = float(sim.data.ctrl[8])
        for i in range(steps):
            a = (i + 1) / steps
            sim.data.ctrl[:7] = (1 - a) * q_start + a * q_target
            sim.data.ctrl[7] = (1 - a) * g_start + a * grip
            sim.data.ctrl[8] = (1 - a) * d_start + a * drawer_ctrl
            sim.step(1)

    def hold(sim, q_target, grip, steps, drawer_ctrl=DRAWER_OPEN_CTRL):
        for _ in range(steps):
            sim.data.ctrl[:7] = q_target
            sim.data.ctrl[7] = grip
            sim.data.ctrl[8] = drawer_ctrl
            sim.step(1)

    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.data.ctrl[8] = DRAWER_OPEN_CTRL
    sim.step(200)
    move_q(sim, Q_HANDLE, 255, 260)
    move_q(sim, Q_HANDLE, 40, 160)
    move_q(sim, Q_PULL, 40, 320)
    hold(sim, Q_PULL, 40, 120)
    move_q(sim, Q_PULL, 255, 100)
    hold(sim, Q_PULL, 255, 40)
    move_q(sim, SAFE, 255, 180)
    hold(sim, SAFE, 255, 40)


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
