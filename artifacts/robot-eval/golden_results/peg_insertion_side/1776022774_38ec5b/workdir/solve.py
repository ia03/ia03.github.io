import numpy as np
import mujoco

from sim import Sim


TARGET_DOWN = np.array([0.0, 0.0, -1.0], dtype=float)
SIDE_DIR = np.array([-1.0, 0.0, 0.0], dtype=float)


def solve_ik(model, data, hand_id, left_id, right_id, target_midpoint, seed, target_dir=TARGET_DOWN):
    q = seed.copy()
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]

    for _ in range(140):
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)

        hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
        midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        pos_err = target_midpoint - midpoint
        ori_err = np.cross(hand_z, target_dir)

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


def best_q(model, data, hand_id, left_id, right_id, target_xyz, seeds, target_dir=TARGET_DOWN):
    target = np.array(target_xyz, dtype=float)
    candidates = []
    for seed in seeds:
        q = solve_ik(model, data, hand_id, left_id, right_id, target, seed, target_dir=target_dir)
        data.qpos[:7] = q
        data.qpos[7:9] = 0.04
        mujoco.mj_forward(model, data)
        midpoint = 0.5 * (data.xpos[left_id] + data.xpos[right_id])
        hand_z = data.xmat[hand_id].reshape(3, 3)[:, 2]
        err = np.linalg.norm(midpoint - target) + 0.2 * np.linalg.norm(np.cross(hand_z, target_dir))
        candidates.append((err, q))
    return min(candidates, key=lambda item: item[0])[1]


def play_segment(sim, q0, q1, g0, g1, steps, peak_z):
    for i in range(steps):
        u = (i + 1) / steps
        sim.data.ctrl[:7] = (1.0 - u) * q0 + u * q1
        sim.data.ctrl[7] = (1.0 - u) * g0 + u * g1
        sim.step(1)
        peak_z = max(peak_z, float(sim.peg_position()[2]))
    return peak_z


def run_attempt(grasp_z, lift_z, slot_z, drop_z, close_grip, hold_steps, open_after=False):
    sim = Sim()
    model = sim.model
    data = sim.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    seeds = [
        np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0]),
        np.array([0.08, -0.19, -0.07, -1.88, -0.01, 1.69, 0.5]),
        np.array([-0.08, -0.19, 0.07, -1.88, 0.01, 1.69, -0.5]),
    ]

    peg = sim.peg_position().copy()
    q_start = np.zeros(7)
    q_above = best_q(model, data, hand_id, left_id, right_id, [peg[0], peg[1], 0.55], seeds, target_dir=TARGET_DOWN)
    q_grasp = best_q(model, data, hand_id, left_id, right_id, [peg[0], peg[1], grasp_z], seeds, target_dir=TARGET_DOWN)
    q_lift = best_q(model, data, hand_id, left_id, right_id, [peg[0], peg[1], lift_z], seeds, target_dir=TARGET_DOWN)
    q_slot_high = best_q(model, data, hand_id, left_id, right_id, [0.695, 0.08, slot_z], seeds, target_dir=SIDE_DIR)
    q_slot_drop = best_q(model, data, hand_id, left_id, right_id, [0.705, 0.08, drop_z], seeds, target_dir=SIDE_DIR)

    peak_z = float(sim.peg_position()[2])
    peak_z = play_segment(sim, q_start, q_above, 0.0, 255.0, 220, peak_z)
    peak_z = play_segment(sim, q_above, q_grasp, 255.0, 255.0, 120, peak_z)
    peak_z = play_segment(sim, q_grasp, q_grasp, 255.0, close_grip, 180, peak_z)
    peak_z = play_segment(sim, q_grasp, q_lift, close_grip, close_grip, 220, peak_z)
    peak_z = play_segment(sim, q_lift, q_slot_high, close_grip, close_grip, 260, peak_z)
    peak_z = play_segment(sim, q_slot_high, q_slot_drop, close_grip, close_grip, 160, peak_z)
    if open_after:
        peak_z = play_segment(sim, q_slot_drop, q_slot_drop, close_grip, 0.0, 80, peak_z)
    for _ in range(320):
        sim.step(1)
        peak_z = max(peak_z, float(sim.peg_position()[2]))

    peg_pos = sim.peg_position().copy()
    peg_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    peg_x_axis = sim.data.xmat[peg_bid].reshape(3, 3)[:, 0]
    inserted = peg_pos[0] >= 0.695 and abs(peg_pos[1] - 0.08) <= 0.02 and abs(peg_pos[2] - 0.48) <= 0.02
    aligned = abs(float(peg_x_axis[0])) >= 0.85
    return {
        "grasp_z": grasp_z,
        "lift_z": lift_z,
        "slot_z": slot_z,
        "drop_z": drop_z,
        "close_grip": close_grip,
        "hold_steps": hold_steps,
        "open_after": open_after,
        "peg_xyz_after": [float(x) for x in peg_pos],
        "peg_x_axis": [float(x) for x in peg_x_axis],
        "inserted": inserted,
        "aligned": aligned,
        "peak_z": peak_z,
    }


def main():
    configs = [
        (0.435, 0.62, 0.56, 0.48, 10.0),
        (0.435, 0.66, 0.60, 0.48, 10.0),
        (0.435, 0.70, 0.64, 0.48, 10.0),
        (0.435, 0.70, 0.64, 0.50, 10.0),
        (0.435, 0.70, 0.64, 0.52, 10.0),
        (0.450, 0.70, 0.60, 0.50, 20.0),
        (0.450, 0.74, 0.64, 0.50, 20.0),
        (0.450, 0.74, 0.64, 0.52, 20.0),
    ]
    best = None
    for i, cfg in enumerate(configs, start=1):
        result = run_attempt(*cfg, hold_steps=0)
        print(i, result, flush=True)
        if best is None or result["peak_z"] > best["peak_z"]:
            best = result
            print("best so far", best, flush=True)
    print("final best", best, flush=True)


if __name__ == "__main__":
    main()
