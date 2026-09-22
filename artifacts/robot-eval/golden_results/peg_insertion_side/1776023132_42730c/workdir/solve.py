import numpy as np
import mujoco

from sim import Sim


TARGET_DOWN = np.array([0.0, 0.0, -1.0], dtype=float)
SIDE_DIR = np.array([-1.0, 0.0, 0.0], dtype=float)
SIDE_Y = np.array([0.0, 1.0, 0.0], dtype=float)


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


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


def play_segment(sim, q0, q1, g0, g1, steps):
    peak_z = -1e9
    for i in range(steps):
        u = (i + 1) / steps
        sim.data.ctrl[:7] = (1.0 - u) * q0 + u * q1
        sim.data.ctrl[7] = (1.0 - u) * g0 + u * g1
        sim.step(1)
        peak_z = max(peak_z, float(sim.peg_position()[2]))
    return peak_z


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


def evaluate(sim):
    model = sim.model
    peg_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    peg_pos = sim.peg_position().copy()
    peg_x_axis = sim.data.xmat[peg_bid].reshape(3, 3)[:, 0]
    inserted = peg_pos[0] >= 0.695 and abs(peg_pos[1] - 0.08) <= 0.02 and abs(peg_pos[2] - 0.48) <= 0.02
    aligned = abs(float(peg_x_axis[0])) >= 0.85
    dx = max(0.0, 0.695 - peg_pos[0])
    dy = max(0.0, abs(peg_pos[1] - 0.08) - 0.02)
    dz = max(0.0, abs(peg_pos[2] - 0.48) - 0.02)
    da = max(0.0, 0.85 - abs(float(peg_x_axis[0])))
    score = -(4.0 * dx + 2.0 * dy + 2.0 * dz + da)
    if inserted and aligned:
        score += 10.0
    return score, peg_pos, peg_x_axis, inserted, aligned


def run_attempt(sim, model, data, hand_id, left_id, right_id, grasp_z, lift_z, slot_z, settle=260, target_dir=TARGET_DOWN, offset=(0.0, 0.0)):
    seeds = [
        np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.0]),
        np.array([0.08, -0.19, -0.07, -1.88, -0.01, 1.69, 0.5]),
        np.array([-0.08, -0.19, 0.07, -1.88, 0.01, 1.69, -0.5]),
    ]

    peg = sim.peg_position().copy()
    q_start = np.zeros(7)
    q_above = best_q(model, data, hand_id, left_id, right_id, [peg[0] + offset[0], peg[1] + offset[1], 0.55], seeds, target_dir=target_dir)
    q_grasp = best_q(model, data, hand_id, left_id, right_id, [peg[0] + offset[0], peg[1] + offset[1], grasp_z], seeds, target_dir=target_dir)
    q_lift = best_q(model, data, hand_id, left_id, right_id, [peg[0] + offset[0], peg[1] + offset[1], lift_z], seeds, target_dir=target_dir)
    q_slot_high = best_q(model, data, hand_id, left_id, right_id, [0.695, 0.08, slot_z], seeds, target_dir=target_dir)
    q_slot_drop = best_q(
        model,
        data,
        hand_id,
        left_id,
        right_id,
        [0.705, 0.08, max(0.50, slot_z - 0.06)],
        seeds,
        target_dir=target_dir,
    )

    peak_z = float(sim.peg_position()[2])
    peak_z = max(peak_z, play_segment(sim, q_start, q_above, 0.0, 255.0, 220))
    peak_z = max(peak_z, play_segment(sim, q_above, q_grasp, 255.0, 255.0, 120))
    peak_z = max(peak_z, play_segment(sim, q_grasp, q_grasp, 255.0, 40.0, 160))
    peak_z = max(peak_z, play_segment(sim, q_grasp, q_grasp, 40.0, 0.0, 80))
    peak_z = max(peak_z, play_segment(sim, q_grasp, q_lift, 0.0, 0.0, 180))
    peak_z = max(peak_z, play_segment(sim, q_lift, q_slot_high, 0.0, 0.0, 220))
    peak_z = max(peak_z, play_segment(sim, q_slot_high, q_slot_drop, 0.0, 0.0, 120))
    peak_z = max(peak_z, play_segment(sim, q_slot_drop, q_slot_drop, 0.0, 0.0, 80))
    for _ in range(settle):
        sim.step(1)
        peak_z = max(peak_z, float(sim.peg_position()[2]))

    score, peg_pos, peg_x_axis, inserted, aligned = evaluate(sim)
    return {
        "grasp_z": grasp_z,
        "lift_z": lift_z,
        "slot_z": slot_z,
        "target_dir": [float(x) for x in target_dir],
        "offset": [float(x) for x in offset],
        "peg_xyz_after": [float(x) for x in peg_pos],
        "peg_x_axis": [float(x) for x in peg_x_axis],
        "inserted": inserted,
        "aligned": aligned,
        "score": float(score),
        "peak_z": peak_z,
        "replay_steps": int(sim._ctrl_trace and len(sim._ctrl_trace) or 0),
    }


def main():
    sim = Sim()
    model = sim.model
    data = sim.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    attempts = [
        (0.418, 0.56, 0.60, TARGET_DOWN, (0.0, 0.0)),
        (0.418, 0.56, 0.60, TARGET_DOWN, (-0.05, 0.0)),
        (0.418, 0.56, 0.60, TARGET_DOWN, (0.05, 0.0)),
        (0.410, 0.68, 0.68, SIDE_DIR, (-0.09, 0.0)),
        (0.410, 0.68, 0.68, SIDE_DIR, (0.09, 0.0)),
        (0.410, 0.68, 0.68, SIDE_DIR, (0.0, -0.09)),
        (0.410, 0.68, 0.68, SIDE_DIR, (0.0, 0.09)),
        (0.406, 0.72, 0.72, SIDE_DIR, (-0.09, 0.0)),
        (0.406, 0.72, 0.72, SIDE_DIR, (0.09, 0.0)),
        (0.404, 0.74, 0.74, SIDE_DIR, (0.0, -0.09)),
        (0.404, 0.74, 0.74, SIDE_DIR, (0.0, 0.09)),
        (0.414, 0.70, 0.70, SIDE_DIR, (-0.06, 0.0)),
        (0.414, 0.70, 0.70, SIDE_DIR, (0.06, 0.0)),
        (0.410, 0.45, 0.45, SIDE_DIR, (0.0, -0.09)),
        (0.410, 0.47, 0.47, SIDE_DIR, (0.0, -0.09)),
        (0.410, 0.49, 0.49, SIDE_DIR, (0.0, -0.09)),
        (0.410, 0.68, 0.68, SIDE_Y, (0.0, -0.09)),
        (0.410, 0.68, 0.68, SIDE_Y, (0.0, 0.09)),
        (0.406, 0.72, 0.72, SIDE_Y, (0.0, -0.06)),
        (0.406, 0.72, 0.72, SIDE_Y, (0.0, 0.06)),
    ]

    best = None
    attempts_log = []
    for i, (grasp_z, lift_z, slot_z, target_dir, offset) in enumerate(attempts, start=1):
        sim.reset()
        data.ctrl[:7] = np.zeros(7)
        data.ctrl[7] = 255.0
        sim.step(20)
        result = run_attempt(sim, model, data, hand_id, left_id, right_id, grasp_z, lift_z, slot_z, target_dir=target_dir, offset=offset)
        print(f"attempt{i}", result, flush=True)
        attempts_log.append(result)
        if best is None or result["score"] > best["score"]:
            best = result
            sim.save_final_state("/work/final_state.npz")
    print("best", best, flush=True)
    with open("/work/attempt_log.txt", "w", encoding="utf-8") as f:
        for row in attempts_log:
            f.write(repr(row) + "\n")
    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
