import math
import time
import numpy as np
import mujoco
from sim import Sim, BIN_CENTER


def hand_pos(sim, hand_id):
    return np.array(sim.data.xpos[hand_id])


def cup_up_z(sim):
    # body xmat is row-major 3x3 world rotation
    mat = np.array(sim.data.xmat[sim.cup_body_id]).reshape(3, 3)
    local_z_world = mat[:, 2]
    return float(local_z_world[2])


def inside_bin(cup):
    return abs(cup[0] - 0.70) <= 0.046 and abs(cup[1] - 0.14) <= 0.046


def settle_contact_fraction(sim, steps=500):
    c = 0
    for _ in range(steps):
        sim.step(1)
        if sim.has_gripper_cup_contact():
            c += 1
    return c / steps


def replay_contact_metrics(sim):
    trace_contact = np.array([x["cup_contact"] for x in sim._trace], dtype=float)
    if trace_contact.size == 0:
        return 0.0, 0.0
    last10 = trace_contact[-10:] if trace_contact.size >= 10 else trace_contact
    return float(last10.mean()), float(trace_contact.max())


def eval_status(sim):
    best_z = max(float(t["cup_pos"][2]) for t in sim._trace)
    had_contact = any(float(t["cup_contact"]) > 0.5 for t in sim._trace)
    last10_contact, _ = replay_contact_metrics(sim)
    settle_frac = settle_contact_fraction(sim, 500)
    cup = sim.cup_position().copy()
    status = {
        "steps": len(sim._ctrl_trace),
        "had_contact": had_contact,
        "best_z": best_z,
        "replay_last10_contact_frac": last10_contact,
        "settle_contact_frac": settle_frac,
        "inside_bin": inside_bin(cup),
        "cup_z": float(cup[2]),
        "abs_up_z": abs(cup_up_z(sim)),
        "cup": cup,
    }
    status["pass"] = (
        status["steps"] >= 20
        and status["had_contact"]
        and status["best_z"] >= 0.50
        and status["replay_last10_contact_frac"] < 0.5
        and status["settle_contact_frac"] < 0.1
        and status["inside_bin"]
        and status["cup_z"] >= 0.42
        and status["abs_up_z"] >= 0.85
    )
    return status


def move_hand(sim, hand_id, target, steps=250, gain=2.2, max_step=0.06, grip=None):
    m, d = sim.model, sim.data
    for _ in range(steps):
        p = hand_pos(sim, hand_id)
        err = target - p
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacBodyCom(m, d, jacp, None, hand_id)
        j = jacp[:, :7]
        dq = gain * np.linalg.pinv(j, rcond=1e-4) @ err
        n = np.linalg.norm(dq)
        if n > max_step:
            dq = dq * (max_step / (n + 1e-9))
        q_des = d.qpos[:7] + dq
        sim.data.ctrl[:7] = np.clip(q_des, m.actuator_ctrlrange[:7, 0], m.actuator_ctrlrange[:7, 1])
        if grip is not None:
            sim.data.ctrl[7] = grip
        sim.step(1)


def hold(sim, steps, grip=None):
    if grip is not None:
        sim.data.ctrl[7] = grip
    for _ in range(steps):
        sim.step(1)


def run_attempt(params):
    sim = Sim()
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

    open_u = params.get("open_u", 220.0)
    close_u = params.get("close_u", 0.0)
    pre_xy_offset = np.array(params.get("pre_xy_offset", [0.0, 0.0]), dtype=float)
    grasp_xy_offset = np.array(params.get("grasp_xy_offset", [0.0, 0.0]), dtype=float)

    # Initialize with open gripper.
    sim.data.ctrl[7] = open_u
    hold(sim, 100, grip=open_u)

    cup = sim.cup_position()
    pre = cup + np.array([pre_xy_offset[0], pre_xy_offset[1], params.get("pre_z", 0.22)])
    grasp = cup + np.array([grasp_xy_offset[0], grasp_xy_offset[1], params.get("grasp_z", 0.14)])

    move_hand(sim, hand_id, pre, steps=params.get("pre_steps", 320), gain=params.get("gain", 2.4), max_step=params.get("max_step", 0.07), grip=open_u)
    move_hand(sim, hand_id, grasp, steps=params.get("grasp_steps", 320), gain=params.get("gain", 2.4), max_step=params.get("max_step", 0.06), grip=open_u)

    # Close and wait for secure contact.
    hold(sim, params.get("close_wait", 180), grip=close_u)

    # Lift.
    lift = np.array([cup[0] + params.get("lift_x", 0.0), cup[1] + params.get("lift_y", 0.0), params.get("lift_z", 0.70)])
    move_hand(sim, hand_id, lift, steps=params.get("lift_steps", 420), gain=params.get("gain", 2.2), max_step=params.get("max_step", 0.06), grip=close_u)

    # Move over bin.
    over_bin = np.array([
        BIN_CENTER[0] + params.get("bin_x", 0.0),
        BIN_CENTER[1] + params.get("bin_y", 0.0),
        params.get("bin_over_z", 0.68),
    ])
    move_hand(sim, hand_id, over_bin, steps=params.get("to_bin_steps", 520), gain=params.get("gain", 2.2), max_step=params.get("max_step", 0.06), grip=close_u)

    # Lower to release height.
    rel = np.array([
        BIN_CENTER[0] + params.get("bin_x", 0.0),
        BIN_CENTER[1] + params.get("bin_y", 0.0),
        params.get("release_z", 0.57),
    ])
    move_hand(sim, hand_id, rel, steps=params.get("down_steps", 300), gain=params.get("gain", 2.0), max_step=params.get("max_step", 0.05), grip=close_u)

    # Release and retreat.
    hold(sim, params.get("open_wait", 180), grip=open_u)
    retreat = np.array([
        BIN_CENTER[0] - 0.12,
        BIN_CENTER[1],
        params.get("retreat_z", 0.70),
    ])
    move_hand(sim, hand_id, retreat, steps=params.get("retreat_steps", 360), gain=params.get("gain", 2.0), max_step=params.get("max_step", 0.06), grip=open_u)

    # Let object settle during replay tail as well.
    hold(sim, params.get("tail_settle", 200), grip=open_u)

    return sim


def main():
    start = time.time()
    configs = [
        {"name": "baseline"},
        {"name": "offset_a", "grasp_xy_offset": [0.0, 0.01], "bin_y": -0.01, "release_z": 0.56},
        {"name": "offset_b", "grasp_xy_offset": [0.005, 0.0], "bin_x": -0.005, "release_z": 0.555},
        {"name": "offset_c", "grasp_xy_offset": [-0.005, 0.0], "bin_x": 0.004, "release_z": 0.565},
    ]

    best_score = -1.0
    best_status = None
    best_name = None

    for i, cfg in enumerate(configs):
        sim = run_attempt(cfg)
        # Mandatory early save on first full attempt.
        if i == 0:
            sim.save_final_state('/work/final_state.npz')

        status = eval_status(sim)
        score = (
            0.05 * float(status["had_contact"])
            + 0.10 * float(np.clip((status["best_z"] - 0.435) / (0.50 - 0.435), 0, 1))
            + 0.40 * float(status["inside_bin"])
            + 0.10 * float(np.clip(1 - np.linalg.norm(status["cup"][:2] - np.array([0.70, 0.14])) / np.linalg.norm(np.array([0.48, -0.12]) - np.array([0.70, 0.14])), 0, 1))
            + 0.10 * float(np.clip(1 - status["settle_contact_frac"] / 0.1, 0, 1))
            + 0.25 * float(np.clip(status["abs_up_z"] / 0.85, 0, 1))
        )
        print(f"attempt {i+1} {cfg['name']} status={status} approx_score={score:.3f}")

        if score > best_score:
            best_score = score
            best_status = status
            best_name = cfg["name"]
            sim.save_final_state('/work/final_state.npz')

        if status["pass"]:
            print("PASS achieved; stopping early.")
            break

        if time.time() - start > 700:
            print("Time guard reached, stopping sweep.")
            break

    print("best", best_name, "score", best_score, "status", best_status)


if __name__ == '__main__':
    main()
