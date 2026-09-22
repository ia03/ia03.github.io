import math
import os

import mujoco
import numpy as np

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


def servo_pose(sim, p_des, r_des, gripper, steps, q_nom=None, pos_gain=11.0, wrist7=None):
    if q_nom is None:
        q_nom = np.array([0.0, -0.6, 0.0, -1.9, 0.0, 1.5, 0.7], dtype=float)

    left_id = sim.model.body("left_finger").id
    right_id = sim.model.body("right_finger").id
    hand_id = sim.model.body("hand").id
    had_contact = False

    for _ in range(steps):
        jacp_l = np.zeros((3, sim.model.nv))
        jacp_r = np.zeros((3, sim.model.nv))
        tmp = np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, jacp_l, tmp, left_id)
        mujoco.mj_jacBody(sim.model, sim.data, jacp_r, tmp, right_id)
        j_pos = 0.5 * (jacp_l[:, :7] + jacp_r[:, :7])

        pinch = 0.5 * (sim.data.xpos[left_id] + sim.data.xpos[right_id])
        hand_r = np.array(sim.data.xmat[hand_id]).reshape(3, 3)

        jp = np.zeros((3, sim.model.nv))
        jr = np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, jp, jr, hand_id)
        j_rot = jr[:, :7]

        err = np.concatenate(
            [
                pos_gain * (p_des - pinch),
                2.8 * orientation_error(hand_r, r_des),
                0.03 * (q_nom - sim.data.qpos[:7]),
            ]
        )
        j = np.vstack([j_pos, j_rot, 0.05 * np.eye(7)])
        dq = np.linalg.solve(j.T @ j + 1e-4 * np.eye(7), j.T @ err)

        q_next = np.clip(
            sim.data.qpos[:7] + 0.18 * dq,
            sim.model.actuator_ctrlrange[:7, 0],
            sim.model.actuator_ctrlrange[:7, 1],
        )
        if wrist7 is not None:
            q_next[6] = wrist7
        sim.data.ctrl[:7] = q_next
        sim.data.ctrl[7] = gripper
        sim.step()
        had_contact = had_contact or sim.has_gripper_cup_contact()

    return had_contact


def settle_and_score(sim, replay_contact):
    best_x = float(sim.cup_position()[0])
    best_z = float(sim.cup_position()[2])
    for entry in sim._trace:
        cup = entry["cup_pos"]
        best_x = min(best_x, float(cup[0]))
        best_z = max(best_z, float(cup[2]))

    settle_contact = 0
    for _ in range(500):
        sim.step()
        settle_contact += int(sim.has_gripper_cup_contact())
        cup = sim.cup_position()
        best_x = min(best_x, float(cup[0]))
        best_z = max(best_z, float(cup[2]))

    final = sim.cup_position()
    settle_fraction = settle_contact / 500.0
    passed = (
        len(sim._ctrl_trace) >= 20
        and replay_contact
        and best_x <= 0.50
        and best_z >= 0.56
        and final[0] <= 0.42
        and final[2] >= 0.56
        and settle_fraction >= 0.5
    )
    score = (
        1000.0 * float(passed)
        + 100.0 * float(replay_contact)
        + 90.0 * float(best_x <= 0.50)
        + 90.0 * float(best_z >= 0.56)
        + 180.0 * settle_fraction
        - 200.0 * max(final[0] - 0.42, 0.0)
        + 300.0 * min(final[2], 0.56)
    )
    return {
        "score": score,
        "passed": passed,
        "best_x": best_x,
        "best_z": best_z,
        "final": final,
        "settle_fraction": settle_fraction,
        "replay_contact": replay_contact,
    }


def run_plan(params):
    yaw = params["yaw"]
    x_far = params["x_far"]
    y_grasp = params["y_grasp"]
    z_grasp = params["z_grasp"]
    x_return = params["x_return"]
    y_return = params["y_return"]
    z_lift = params["z_lift"]
    final_y = params["final_y"]
    final_z = params["final_z"]
    close_grip = params["close_grip"]
    close_steps = params["close_steps"]
    wrist7 = params["wrist7"]

    sim = Sim()
    r = rotz(yaw) @ np.diag([1.0, 1.0, -1.0])
    replay_contact = False

    seq = [
        (np.array([0.56, 0.20, 0.78]), 255.0, 150, 9.0),
        (np.array([x_far, 0.20, 0.68]), 255.0, 140, 10.0),
        (np.array([x_far, y_grasp, z_grasp + 0.04]), 255.0, 150, 12.0),
        (np.array([x_far, y_grasp, z_grasp]), 180.0, 130, 12.5),
        (np.array([x_far, y_grasp, z_grasp]), close_grip, close_steps, 12.5),
        (np.array([x_far, y_return, z_lift]), close_grip, 180, 10.0),
        (np.array([x_return, y_return, z_lift]), close_grip, 220, 10.0),
        (np.array([x_return, final_y, final_z]), close_grip, 200, 10.0),
        (np.array([x_return, final_y, final_z]), close_grip, 160, 8.0),
    ]

    for p_des, grip, steps, pos_gain in seq:
        replay_contact = servo_pose(sim, p_des, r, grip, steps, pos_gain=pos_gain, wrist7=wrist7) or replay_contact

    metrics = settle_and_score(sim, replay_contact)
    metrics["params"] = params
    return sim, metrics


def candidate_params():
    for grasp_z in [0.42, 0.435, 0.45]:
        for lift_z in [0.82, 0.86]:
            for close_steps in [240, 320]:
                    yield {
                        "yaw": math.pi / 2,
                        "x_far": 0.57,
                        "y_grasp": 0.10,
                        "z_grasp": grasp_z,
                        "close_grip": 0.0,
                        "close_steps": close_steps,
                        "x_return": 0.38,
                        "y_return": 0.10,
                        "z_lift": lift_z,
                        "final_y": 0.06,
                        "final_z": max(0.60, lift_z - 0.02),
                        "wrist7": 1.0,
                    }


def main():
    best_metrics = None
    best_sim = None
    for i, params in enumerate(candidate_params(), start=1):
        sim, metrics = run_plan(params)
        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_metrics = metrics
            best_sim = sim
            best_sim.save_final_state("/work/final_state.npz")
            print("new best", i, metrics, flush=True)
        if i % 20 == 0:
            print("checked", i, "best", best_metrics, flush=True)
        if best_metrics["passed"]:
            break

    if best_sim is not None and not os.path.exists("/work/final_state.npz"):
        best_sim.save_final_state("/work/final_state.npz")
    print("final best", best_metrics, flush=True)


if __name__ == "__main__":
    main()
