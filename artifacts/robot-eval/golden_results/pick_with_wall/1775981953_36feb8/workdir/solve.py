import math
import os

import mujoco
import numpy as np

from sim import Sim


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


def servo_pose(sim, p_des, r_des, gripper, steps, q_nom=None, pos_gain=11.0):
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

        sim.data.ctrl[:7] = np.clip(
            sim.data.qpos[:7] + 0.18 * dq,
            sim.model.actuator_ctrlrange[:7, 0],
            sim.model.actuator_ctrlrange[:7, 1],
        )
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

    final = sim.cup_position().copy()
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

    sim = Sim()
    r = rotz(yaw) @ np.diag([1.0, 1.0, -1.0])
    replay_contact = False

    seq = [
        (np.array([0.56, 0.20, 0.78]), 255.0, 150, 9.0),
        (np.array([x_far, 0.20, 0.68]), 255.0, 140, 10.0),
        (np.array([x_far, y_grasp, z_grasp + 0.04]), 255.0, 150, 12.0),
        (np.array([x_far, y_grasp, z_grasp]), 180.0, 130, 12.5),
        (np.array([x_far, y_grasp, z_grasp]), close_grip, close_steps, 12.5),
        (np.array([x_far, y_return, z_lift]), close_grip, 160, 10.0),
        (np.array([x_return, y_return, z_lift]), close_grip, 200, 10.0),
        (np.array([x_return, final_y, final_z]), close_grip, 180, 10.0),
        (np.array([x_return, final_y, final_z]), close_grip, 160, 8.0),
    ]

    for p_des, grip, steps, pos_gain in seq:
        replay_contact = servo_pose(sim, p_des, r, grip, steps, pos_gain=pos_gain) or replay_contact

    metrics = settle_and_score(sim, replay_contact)
    metrics["params"] = params
    return sim, metrics


def candidate_params():
    for yaw in [math.pi / 2, -math.pi / 2, 2.09, -2.09]:
        for x_far in [0.57, 0.59, 0.61]:
            for y_grasp in [0.10, 0.12, 0.14, 0.16]:
                for z_grasp in [0.44, 0.45, 0.46, 0.47]:
                    for close_grip in [0.0, 10.0, 20.0]:
                        for close_steps in [120, 160, 220]:
                            for x_return in [0.40, 0.38, 0.36]:
                                for y_return in [0.22, 0.24, 0.26]:
                                    for z_lift in [0.60, 0.64, 0.68, 0.72]:
                                        yield {
                                            "yaw": yaw,
                                            "x_far": x_far,
                                            "y_grasp": y_grasp,
                                            "z_grasp": z_grasp,
                                            "close_grip": close_grip,
                                            "close_steps": close_steps,
                                            "x_return": x_return,
                                            "y_return": y_return,
                                            "z_lift": z_lift,
                                            "final_y": 0.06,
                                            "final_z": max(0.60, z_lift - 0.02),
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
