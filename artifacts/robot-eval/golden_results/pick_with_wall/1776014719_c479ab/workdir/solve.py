import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


HAND_BODY = "hand"
LEFT_FINGER_BODY = "left_finger"
RIGHT_FINGER_BODY = "right_finger"
TARGET_DOWN = np.array([0.0, 0.0, -1.0])


def orientation_error(R: np.ndarray, Rd: np.ndarray) -> np.ndarray:
    return 0.5 * (
        np.cross(R[:, 0], Rd[:, 0])
        + np.cross(R[:, 1], Rd[:, 1])
        + np.cross(R[:, 2], Rd[:, 2])
    )


def rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def servo_pose(sim: Sim, p_des: np.ndarray, R_des: np.ndarray, gripper: float, steps: int, q_nom=None):
    if q_nom is None:
        q_nom = np.array([0.0, -0.6, 0.0, -1.9, 0.0, 1.5, 0.7])

    left_id = sim.model.body(LEFT_FINGER_BODY).id
    right_id = sim.model.body(RIGHT_FINGER_BODY).id
    hand_id = sim.model.body(HAND_BODY).id
    had_contact = False

    for _ in range(steps):
        jacp_l = np.zeros((3, sim.model.nv))
        jacp_r = np.zeros((3, sim.model.nv))
        tmp = np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, jacp_l, tmp, left_id)
        mujoco.mj_jacBody(sim.model, sim.data, jacp_r, tmp, right_id)
        j_pos = 0.5 * (jacp_l[:, :7] + jacp_r[:, :7])

        pinch = 0.5 * (sim.data.xpos[left_id] + sim.data.xpos[right_id])
        hand_R = np.array(sim.data.xmat[hand_id]).reshape(3, 3)

        jp = np.zeros((3, sim.model.nv))
        jr = np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, jp, jr, hand_id)
        j_rot = jr[:, :7]

        pos_gain = 12.0 if p_des[2] < 0.55 else 10.0
        rot_gain = 2.5
        err = np.concatenate(
            [
                pos_gain * (p_des - pinch),
                rot_gain * orientation_error(hand_R, R_des),
                0.02 * (q_nom - sim.data.qpos[:7]),
            ]
        )
        J = np.vstack([j_pos, j_rot, 0.05 * np.eye(7)])
        dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(7), J.T @ err)
        sim.data.ctrl[:7] = np.clip(
            sim.data.qpos[:7] + 0.16 * dq,
            sim.model.actuator_ctrlrange[:7, 0],
            sim.model.actuator_ctrlrange[:7, 1],
        )
        sim.data.ctrl[7] = gripper
        sim.step()
        had_contact = had_contact or sim.has_gripper_cup_contact()

    pinch = 0.5 * (sim.data.xpos[left_id] + sim.data.xpos[right_id])
    return pinch, had_contact


def run_candidate(params, save=False):
    yaw, x_far, y_grasp, z_grasp, x_return, y_return, z_lift = params
    R = rotz(yaw) @ np.diag([1.0, 1.0, -1.0])
    sim = Sim()

    # A replay-first baseline: approach high, pinch, close fully, then lift and retreat over the wall.
    sequence = [
        (np.array([x_far, 0.18, max(0.74, z_lift + 0.08)]), 255.0, 160),
        (np.array([x_far, y_grasp, max(0.62, z_grasp + 0.10)]), 255.0, 160),
        (np.array([x_far, y_grasp, z_grasp + 0.03]), 255.0, 140),
        (np.array([x_far, y_grasp, z_grasp]), 255.0, 120),
    ]

    replay_contact = False
    for p_des, grip, steps in sequence:
        _, had = servo_pose(sim, p_des, R, grip, steps=steps)
        replay_contact = replay_contact or had

    # Fully close and let the grasp settle before moving.
    _, had = servo_pose(sim, np.array([x_far, y_grasp, z_grasp]), R, 0.0, steps=160)
    replay_contact = replay_contact or had

    post_grasp = [
        np.array([x_far, y_grasp, max(z_lift, 0.74)]),
        np.array([x_far, y_return, max(z_lift, 0.74)]),
        np.array([x_return, y_return, max(z_lift, 0.74)]),
        np.array([x_return, y_return, max(z_lift, 0.70)]),
    ]
    for target in post_grasp:
        _, had = servo_pose(sim, target, R, 0.0, steps=180)
        replay_contact = replay_contact or had

    for _ in range(180):
        sim.data.ctrl[:7] = sim.data.qpos[:7]
        sim.data.ctrl[7] = 0.0
        sim.step()

    # Score from replayed trace and final settle.
    trace = np.array(sim._trace, dtype=object)
    cup_trace = np.array([entry["cup_pos"] for entry in sim._trace], dtype=float)
    contact_trace = np.array([entry["cup_contact"] for entry in sim._trace], dtype=float)
    ever_contact = bool(np.any(contact_trace > 0.5))
    best_x = float(np.min(cup_trace[:, 0]))
    best_z = float(np.max(cup_trace[:, 2]))
    final = sim.cup_position().copy()

    settle_contacts = []
    sim2 = Sim()
    sim2.data.qpos[:] = sim.data.qpos
    sim2.data.qvel[:] = sim.data.qvel
    sim2.data.ctrl[:] = sim.data.ctrl
    mujoco.mj_forward(sim2.model, sim2.data)
    for _ in range(500):
        settle_contacts.append(float(sim2.has_gripper_cup_contact()))
        sim2.step(1)
    settle_contact = float(np.mean(settle_contacts))

    grasp_progress = 1.0 if ever_contact else 0.0
    return_progress = float(np.clip((0.57 - best_x) / (0.57 - 0.50), 0.0, 1.0))
    lift_progress = float(np.clip((best_z - 0.435) / (0.56 - 0.435), 0.0, 1.0))
    settle_progress = float(np.clip(settle_contact / 0.5, 0.0, 1.0))
    s1 = grasp_progress
    s2 = grasp_progress * return_progress
    s3 = s2 * lift_progress
    s4 = s3 * settle_progress
    progress = (s1 + s2 + s3 + s4) / 4.0
    passed = (
        len(sim._ctrl_trace) >= 20
        and ever_contact
        and best_x <= 0.50
        and best_z >= 0.56
        and float(final[0]) <= 0.42
        and float(final[2]) >= 0.56
        and settle_contact >= 0.5
    )

    if save:
        sim.save_final_state("/work/final_state.npz")

    return {
        "params": params,
        "progress": progress,
        "passed": passed,
        "contact": ever_contact,
        "best_x": best_x,
        "best_z": best_z,
        "final": final,
        "settle": settle_contact,
        "sim": sim,
    }


def main():
    candidates = []
    for yaw in [2.62, -2.62, math.pi / 2, -math.pi / 2]:
        for x_far in [0.57, 0.59, 0.61]:
            for y_grasp in [0.14, 0.16, 0.18]:
                for z_grasp in [0.45, 0.47]:
                    for x_return in [0.40, 0.36]:
                        for z_lift in [0.70, 0.74, 0.78]:
                            candidates.append((yaw, x_far, y_grasp, z_grasp, x_return, 0.24, z_lift))

    best = None
    for i, params in enumerate(candidates, 1):
        result = run_candidate(params, save=False)
        print(
            f"trial {i}: score={result['progress']:.3f} passed={result['passed']} "
            f"contact={result['contact']} best_x={result['best_x']:.3f} best_z={result['best_z']:.3f} "
            f"final=({result['final'][0]:.3f}, {result['final'][2]:.3f}) settle={result['settle']:.3f}"
        )
        if best is None or result["progress"] > best["progress"] or result["passed"]:
            best = result
            best["sim"].save_final_state("/work/final_state.npz")
            print("saved best-so-far to /work/final_state.npz")
        if result["passed"]:
            break

    if best is not None:
        print("best params", best["params"])


if __name__ == "__main__":
    main()
