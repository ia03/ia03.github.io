import numpy as np
import mujoco

from sim import Sim


HAND_BODY = "hand"
PEG_BODY = "peg"
HAND_TO_PINCH = np.array([0.0, 0.0, 0.103])
BOARD_TARGET = np.array([0.715, 0.08, 0.48])
NOMINAL_Q = np.array([0.35, 0.25, 0.0, -2.1, 0.0, 2.35, 0.78])

TARGET_R = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)


def rotation_error(R_current, R_target):
    return 0.5 * (
        np.cross(R_current[:, 0], R_target[:, 0])
        + np.cross(R_current[:, 1], R_target[:, 1])
        + np.cross(R_current[:, 2], R_target[:, 2])
    )


def make_pose(position, x_axis, z_axis):
    x_axis = np.array(x_axis, dtype=float)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.array(z_axis, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return position, np.column_stack([x_axis, y_axis, z_axis])


def desired_hand_pose(grasp_center, x_axis=(1.0, 0.0, 0.0)):
    _, R = make_pose(np.zeros(3), x_axis=x_axis, z_axis=(0.0, 0.0, -1.0))
    hand_pos = np.array(grasp_center) - R @ HAND_TO_PINCH
    return hand_pos, R


def set_gripper(sim, open_gripper):
    sim.data.ctrl[7] = 255.0 if open_gripper else 0.0


def drive_to_pose(sim, hand_target, R_target, steps, open_gripper, pos_gain=4.0, rot_gain=2.5):
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    for _ in range(steps):
        q = sim.data.qpos[:7].copy()
        hand_pos = sim.data.xpos[hand_id].copy()
        R_current = sim.data.xmat[hand_id].reshape(3, 3).copy()
        pos_err = hand_target - hand_pos
        rot_err = rotation_error(R_current, R_target)
        task = np.concatenate([pos_gain * pos_err, rot_gain * rot_err])
        mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, hand_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), task)
        dq += 0.05 * (NOMINAL_Q - q)
        q_target = np.clip(q + 0.08 * dq, sim.model.actuator_ctrlrange[:7, 0], sim.model.actuator_ctrlrange[:7, 1])
        sim.data.ctrl[:7] = q_target
        set_gripper(sim, open_gripper)
        sim.step()


def hold(sim, steps, open_gripper):
    sim.data.ctrl[:7] = sim.data.qpos[:7]
    set_gripper(sim, open_gripper)
    sim.step(steps)


def peg_metrics(sim):
    peg_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, PEG_BODY)
    pos = sim.data.xpos[peg_id].copy()
    R = sim.data.xmat[peg_id].reshape(3, 3).copy()
    align = abs(float(R[0, 0]))
    inserted = bool(pos[0] >= 0.695 and abs(pos[1] - 0.08) <= 0.02 and abs(pos[2] - 0.48) <= 0.02 and align >= 0.85)
    return {
        "pos": pos,
        "align": align,
        "pass": inserted,
    }


def replay_metrics(ctrl_trace):
    sim = Sim()
    for ctrl in ctrl_trace:
        sim.data.ctrl[:] = ctrl
        sim.step()
    sim.step(500)
    return peg_metrics(sim)


def run_attempt(save_path="/work/final_state.npz"):
    sim = Sim()

    peg_start = sim.peg_position().copy()
    pregrasp_center = peg_start + np.array([0.0, 0.0, 0.11])
    grasp_center = peg_start + np.array([0.0, 0.0, 0.004])
    lift1 = np.array([0.54, -0.10, 0.52])
    lift2 = np.array([0.60, -0.02, 0.52])
    pre_insert = np.array([0.64, 0.08, 0.48])
    insert_deep = BOARD_TARGET.copy()

    plan = [
        (pregrasp_center, True, 260),
        (grasp_center, True, 240),
        (grasp_center, False, 240),
        (lift1, False, 260),
        (lift2, False, 260),
        (pre_insert, False, 260),
        (insert_deep, False, 360),
        (insert_deep, False, 240),
    ]

    best_score = float("-inf")
    for idx, (center, open_gripper, steps) in enumerate(plan):
        hand_target, R_target = desired_hand_pose(center)
        drive_to_pose(sim, hand_target, R_target, steps=steps, open_gripper=open_gripper)
        metrics = peg_metrics(sim)
        score = -(4.0 * max(0.0, 0.695 - metrics["pos"][0]) + 2.0 * max(0.0, abs(metrics["pos"][1] - 0.08) - 0.02) + 2.0 * max(0.0, abs(metrics["pos"][2] - 0.48) - 0.02) + max(0.0, 0.85 - metrics["align"]))
        if score > best_score:
            best_score = score
            sim.save_final_state(save_path)
        if idx == 2:
            hold(sim, 80, open_gripper=False)

    hold(sim, 320, open_gripper=False)
    sim.save_final_state(save_path)
    return sim, replay_metrics(np.array(sim._ctrl_trace))


if __name__ == "__main__":
    sim, metrics = run_attempt()
    print("final", peg_metrics(sim))
    print("replay", metrics)
