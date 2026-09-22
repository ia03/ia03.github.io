import numpy as np
import mujoco
from PIL import Image

from sim import Sim


HAND_BODY = "hand"
PEG_BODY = "peg"
HAND_TO_PINCH = np.array([0.0378, 0.0378, 0.0844])
TARGET_R = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)
NOMINAL_Q = np.array([0.35, 0.25, 0.0, -2.1, 0.0, 2.35, 0.78])


def rotation_error(R_current, R_target):
    return 0.5 * (
        np.cross(R_current[:, 0], R_target[:, 0])
        + np.cross(R_current[:, 1], R_target[:, 1])
        + np.cross(R_current[:, 2], R_target[:, 2])
    )


def desired_hand_pose(grasp_center):
    hand_pos = np.array(grasp_center) - TARGET_R @ HAND_TO_PINCH
    return hand_pos, TARGET_R


def set_gripper(sim, open_gripper):
    sim.data.ctrl[7] = 255.0 if open_gripper else 0.0


def drive_to_pose(sim, hand_target, steps, open_gripper, pos_gain=4.0, rot_gain=2.5):
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    for _ in range(steps):
        q = sim.data.qpos[:7].copy()
        hand_pos = sim.data.xpos[hand_id].copy()
        R_current = sim.data.xmat[hand_id].reshape(3, 3).copy()
        pos_err = hand_target - hand_pos
        rot_err = rotation_error(R_current, TARGET_R)
        task = np.concatenate([pos_gain * pos_err, rot_gain * rot_err])
        mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, hand_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), task)
        dq += 0.05 * (NOMINAL_Q - q)
        q_target = np.clip(
            q + 0.08 * dq,
            sim.model.actuator_ctrlrange[:7, 0],
            sim.model.actuator_ctrlrange[:7, 1],
        )
        sim.data.ctrl[:7] = q_target
        set_gripper(sim, open_gripper)
        sim.step()


def move_linear(sim, start, end, segments=60, steps_per=10, open_gripper=True):
    for a in np.linspace(0.0, 1.0, segments):
        p = (1.0 - a) * start + a * end
        drive_to_pose(sim, p, steps=steps_per, open_gripper=open_gripper)


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


def score_like_hidden(sim):
    metrics = peg_metrics(sim)
    pos = metrics["pos"]
    align = metrics["align"]
    dx = max(0.0, 0.695 - pos[0])
    dy = max(0.0, abs(pos[1] - 0.08) - 0.02)
    dz = max(0.0, abs(pos[2] - 0.48) - 0.02)
    da = max(0.0, 0.85 - align)
    score = -(4.0 * dx + 2.0 * dy + 2.0 * dz + da)
    if metrics["pass"]:
        score += 10.0
    return score, metrics


def save_frame(sim, path):
    import imageio.v2 as iio

    img = sim.render(width=640, height=480)
    iio.imwrite(path, img)


def run_attempt(name: str, grasp_z: float, pre_y: float, carry_z: float, insert_x: float, settle: int = 240):
    sim = Sim()

    peg_start = sim.peg_position().copy()
    hand_from_pinch = TARGET_R @ HAND_TO_PINCH
    pinch_hover = peg_start + np.array([0.0, 0.0, 0.14])
    pinch_pregrasp = peg_start + np.array([0.0, 0.0, 0.05])
    pinch_grasp = peg_start + np.array([0.0, 0.0, grasp_z])
    hover = pinch_hover - hand_from_pinch
    pregrasp = pinch_pregrasp - hand_from_pinch
    grasp = pinch_grasp - hand_from_pinch
    lift = np.array([0.56, -0.02, 0.48 + carry_z])
    pre_insert = np.array([0.62, 0.08, 0.48])
    insert = np.array([insert_x, 0.08, 0.48])

    move_linear(sim, hover, hover, segments=1, steps_per=1, open_gripper=True)
    sim.step(20)
    sim.save_final_state("/work/final_state.npz")

    move_linear(sim, hover, pregrasp, segments=40, steps_per=8, open_gripper=True)
    move_linear(sim, pregrasp, grasp, segments=24, steps_per=8, open_gripper=True)
    hold(sim, 40, open_gripper=False)
    sim.save_final_state("/work/final_state.npz")

    move_linear(sim, grasp, lift, segments=60, steps_per=8, open_gripper=False)
    hold(sim, 50, open_gripper=False)
    sim.save_final_state("/work/final_state.npz")

    move_linear(sim, lift, pre_insert, segments=70, steps_per=8, open_gripper=False)
    move_linear(sim, pre_insert, insert, segments=40, steps_per=8, open_gripper=False)
    hold(sim, settle, open_gripper=False)
    sim.save_final_state("/work/final_state.npz")

    for _ in range(4):
        hold(sim, 80, open_gripper=False)
        sim.save_final_state("/work/final_state.npz")

    save_frame(sim, f"/work/{name}.png")
    return sim


def main():
    attempts = [
        ("attempt1", 0.018, 0.0, 0.10, 0.715, 260),
        ("attempt2", 0.014, 0.0, 0.09, 0.720, 280),
        ("attempt3", 0.020, 0.0, 0.11, 0.710, 280),
    ]
    best_sim = None
    best_score = float("-inf")
    best_name = None
    best_metrics = None
    for params in attempts:
        sim = run_attempt(*params)
        score, metrics = score_like_hidden(sim)
        print(params[0], metrics, score)
        if score > best_score:
            best_score = score
            best_sim = sim
            best_name = params[0]
            best_metrics = metrics
            sim.save_final_state("/work/final_state.npz")
    print("best", best_name, best_metrics, best_score)
    if best_sim is not None:
        best_sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
