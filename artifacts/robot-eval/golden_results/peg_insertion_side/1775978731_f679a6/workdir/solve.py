import numpy as np
import mujoco

from sim import Sim


ARM_DOF = np.arange(7)
Q_HOME = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 2.0, 0.78])
LEFT_FINGER_BODY = "left_finger"
RIGHT_FINGER_BODY = "right_finger"
HAND_BODY = "hand"


def pose_error(current_R, target_R):
    return 0.5 * (
        np.cross(current_R[:, 0], target_R[:, 0])
        + np.cross(current_R[:, 1], target_R[:, 1])
        + np.cross(current_R[:, 2], target_R[:, 2])
    )


class Controller:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.hand_body = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.left_finger = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, LEFT_FINGER_BODY)
        self.right_finger = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_FINGER_BODY)
        self.offset_local = np.array([0.0, 0.0, 0.0584])
        self.q_home = Q_HOME.copy()

    def hand_pose(self):
        pos = np.array(self.sim.data.xpos[self.hand_body])
        R = np.array(self.sim.data.xmat[self.hand_body]).reshape(3, 3)
        return pos, R

    def ik_step(self, target_mid, target_R, gripper, pos_gain=3.0, rot_gain=1.25):
        hand_pos, hand_R = self.hand_pose()
        point = hand_pos + hand_R @ self.offset_local

        jacp = np.zeros((3, self.sim.model.nv))
        jacr = np.zeros((3, self.sim.model.nv))
        mujoco.mj_jac(self.sim.model, self.sim.data, jacp, jacr, point, self.hand_body)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])

        pos_err = target_mid - point
        rot_err = pose_error(hand_R, target_R)
        err = np.concatenate([pos_gain * pos_err, rot_gain * rot_err])

        H = J.T @ J + 1e-4 * np.eye(7)
        bias = 0.02 * (self.q_home - self.sim.data.qpos[:7])
        dq = np.linalg.solve(H, J.T @ err + bias)
        q_target = self.sim.data.qpos[:7] + np.clip(dq, -0.05, 0.05)
        q_target = np.clip(
            q_target,
            self.sim.model.actuator_ctrlrange[:7, 0],
            self.sim.model.actuator_ctrlrange[:7, 1],
        )
        self.sim.data.ctrl[:7] = q_target
        self.sim.data.ctrl[7] = gripper
        self.sim.step(1)

    def move_to(self, target_mid, target_R, gripper, seconds):
        steps = max(1, int(seconds / 0.002))
        for _ in range(steps):
            self.ik_step(target_mid, target_R, gripper)


def score(sim: Sim):
    peg = sim.peg_position()
    peg_R = np.array(sim.data.xmat[sim.peg_body_id]).reshape(3, 3)
    align = abs(float(peg_R[0, 0]))
    insertion_progress = np.clip((peg[0] - 0.47) / (0.695 - 0.47), 0.0, 1.0)
    y_progress = np.clip(1.0 - abs(peg[1] - 0.08) / 0.02, 0.0, 1.0)
    z_progress = np.clip(1.0 - abs(peg[2] - 0.48) / 0.02, 0.0, 1.0)
    alignment_progress = np.clip(align / 0.85, 0.0, 1.0)
    s1 = insertion_progress
    s2 = insertion_progress * y_progress
    s3 = insertion_progress * y_progress * z_progress
    s4 = insertion_progress * y_progress * z_progress * alignment_progress
    return (s1 + s2 + s3 + s4) / 4.0, peg, align


def run_attempt(save_path="/work/final_state.npz", verbose=True):
    sim = Sim()
    ctrl = Controller(sim)
    target_R = np.eye(3)

    peg0 = sim.peg_position().copy()
    if verbose:
        print("start peg", peg0)

    stages = [
        ("open_home", np.array([0.36, -0.10, 0.55]), 255, 2.0),
        ("pregrasp", peg0 + np.array([-0.05, 0.00, 0.09]), 255, 2.0),
        ("align_grasp", peg0 + np.array([-0.01, 0.00, 0.012]), 255, 1.6),
        ("close", peg0 + np.array([-0.01, 0.00, 0.012]), 0, 1.5),
        ("lift", np.array([0.50, -0.10, 0.50]), 0, 2.0),
        ("translate_y", np.array([0.60, -0.02, 0.50]), 0, 1.8),
        ("approach_slot", np.array([0.66, 0.04, 0.49]), 0, 2.0),
        ("insert", np.array([0.705, 0.08, 0.48]), 0, 2.8),
        ("settle", np.array([0.705, 0.08, 0.48]), 0, 1.2),
    ]

    for name, target_mid, grip, seconds in stages:
        ctrl.move_to(target_mid, target_R, grip, seconds)
        if verbose:
            progress, peg, align = score(sim)
            print(name, "peg", peg, "align", align, "progress", progress)

    sim.step(800)
    sim.save_final_state(save_path)
    return sim


if __name__ == "__main__":
    sim = run_attempt()
    progress, peg, align = score(sim)
    print("final peg", peg)
    print("alignment", align)
    print("progress_score", progress)
