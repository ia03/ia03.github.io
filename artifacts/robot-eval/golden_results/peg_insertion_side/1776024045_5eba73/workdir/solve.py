import numpy as np
import mujoco

from sim import Sim


OPEN_GRIP = 255.0
CLOSE_GRIP = 0.0
SAVE_PATH = "/work/final_state.npz"

PEG_TARGET = np.array([0.705, 0.08, 0.48], dtype=float)
PEG_START = np.array([0.47, -0.12, 0.4 + 0.012], dtype=float)

# Keep the hand x-axis aligned with world x while the peg is being held.
TARGET_ROT = np.column_stack(
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    ]
)


def pose_error(current_R, target_R):
    return 0.5 * (
        np.cross(current_R[:, 0], target_R[:, 0])
        + np.cross(current_R[:, 1], target_R[:, 1])
        + np.cross(current_R[:, 2], target_R[:, 2])
    )


def lerp(a, b, t):
    return (1.0 - t) * a + t * b


class Controller:
    def __init__(self):
        self.sim = Sim()
        self.model = self.sim.model
        self.data = self.sim.data
        self.hand_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.left_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.peg_body = self.sim.peg_body_id
        self.joint_min = self.model.actuator_ctrlrange[:7, 0].copy()
        self.joint_max = self.model.actuator_ctrlrange[:7, 1].copy()
        self.offset_local = np.array([0.0, 0.0, 0.0584], dtype=float)
        self.q_home = np.array([0.0, -0.7, 0.0, -2.2, 0.0, 2.0, 0.78], dtype=float)

    def hand_pose(self):
        pos = np.array(self.data.xpos[self.hand_body])
        rot = np.array(self.data.xmat[self.hand_body]).reshape(3, 3)
        return pos, rot

    def grip_midpoint(self):
        lf = np.array(self.data.xpos[self.left_body])
        rf = np.array(self.data.xpos[self.right_body])
        return 0.5 * (lf + rf)

    def peg_pose(self):
        pos = np.array(self.data.xpos[self.peg_body])
        rot = np.array(self.data.xmat[self.peg_body]).reshape(3, 3)
        return pos, rot

    def peg_x_axis(self):
        return self.peg_pose()[1][:, 0].copy()

    def ik_step(self, target_mid, target_rot, grip, pos_gain=3.0, rot_gain=2.0, damping=1e-4, step_gain=0.75):
        q = self.data.qpos[:7].copy()
        hand_pos, hand_rot = self.hand_pose()
        point = hand_pos + hand_rot @ self.offset_local

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.hand_body)
        J = np.vstack([pos_gain * jacp[:, :7], rot_gain * jacr[:, :7]])

        pos_err = target_mid - point
        rot_err = pose_error(hand_rot, target_rot)
        err = np.concatenate([pos_gain * pos_err, rot_gain * rot_err])

        H = J.T @ J + damping * np.eye(7)
        bias = 0.02 * (self.q_home - q)
        dq = np.linalg.solve(H, J.T @ err + bias)
        q_des = q + step_gain * np.clip(dq, -0.06, 0.06)
        q_des = np.clip(q_des, self.joint_min, self.joint_max)

        self.data.ctrl[:7] = q_des
        self.data.ctrl[7] = grip
        self.sim.step(1)

    def move_to(self, target_mid, target_rot, grip, steps, pos_tol=0.006, rot_tol=0.05):
        for _ in range(steps):
            self.ik_step(target_mid, target_rot, grip)
            hand_pos, hand_rot = self.hand_pose()
            point = hand_pos + hand_rot @ self.offset_local
            pos_err = np.linalg.norm(target_mid - point)
            rot_err = np.linalg.norm(pose_error(hand_rot, target_rot))
            if pos_err < pos_tol and rot_err < rot_tol:
                break

    def settle(self, steps, grip=CLOSE_GRIP):
        self.data.ctrl[:7] = self.data.qpos[:7]
        self.data.ctrl[7] = grip
        self.sim.step(steps)

    def move_grasped_peg(self, desired_peg, grip, steps):
        peg_pos, _ = self.peg_pose()
        grasp_offset = self.grip_midpoint() - peg_pos
        target_mid = desired_peg + grasp_offset
        self.move_to(target_mid, TARGET_ROT, grip, steps)

    def run(self):
        # Establish the arm in a comfortable approach configuration.
        self.data.ctrl[:7] = self.q_home
        self.data.ctrl[7] = OPEN_GRIP
        self.sim.step(600)

        peg0 = self.peg_pose()[0].copy()
        above = np.array([peg0[0], peg0[1], peg0[2] + 0.16], dtype=float)
        pregrasp = np.array([peg0[0], peg0[1], peg0[2] + 0.035], dtype=float)
        grasp = np.array([peg0[0], peg0[1], peg0[2] + 0.006], dtype=float)
        lift = np.array([peg0[0] + 0.025, peg0[1], 0.56], dtype=float)

        # Approach and secure the peg before any lateral motion.
        self.move_to(above, TARGET_ROT, OPEN_GRIP, steps=420)
        self.move_to(pregrasp, TARGET_ROT, OPEN_GRIP, steps=320)
        self.move_to(grasp, TARGET_ROT, OPEN_GRIP, steps=220)
        self.move_to(grasp, TARGET_ROT, CLOSE_GRIP, steps=220)
        self.settle(180, grip=CLOSE_GRIP)
        self.move_to(lift, TARGET_ROT, CLOSE_GRIP, steps=320)
        self.settle(160, grip=CLOSE_GRIP)

        # Dense, incremental transport path.  Keep the peg high while moving laterally.
        peg_pos, _ = self.peg_pose()
        route = [
            np.array([0.54, peg_pos[1], 0.58], dtype=float),
            np.array([0.58, -0.08, 0.58], dtype=float),
            np.array([0.62, -0.05, 0.57], dtype=float),
            np.array([0.66, -0.02, 0.55], dtype=float),
            np.array([0.68, 0.02, 0.53], dtype=float),
            np.array([0.695, 0.05, 0.50], dtype=float),
            np.array([0.705, 0.08, 0.49], dtype=float),
            PEG_TARGET.copy(),
            PEG_TARGET.copy() + np.array([0.02, 0.0, 0.0], dtype=float),
        ]

        current = peg_pos.copy()
        for target in route:
            # Interpolate a short sub-path so each correction is small.
            for alpha in np.linspace(0.2, 1.0, 4):
                waypoint = lerp(current, target, float(alpha))
                self.move_grasped_peg(waypoint, CLOSE_GRIP, steps=220)
            current = target.copy()

        self.settle(320, grip=CLOSE_GRIP)
        self.sim.save_final_state(SAVE_PATH)


if __name__ == "__main__":
    Controller().run()
