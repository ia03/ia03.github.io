import math
import numpy as np
import mujoco

from sim import Sim, BOARD_CENTER


ARM_JOINTS = 7
GRASP_LOCAL = np.array([0.0, 0.0, 0.0584])
HAND_Z_DOWN = np.diag([-1.0, 1.0, -1.0])


def rotation_error(current, target):
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )


class Controller:
    def __init__(self, sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.peg_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.arm_low = self.model.actuator_ctrlrange[:ARM_JOINTS, 0].copy()
        self.arm_high = self.model.actuator_ctrlrange[:ARM_JOINTS, 1].copy()

    def hand_rot(self):
        return self.data.xmat[self.hand_id].reshape(3, 3).copy()

    def hand_pos(self):
        return self.data.xpos[self.hand_id].copy()

    def grasp_point(self):
        return self.hand_pos() + self.hand_rot() @ GRASP_LOCAL

    def peg_pos(self):
        return self.data.xpos[self.peg_id].copy()

    def peg_rot(self):
        return self.data.xmat[self.peg_id].reshape(3, 3).copy()

    def set_gripper(self, command):
        self.data.ctrl[7] = float(np.clip(command, 0.0, 255.0))

    def hold(self, steps, grip=None):
        if grip is not None:
            self.set_gripper(grip)
        self.sim.step(steps)

    def move_to(self, grasp_target, rot_target, grip, steps=220, pos_gain=8.0, rot_gain=4.0):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        q = self.data.qpos[:ARM_JOINTS]
        for _ in range(steps):
            current_rot = self.hand_rot()
            current_grasp = self.grasp_point()
            pos_err = grasp_target - current_grasp
            rot_err = rotation_error(current_rot, rot_target)
            twist = np.concatenate([pos_gain * pos_err, rot_gain * rot_err])

            jacp.fill(0.0)
            jacr.fill(0.0)
            mujoco.mj_jac(self.model, self.data, jacp, jacr, current_grasp, self.hand_id)
            J = np.vstack([jacp[:, :ARM_JOINTS], jacr[:, :ARM_JOINTS]])
            lhs = J.T @ J + 1e-4 * np.eye(ARM_JOINTS)
            dq = np.linalg.solve(lhs, J.T @ twist)
            q = np.clip(q + np.clip(dq, -0.04, 0.04), self.arm_low, self.arm_high)
            self.data.ctrl[:ARM_JOINTS] = q
            self.set_gripper(grip)
            self.sim.step()


def attempt(sim):
    ctl = Controller(sim)

    pre_grasp = np.array([0.435, -0.12, 0.495])
    grasp = np.array([0.435, -0.12, 0.4145])
    lift = np.array([0.44, -0.12, 0.545])
    transit1 = np.array([0.54, -0.04, 0.56])
    transit2 = np.array([0.60, 0.04, 0.54])
    align = np.array([0.62, BOARD_CENTER[1], BOARD_CENTER[2]])
    insert_a = np.array([0.645, BOARD_CENTER[1], BOARD_CENTER[2]])
    insert_b = np.array([0.662, BOARD_CENTER[1], BOARD_CENTER[2]])

    ctl.set_gripper(255)
    ctl.move_to(pre_grasp, HAND_Z_DOWN, grip=255, steps=280)
    ctl.move_to(grasp, HAND_Z_DOWN, grip=255, steps=220)
    ctl.hold(40, grip=90)

    ctl.hold(220, grip=0)
    ctl.move_to(lift, HAND_Z_DOWN, grip=0, steps=240)
    ctl.move_to(transit1, HAND_Z_DOWN, grip=0, steps=220)
    ctl.move_to(transit2, HAND_Z_DOWN, grip=0, steps=220)
    ctl.move_to(align, HAND_Z_DOWN, grip=0, steps=260)
    ctl.move_to(insert_a, HAND_Z_DOWN, grip=0, steps=220, pos_gain=7.0)
    ctl.move_to(insert_b, HAND_Z_DOWN, grip=0, steps=260, pos_gain=6.0)
    ctl.hold(120, grip=0)

    # Save the first potentially passing witness and stop before the old retreat.
    sim.save_final_state("/work/final_state.npz")


def summarize(sim):
    peg_pos = sim.peg_position()
    peg_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
    peg_rot = sim.data.xmat[peg_id].reshape(3, 3)
    print("time", sim.data.time)
    print("peg_pos", peg_pos)
    print("peg_x_axis", peg_rot[:, 0])


if __name__ == "__main__":
    sim = Sim()
    attempt(sim)
    summarize(sim)
    sim.save_final_state("/work/final_state.npz")
