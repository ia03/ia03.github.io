import numpy as np
import mujoco

from sim import Sim


PEG_TARGET_Y = 0.08
PEG_TARGET_Z = 0.48


class Controller:
    def __init__(self):
        self.sim = Sim()
        self.m = self.sim.model
        self.d = self.sim.data
        self.hand_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.lf_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.rf_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.q_ref = np.array(
            [-0.22547315, -0.12656390, 0.02426566, -1.93103586, -0.13569297, 1.58618460, 0.90402641],
            dtype=float,
        )
        self.last_q = self.q_ref.copy()

    def grip_center(self, data=None):
        data = self.d if data is None else data
        return 0.5 * (data.xpos[self.lf_id] + data.xpos[self.rf_id])

    def hand_rot(self, data=None):
        data = self.d if data is None else data
        return data.xmat[self.hand_id].reshape(3, 3).copy()

    def peg_pos(self):
        return self.sim.peg_position().copy()

    def solve_q_for_point(self, target_p, q_init=None, q_ref=None, iters=100):
        q = self.last_q.copy() if q_init is None else q_init.copy()
        q_ref = self.q_ref if q_ref is None else q_ref
        base_qpos = self.sim._initial_qpos.copy()
        tmp = mujoco.MjData(self.m)
        for _ in range(iters):
            tmp.qpos[:] = base_qpos
            tmp.qpos[:7] = q
            tmp.qpos[7:9] = 0.04
            mujoco.mj_kinematics(self.m, tmp)
            mujoco.mj_comPos(self.m, tmp)
            p = self.grip_center(tmp)
            ep = target_p - p
            if np.linalg.norm(ep) < 1e-4:
                break
            jp1 = np.zeros((3, self.m.nv))
            jr1 = np.zeros((3, self.m.nv))
            jp2 = np.zeros((3, self.m.nv))
            jr2 = np.zeros((3, self.m.nv))
            mujoco.mj_jacBody(self.m, tmp, jp1, jr1, self.lf_id)
            mujoco.mj_jacBody(self.m, tmp, jp2, jr2, self.rf_id)
            J = 0.5 * (jp1[:, :7] + jp2[:, :7])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), ep)
            dq += 0.08 * (q_ref - q)
            q += 0.5 * dq
            q = np.clip(q, self.m.actuator_ctrlrange[:7, 0], self.m.actuator_ctrlrange[:7, 1])
        return q

    def hold(self, q_target, grip, steps, arm_alpha=0.08):
        for _ in range(steps):
            cur = self.d.ctrl[:7].copy()
            self.d.ctrl[:7] = cur + arm_alpha * (q_target - cur)
            self.d.ctrl[7] = grip
            self.sim.step(1)

    def move_grip_center(self, target_p, grip, steps, arm_alpha=0.08):
        q_target = self.solve_q_for_point(target_p)
        self.last_q = q_target.copy()
        self.hold(q_target, grip, steps, arm_alpha=arm_alpha)
        return q_target

    def print_status(self, label):
        hand_p = self.grip_center()
        peg_p = self.peg_pos()
        hand_x = self.hand_rot()[:, 0]
        print(label, "hand", np.round(hand_p, 4), "peg", np.round(peg_p, 4), "hand_x", np.round(hand_x, 4))

    def run(self):
        self.d.ctrl[:7] = 0.0
        self.d.ctrl[7] = 255.0
        self.sim.step(300)
        self.print_status("settled")

        self.move_grip_center(np.array([0.47, -0.12, 0.50]), grip=255.0, steps=240, arm_alpha=0.06)
        self.print_status("hover")
        self.move_grip_center(np.array([0.47, -0.12, 0.435]), grip=255.0, steps=220, arm_alpha=0.05)
        self.print_status("pregrasp")
        self.hold(self.last_q, grip=0.0, steps=240, arm_alpha=0.05)
        self.print_status("closed")
        self.move_grip_center(np.array([0.49, -0.11, 0.57]), grip=0.0, steps=320, arm_alpha=0.04)
        self.print_status("lifted")
        self.move_grip_center(np.array([0.58, -0.02, 0.56]), grip=0.0, steps=260, arm_alpha=0.04)
        self.print_status("midway")
        self.move_grip_center(np.array([0.66, 0.04, 0.56]), grip=0.0, steps=300, arm_alpha=0.035)
        self.print_status("pre_align")
        self.move_grip_center(np.array([0.70, PEG_TARGET_Y, 0.53]), grip=0.0, steps=320, arm_alpha=0.03)
        self.print_status("approach")
        self.move_grip_center(np.array([0.76, PEG_TARGET_Y, 0.53]), grip=0.0, steps=360, arm_alpha=0.025)
        self.print_status("insert")
        self.hold(self.last_q, grip=0.0, steps=300, arm_alpha=0.02)
        self.print_status("settle")
        self.sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    Controller().run()
