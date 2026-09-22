import numpy as np
import mujoco
from sim import Sim


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


class Controller:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.m = sim.model
        self.d = sim.data
        self.lf = body_id(self.m, "left_finger")
        self.rf = body_id(self.m, "right_finger")
        self.hand = body_id(self.m, "hand")
        self.cup = body_id(self.m, "cup")
        self.ctrl_min = self.m.actuator_ctrlrange[:7, 0]
        self.ctrl_max = self.m.actuator_ctrlrange[:7, 1]

    def pinch_pos(self):
        return 0.5 * (self.d.xpos[self.lf] + self.d.xpos[self.rf])

    def cup_pos(self):
        return np.array(self.d.xpos[self.cup])

    def step_to(self, target_pos, grip, n_steps, pos_gain=3.5, rot_gain=0.8, damp=2e-3):
        target_z = np.array([0.0, 0.0, -1.0])
        for _ in range(n_steps):
            mujoco.mj_forward(self.m, self.d)

            pos = self.pinch_pos()
            pos_err = target_pos - pos

            jacp_l = np.zeros((3, self.m.nv))
            jacr_l = np.zeros((3, self.m.nv))
            jacp_r = np.zeros((3, self.m.nv))
            jacr_r = np.zeros((3, self.m.nv))
            jacp_h = np.zeros((3, self.m.nv))
            jacr_h = np.zeros((3, self.m.nv))
            mujoco.mj_jacBodyCom(self.m, self.d, jacp_l, jacr_l, self.lf)
            mujoco.mj_jacBodyCom(self.m, self.d, jacp_r, jacr_r, self.rf)
            mujoco.mj_jacBodyCom(self.m, self.d, jacp_h, jacr_h, self.hand)

            jacp = 0.5 * (jacp_l[:, :7] + jacp_r[:, :7])
            jacr = jacr_h[:, :7]

            R = self.d.xmat[self.hand].reshape(3, 3)
            hand_z = R[:, 2]
            rot_err = np.cross(hand_z, target_z)

            A = jacp.T @ jacp + (rot_gain * jacr.T) @ jacr + damp * np.eye(7)
            b = jacp.T @ (pos_gain * pos_err) + jacr.T @ (rot_gain * rot_err)
            dq = np.linalg.solve(A, b)

            q = self.d.qpos[:7].copy()
            q_cmd = np.clip(q + dq, self.ctrl_min, self.ctrl_max)
            self.d.ctrl[:7] = q_cmd
            self.d.ctrl[7] = float(grip)
            self.sim.step(1)


def run():
    sim = Sim()
    ctrl = Controller(sim)

    cup0 = ctrl.cup_pos()

    # 1) open and go above cup
    ctrl.step_to(cup0 + np.array([0.0, 0.0, 0.12]), grip=255, n_steps=900)

    # 2) descend to grasp height
    ctrl.step_to(cup0 + np.array([0.0, 0.0, 0.015]), grip=255, n_steps=700)

    # 3) close gripper
    ctrl.step_to(ctrl.pinch_pos(), grip=0, n_steps=700)

    # 4) lift
    cp = ctrl.cup_pos()
    ctrl.step_to(np.array([cp[0], cp[1], 0.57]), grip=0, n_steps=900)

    # 5) move above bin
    ctrl.step_to(np.array([0.70, 0.14, 0.58]), grip=0, n_steps=1000)

    # 6) lower into bin
    ctrl.step_to(np.array([0.70, 0.14, 0.49]), grip=0, n_steps=700)

    # 7) release and move up
    ctrl.step_to(np.array([0.70, 0.14, 0.54]), grip=255, n_steps=900)

    # 8) retreat slightly and settle
    ctrl.step_to(np.array([0.62, 0.14, 0.58]), grip=255, n_steps=700)
    sim.step(1200)

    sim.save_final_state('/work/final_state.npz')

    cupf = ctrl.cup_pos()
    print('saved /work/final_state.npz')
    print('final cup', cupf)
    print('cup contact now', sim.has_gripper_cup_contact())
    print('trace len', len(sim._ctrl_trace))


if __name__ == '__main__':
    run()
