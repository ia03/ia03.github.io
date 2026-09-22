import numpy as np
import mujoco

from sim import Sim


PEG_TARGET_Y = -0.04
PEG_TARGET_Z = 0.48


def lerp(a, b, t):
    return (1.0 - t) * a + t * b


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


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
        self.best_proxy = -1e9

    def grip_center(self, data=None):
        data = self.d if data is None else data
        return 0.5 * (data.xpos[self.lf_id] + data.xpos[self.rf_id])

    def hand_rot(self, data=None):
        data = self.d if data is None else data
        return data.xmat[self.hand_id].reshape(3, 3).copy()

    def peg_pos(self):
        return self.sim.peg_position().copy()

    def peg_rot(self):
        return self.sim.data.xmat[self.sim.peg_body_id].reshape(3, 3).copy()

    def solve_q_for_point(self, target_p, q_init=None, q_ref=None, iters=140):
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
            if np.linalg.norm(ep) < 5e-5:
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
            q += 0.45 * dq
            q = np.clip(q, self.m.actuator_ctrlrange[:7, 0], self.m.actuator_ctrlrange[:7, 1])
        return q

    def hold(self, q_target, grip, steps, arm_alpha=0.06):
        for _ in range(steps):
            cur = self.d.ctrl[:7].copy()
            self.d.ctrl[:7] = cur + arm_alpha * (q_target - cur)
            self.d.ctrl[7] = grip
            self.sim.step(1)

    def move_grip_center(self, target_p, grip, steps, arm_alpha=0.06):
        q_target = self.solve_q_for_point(target_p)
        self.last_q = q_target.copy()
        self.hold(q_target, grip, steps, arm_alpha=arm_alpha)
        self.save_if_better(f"move_{np.round(target_p, 3).tolist()}")
        return q_target

    def save_if_better(self, label):
        peg = self.peg_pos()
        axis = self.peg_rot()[:, 0]
        insertion_progress = clamp01((peg[0] - 0.47) / (0.695 - 0.47))
        y_progress = clamp01(1.0 - abs(peg[1] - PEG_TARGET_Y) / 0.02)
        z_progress = clamp01(1.0 - abs(peg[2] - PEG_TARGET_Z) / 0.02)
        alignment_progress = clamp01(abs(axis[0]) / 0.85)
        proxy = 0.5 * insertion_progress + 6.0 * y_progress + 2.0 * z_progress + 0.2 * alignment_progress
        if proxy > self.best_proxy + 1e-9:
            self.best_proxy = proxy
            self.sim.save_final_state("/work/final_state.npz")
            print("save", label, "peg", np.round(peg, 4), "axis", np.round(axis, 4), "proxy", round(proxy, 4))

    def print_status(self, label):
        hand_p = self.grip_center()
        peg_p = self.peg_pos()
        hand_x = self.hand_rot()[:, 0]
        print(label, "hand", np.round(hand_p, 4), "peg", np.round(peg_p, 4), "hand_x", np.round(hand_x, 4))

    def dense_path(self, waypoints, grip, steps, arm_alpha=0.03):
        current = self.grip_center()
        for target in waypoints:
            for alpha in np.linspace(0.2, 1.0, 5):
                intermediate = lerp(current, target, float(alpha))
                self.move_grip_center(intermediate, grip=grip, steps=steps, arm_alpha=arm_alpha)
            current = target.copy()

    def run(self):
        self.d.ctrl[:7] = 0.0
        self.d.ctrl[7] = 255.0
        self.sim.step(320)
        self.print_status("settled")
        self.save_if_better("settled")

        self.move_grip_center(np.array([0.47, -0.12, 0.50]), grip=255.0, steps=240, arm_alpha=0.06)
        self.print_status("hover")
        self.move_grip_center(np.array([0.47, -0.12, 0.418]), grip=255.0, steps=260, arm_alpha=0.05)
        self.print_status("pregrasp")
        self.hold(self.last_q, grip=0.0, steps=260, arm_alpha=0.05)
        self.save_if_better("closed")
        self.print_status("closed")
        self.move_grip_center(np.array([0.49, -0.11, 0.57]), grip=0.0, steps=340, arm_alpha=0.04)
        self.print_status("lifted")
        self.hold(self.last_q, grip=0.0, steps=220, arm_alpha=0.03)
        self.save_if_better("stabilized")
        self.print_status("stabilized")

        phase1 = [
            np.array([0.54, -0.11, 0.58]),
            np.array([0.58, -0.10, 0.58]),
            np.array([0.62, -0.09, 0.57]),
            np.array([0.66, -0.08, 0.56]),
            np.array([0.68, -0.07, 0.55]),
            np.array([0.695, -0.06, 0.54]),
            np.array([0.705, -0.05, 0.52]),
        ]
        self.dense_path(phase1, grip=0.0, steps=150, arm_alpha=0.02)
        self.save_if_better("phase1")
        self.print_status("phase1")

        phase2 = [
            np.array([0.656, -0.080, 0.457]),
            np.array([0.656, -0.070, 0.457]),
            np.array([0.657, -0.060, 0.458]),
            np.array([0.660, -0.050, 0.460]),
            np.array([0.670, PEG_TARGET_Y, 0.465]),
            np.array([0.690, PEG_TARGET_Y, 0.472]),
            np.array([0.705, PEG_TARGET_Y, PEG_TARGET_Z]),
        ]
        self.dense_path(phase2, grip=0.0, steps=140, arm_alpha=0.018)
        self.save_if_better("phase2")
        self.print_status("phase2")

        # Final tiny nudge attempts after the best mid-state has already been captured.
        finale = [
            np.array([0.680, -0.050, 0.468]),
            np.array([0.695, -0.045, 0.476]),
            np.array([0.705, PEG_TARGET_Y, PEG_TARGET_Z]),
        ]
        self.dense_path(finale, grip=0.0, steps=120, arm_alpha=0.015)
        self.save_if_better("finale")
        self.print_status("finale")

        y_finish = [
            np.array([0.665, -0.080, 0.468]),
            np.array([0.665, -0.070, 0.468]),
            np.array([0.665, -0.060, 0.468]),
            np.array([0.665, -0.050, 0.468]),
            np.array([0.665, -0.045, 0.468]),
            np.array([0.665, -0.042, 0.469]),
            np.array([0.665, PEG_TARGET_Y, 0.470]),
        ]
        self.dense_path(y_finish, grip=0.0, steps=160, arm_alpha=0.010)
        self.save_if_better("y_finish")
        self.print_status("y_finish")


if __name__ == "__main__":
    Controller().run()
