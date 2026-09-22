import numpy as np
import mujoco

from sim import Sim


PEG_TARGET_Y = -0.10
PEG_TARGET_Z = 0.48
PEG_TARGET_X = 0.585


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
        self.insert_q_ref = np.array([0.0, -0.6, 0.0, -1.8, 0.0, 1.8, 0.8], dtype=float)
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

    def solve_q_for_point(self, target_p, q_init=None, q_ref=None, iters=180, reg=0.08, desired_x=None, desired_z=None):
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
            R = self.hand_rot(tmp)
            ep = target_p - p
            er_terms = []
            if desired_x is not None:
                er_terms.append(np.cross(R[:, 0], desired_x))
            if desired_z is not None:
                er_terms.append(np.cross(R[:, 2], desired_z))
            er = np.sum(er_terms, axis=0) if er_terms else np.zeros(3)
            if np.linalg.norm(ep) < 5e-5 and np.linalg.norm(er) < 1e-4:
                break
            jp1 = np.zeros((3, self.m.nv))
            jr1 = np.zeros((3, self.m.nv))
            jp2 = np.zeros((3, self.m.nv))
            jr2 = np.zeros((3, self.m.nv))
            mujoco.mj_jacBody(self.m, tmp, jp1, jr1, self.lf_id)
            mujoco.mj_jacBody(self.m, tmp, jp2, jr2, self.rf_id)
            Jp = 0.5 * (jp1[:, :7] + jp2[:, :7])
            jr_hand = np.zeros((3, self.m.nv))
            jp_hand = np.zeros((3, self.m.nv))
            mujoco.mj_jacBody(self.m, tmp, jp_hand, jr_hand, self.hand_id)
            if desired_x is not None or desired_z is not None:
                J = np.vstack([Jp, 0.20 * jr_hand[:, :7]])
                e = np.concatenate([ep, 0.20 * er])
            else:
                J = Jp
                e = ep
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(J.shape[0]), e)
            dq += reg * (q_ref - q)
            q += 0.45 * dq
            q = np.clip(q, self.m.actuator_ctrlrange[:7, 0], self.m.actuator_ctrlrange[:7, 1])
        return q

    def hold(self, q_target, grip, steps, arm_alpha=0.06):
        for _ in range(steps):
            cur = self.d.ctrl[:7].copy()
            self.d.ctrl[:7] = cur + arm_alpha * (q_target - cur)
            self.d.ctrl[7] = grip
            self.sim.step(1)

    def move_grip_center(self, target_p, grip, steps, arm_alpha=0.06, q_ref=None, reg=0.08, desired_x=None, desired_z=None):
        q_target = self.solve_q_for_point(target_p, q_ref=q_ref, reg=reg, desired_x=desired_x, desired_z=desired_z)
        self.last_q = q_target.copy()
        self.hold(q_target, grip, steps, arm_alpha=arm_alpha)
        self.save_if_better(f"move_{np.round(target_p, 3).tolist()}")
        return q_target

    def save_if_better(self, label):
        peg = self.peg_pos()
        axis = self.peg_rot()[:, 0]
        insertion_progress = clamp01((peg[0] - 0.47) / (0.57 - 0.47))
        y_progress = clamp01(1.0 - abs(peg[1] - PEG_TARGET_Y) / 0.03)
        z_progress = clamp01(1.0 - abs(peg[2] - PEG_TARGET_Z) / 0.025)
        alignment_progress = clamp01(abs(axis[0]) / 0.85)
        proxy = 5.0 * insertion_progress + 3.0 * y_progress + 1.5 * z_progress + 0.4 * alignment_progress
        if proxy > self.best_proxy + 1e-9:
            self.best_proxy = proxy
            self.sim.save_final_state("/work/final_state.npz")
            print("save", label, "peg", np.round(peg, 4), "axis", np.round(axis, 4), "proxy", round(proxy, 4))

    def print_status(self, label):
        hand_p = self.grip_center()
        peg_p = self.peg_pos()
        hand_x = self.hand_rot()[:, 0]
        print(label, "hand", np.round(hand_p, 4), "peg", np.round(peg_p, 4), "hand_x", np.round(hand_x, 4))

    def dense_path(self, waypoints, grip, steps, arm_alpha=0.03, q_ref=None, reg=0.08, desired_x=None, desired_z=None):
        current = self.grip_center()
        for target in waypoints:
            for alpha in np.linspace(0.2, 1.0, 5):
                intermediate = lerp(current, target, float(alpha))
                self.move_grip_center(intermediate, grip=grip, steps=steps, arm_alpha=arm_alpha, q_ref=q_ref, reg=reg, desired_x=desired_x, desired_z=desired_z)
            current = target.copy()

    def run(self):
        desired_x = np.array([1.0, 0.0, 0.0], dtype=float)
        desired_z = np.array([0.0, 0.0, -1.0], dtype=float)
        self.d.ctrl[:7] = 0.0
        self.d.ctrl[7] = 255.0
        self.sim.step(320)
        self.print_status("settled")
        self.save_if_better("settled")

        self.move_grip_center(np.array([0.47, -0.12, 0.50]), grip=255.0, steps=240, arm_alpha=0.06, desired_x=desired_x, desired_z=desired_z)
        self.print_status("hover")
        self.move_grip_center(np.array([0.47, -0.12, 0.418]), grip=255.0, steps=260, arm_alpha=0.05, desired_x=desired_x, desired_z=desired_z)
        self.print_status("pregrasp")
        self.hold(self.last_q, grip=0.0, steps=260, arm_alpha=0.05)
        self.save_if_better("closed")
        self.print_status("closed")
        self.move_grip_center(np.array([0.49, -0.11, 0.57]), grip=0.0, steps=340, arm_alpha=0.04, desired_x=desired_x, desired_z=desired_z)
        self.print_status("lifted")
        self.hold(self.last_q, grip=0.0, steps=220, arm_alpha=0.03)
        self.save_if_better("stabilized")
        self.print_status("stabilized")

        insertion_path = [
            np.array([0.53, -0.11, 0.58]),
            np.array([0.55, -0.105, 0.56]),
            np.array([0.565, -0.102, 0.535]),
            np.array([0.575, -0.100, 0.505]),
            np.array([0.585, PEG_TARGET_Y, 0.488]),
        ]
        self.dense_path(insertion_path, grip=0.0, steps=220, arm_alpha=0.018, desired_x=desired_x, desired_z=desired_z)
        self.save_if_better("insertion_path")
        self.print_status("insertion_path")

        settle_pose = np.array([0.595, PEG_TARGET_Y, 0.485])
        self.move_grip_center(settle_pose, grip=0.0, steps=480, arm_alpha=0.030, q_ref=self.insert_q_ref, reg=0.01, desired_x=desired_x, desired_z=desired_z)
        self.hold(self.last_q, grip=0.0, steps=1200, arm_alpha=0.025)
        press_pose = np.array([0.625, PEG_TARGET_Y, 0.484])
        self.move_grip_center(press_pose, grip=0.0, steps=520, arm_alpha=0.035, q_ref=self.insert_q_ref, reg=0.01, desired_x=desired_x, desired_z=desired_z)
        self.hold(self.last_q, grip=0.0, steps=1100, arm_alpha=0.030)
        self.save_if_better("settled_insert")
        self.print_status("settled_insert")


if __name__ == "__main__":
    Controller().run()
