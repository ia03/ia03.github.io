import numpy as np
import mujoco

from sim import Sim


PEG_TARGET_Y = -0.10
PEG_TARGET_Z = 0.48
PEG_TARGET_X = 0.565
PEG_MIN_X = 0.55


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
        q_ref = q.copy() if q_ref is None else q_ref
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
        insertion_progress = clamp01((peg[0] - 0.505) / (PEG_MIN_X - 0.505))
        y_progress = clamp01(1.0 - abs(peg[1] - PEG_TARGET_Y) / 0.025)
        z_progress = clamp01(1.0 - abs(peg[2] - PEG_TARGET_Z) / 0.020)
        alignment_progress = clamp01(abs(axis[0]) / 0.45)
        proxy = 0.45 * insertion_progress + 0.15 * y_progress + 0.15 * z_progress + 0.20 * alignment_progress
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

    def reset_trial(self):
        self.sim.reset()
        self.last_q = self.q_ref.copy()
        self.d.ctrl[:7] = 0.0
        self.d.ctrl[7] = 255.0
        self.sim.step(320)
        self.save_if_better("settled")

    def is_success_like(self):
        peg = self.peg_pos()
        axis = self.peg_rot()[:, 0]
        return (
            peg[0] >= PEG_MIN_X
            and abs(peg[1] - PEG_TARGET_Y) <= 0.025
            and abs(peg[2] - PEG_TARGET_Z) <= 0.020
            and abs(axis[0]) >= 0.45
        )

    def run_candidate(self, contact_y, contact_z, grip, push_z, push_end_x):
        self.reset_trial()
        self.move_grip_center(np.array([0.455, contact_y, push_z + 0.035]), grip=255.0, steps=220, arm_alpha=0.06)
        self.move_grip_center(np.array([0.458, contact_y, contact_z]), grip=255.0, steps=240, arm_alpha=0.05)
        self.hold(self.last_q, grip=grip, steps=160, arm_alpha=0.05)
        self.save_if_better(f"captured_{contact_y:.3f}_{contact_z:.3f}_{grip:.1f}")
        path = [
            np.array([0.475, contact_y, push_z]),
            np.array([0.500, -0.101, push_z]),
            np.array([0.528, -0.100, push_z]),
            np.array([0.552, -0.100, push_z]),
            np.array([push_end_x, -0.100, push_z]),
        ]
        self.dense_path(path, grip=grip, steps=180, arm_alpha=0.05, reg=0.01)
        self.hold(self.last_q, grip=grip, steps=900, arm_alpha=0.03)
        self.save_if_better(f"end_{contact_y:.3f}_{contact_z:.3f}_{grip:.1f}_{push_end_x:.3f}")
        return self.is_success_like()

    def run(self):
        self.d.ctrl[:7] = 0.0
        self.d.ctrl[7] = 255.0
        search = [
            (-0.104, 0.480, 20.0, 0.480, 0.575),
            (-0.102, 0.480, 20.0, 0.480, 0.585),
            (-0.100, 0.480, 20.0, 0.480, 0.590),
            (-0.102, 0.482, 10.0, 0.482, 0.585),
            (-0.100, 0.482, 10.0, 0.482, 0.590),
            (-0.102, 0.478, 30.0, 0.478, 0.585),
        ]
        for params in search:
            if self.run_candidate(*params):
                print("success params", params)
                return


if __name__ == "__main__":
    Controller().run()
