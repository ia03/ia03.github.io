import numpy as np
import mujoco

from sim import Sim


PEG_INIT_POS = np.array([0.47, -0.12, 0.412], dtype=float)
BOARD_CENTER = np.array([0.70, 0.08, 0.48], dtype=float)
IDEAL_REPLAY_STEPS = 1200


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def peg_inserted(pos):
    x_ok = float(pos[0]) >= 0.695
    y_ok = abs(float(pos[1]) - 0.08) <= 0.02
    z_ok = abs(float(pos[2]) - 0.48) <= 0.02
    return x_ok and y_ok and z_ok


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

    def score_state(self):
        peg_p = self.peg_pos()
        peg_x_axis = self.sim.data.xmat[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "peg")].reshape(3, 3)[:, 0]
        aligned = abs(float(peg_x_axis[0])) >= 0.85
        inserted = peg_inserted(peg_p)
        insertion_progress = clamp01((peg_p[0] - PEG_INIT_POS[0]) / ((BOARD_CENTER[0] - 0.005) - PEG_INIT_POS[0]))
        y_progress = clamp01(1.0 - abs(float(peg_p[1]) - BOARD_CENTER[1]) / 0.02)
        z_progress = clamp01(1.0 - abs(float(peg_p[2]) - BOARD_CENTER[2]) / 0.02)
        alignment_progress = clamp01(abs(float(peg_x_axis[0])) / 0.85)
        efficiency_progress = clamp01(IDEAL_REPLAY_STEPS / max(len(self.sim._ctrl_trace), 1))
        score = (
            0.35 * insertion_progress
            + 0.15 * y_progress
            + 0.15 * z_progress
            + 0.20 * alignment_progress
            + 0.15 * efficiency_progress
        )
        return score, {
            "peg_xyz_after": [float(x) for x in peg_p],
            "peg_x_axis": [float(x) for x in peg_x_axis],
            "inserted": inserted,
            "aligned": aligned,
            "insertion_progress": insertion_progress,
            "y_progress": y_progress,
            "z_progress": z_progress,
            "alignment_progress": alignment_progress,
            "efficiency_progress": efficiency_progress,
            "progress_score": score,
        }

    def print_status(self, label):
        hand_p = self.grip_center()
        peg_p = self.peg_pos()
        hand_x = self.hand_rot()[:, 0]
        print(label, "hand", np.round(hand_p, 4), "peg", np.round(peg_p, 4), "hand_x", np.round(hand_x, 4))

    def run_plan(self, waypoints, final_grip=0.0):
        self.d.ctrl[:7] = 0.0
        self.d.ctrl[7] = 255.0
        self.sim.step(300)
        self.print_status("settled")

        self.move_grip_center(np.array([0.47, -0.12, 0.50]), grip=255.0, steps=220, arm_alpha=0.06)
        self.print_status("hover")
        self.move_grip_center(np.array([0.47, -0.12, 0.435]), grip=255.0, steps=220, arm_alpha=0.05)
        self.print_status("pregrasp")
        self.hold(self.last_q, grip=0.0, steps=220, arm_alpha=0.05)
        self.print_status("closed")
        self.move_grip_center(np.array([0.49, -0.11, 0.57]), grip=0.0, steps=320, arm_alpha=0.04)
        self.print_status("lifted")
        for idx, waypoint in enumerate(waypoints):
            self.move_grip_center(np.array(waypoint), grip=0.0, steps=260, arm_alpha=0.03 if idx < len(waypoints) - 1 else 0.025)
            self.print_status(f"wp{idx}")
        self.hold(self.last_q, grip=final_grip, steps=360, arm_alpha=0.02)
        self.print_status("settle")
        score, info = self.score_state()
        self.sim.save_final_state("/work/final_state.npz")
        return score, info

    def run(self):
        plans = [
            [
                (0.58, -0.02, 0.60),
                (0.66, 0.04, 0.60),
                (0.72, 0.08, 0.60),
                (0.80, 0.08, 0.60),
            ],
            [
                (0.58, -0.02, 0.59),
                (0.66, 0.04, 0.60),
                (0.74, 0.08, 0.60),
                (0.82, 0.08, 0.60),
            ],
            [
                (0.56, -0.04, 0.60),
                (0.64, 0.02, 0.60),
                (0.72, 0.08, 0.60),
                (0.84, 0.08, 0.60),
            ],
        ]
        best = None
        best_info = None
        for idx, waypoints in enumerate(plans):
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
            score, info = self.run_plan(waypoints)
            print("plan", idx, "score", round(score, 4), info, flush=True)
            if best is None or score > best:
                best = score
                best_info = info
                self.sim.save_final_state("/work/final_state.npz")
        print("best", best, best_info, flush=True)


if __name__ == "__main__":
    Controller().run()
