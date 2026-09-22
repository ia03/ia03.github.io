import numpy as np
import mujoco
from dataclasses import dataclass

from sim import Sim


GRASP_OFFSET_LOCAL = np.array([0.0, 0.0, 0.0584], dtype=float)
TARGET_R = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)
SIDE_R = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)


@dataclass
class AttemptResult:
    name: str
    score: float
    peg_pos: np.ndarray
    align: float
    inserted: bool


class Controller:
    def __init__(self):
        self.sim = Sim()
        self.model = self.sim.model
        self.data = self.sim.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.peg_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.qhome = self.data.qpos[:7].copy()
        self.best = None
        self.attempt_summaries = []
        self.save_best("initial")

    def save_best(self, name: str):
        result = self.evaluate(name)
        if self.best is None or result.score > self.best.score:
            self.best = result
            self.save_state_with_debug("/work/final_state.npz")
            print(
                f"saved {name}: score={result.score:.4f} peg={np.round(result.peg_pos,4)} "
                f"align={result.align:.3f} inserted={result.inserted}",
                flush=True,
            )

    def save_state_with_debug(self, path):
        payload = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "ctrl_trace": np.array(self.sim._ctrl_trace, dtype=float).reshape(-1, self.model.nu),
            "initial_qpos": self.sim._initial_qpos.copy(),
            "peg_pos": self.sim.peg_position().copy(),
            "time": float(self.data.time),
            "attempt_summaries": np.array(self.attempt_summaries, dtype=object),
        }
        self.sim.save_final_state(path)
        np.savez(path, **payload)

    def evaluate(self, name: str) -> AttemptResult:
        peg_pos = self.sim.peg_position().copy()
        peg_xmat = self.data.xmat[self.peg_id].reshape(3, 3)
        align = abs(float(peg_xmat[0, 0]))
        dx = max(0.0, 0.695 - peg_pos[0])
        dy = max(0.0, abs(peg_pos[1] - 0.08) - 0.02)
        dz = max(0.0, abs(peg_pos[2] - 0.48) - 0.02)
        da = max(0.0, 0.85 - align)
        score = -(4.0 * dx + 2.0 * dy + 2.0 * dz + da)
        inserted = dx == 0.0 and dy == 0.0 and dz == 0.0 and da == 0.0
        if inserted:
            score += 10.0
        return AttemptResult(name, score, peg_pos, align, inserted)

    def hand_pose(self):
        pos = self.data.xpos[self.hand_id].copy()
        rot = self.data.xmat[self.hand_id].reshape(3, 3).copy()
        return pos, rot

    def set_gripper(self, value: float):
        self.data.ctrl[7] = np.clip(value, self.model.actuator_ctrlrange[7, 0], self.model.actuator_ctrlrange[7, 1])

    def hold(self, steps: int, gripper: float | None = None):
        if gripper is not None:
            self.set_gripper(gripper)
        for _ in range(steps):
            self.data.ctrl[:7] = np.clip(self.data.ctrl[:7], self.model.actuator_ctrlrange[:7, 0], self.model.actuator_ctrlrange[:7, 1])
            self.sim.step(1)

    def solve_ik(self, target_pos, target_quat, iters=120, pos_gain=1.0, rot_gain=0.35, damping=5e-3):
        q_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()
        ctrl_backup = self.data.ctrl.copy()
        work = self.data
        q = work.qpos[:7].copy()
        for _ in range(iters):
            work.qpos[:] = q_backup
            work.qvel[:] = 0
            work.qpos[:7] = q
            mujoco.mj_forward(self.model, work)
            pos, quat = self._hand_pose_from_data(work)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, work, jacp, jacr, self.hand_id)
            J = np.vstack([pos_gain * jacp[:, :7], rot_gain * jacr[:, :7]])
            err = np.concatenate([
                pos_gain * (target_pos - pos),
                rot_gain * self.quat_err(quat, target_quat),
            ])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])
            if np.linalg.norm(err[:3]) < 2e-3 and np.linalg.norm(err[3:]) < 2e-2:
                break
        work.qpos[:] = q_backup
        work.qvel[:] = qvel_backup
        work.ctrl[:] = ctrl_backup
        mujoco.mj_forward(self.model, work)
        return q

    def quat_err(self, current, target):
        dq = self.quat_mul(target, self.quat_conj(current))
        if dq[0] < 0:
            dq = -dq
        return 2.0 * dq[1:]

    def quat_conj(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quat_mul(self, a, b):
        return np.array([
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ])

    def quat_from_mat(self, mat):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.reshape(-1))
        if quat[0] < 0:
            quat = -quat
        return quat

    def _hand_pose_from_data(self, data):
        pos = data.xpos[self.hand_id].copy()
        quat = self.quat_from_mat(data.xmat[self.hand_id].reshape(3, 3))
        return pos, quat

    def ik_to_pose(self, pos, quat, steps=220, gripper=None, pos_gain=2.5, rot_gain=1.2):
        qmin = self.model.actuator_ctrlrange[:7, 0]
        qmax = self.model.actuator_ctrlrange[:7, 1]
        for _ in range(steps):
            hand_pos, hand_rot = self.hand_pose()
            pos_err = pos - hand_pos
            quat_c = self.quat_from_mat(hand_rot)
            quat_t = quat
            qerr = self.quat_mul(quat_t, self.quat_conj(quat_c))
            if qerr[0] < 0:
                qerr *= -1
            rot_err = np.zeros(3)
            mujoco.mju_quat2Vel(rot_err, qerr, 1.0)
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_id)
            J = np.vstack([pos_gain * jacp[:, :7], rot_gain * jacr[:, :7]])
            err = np.concatenate([pos_gain * pos_err, rot_gain * rot_err])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = np.clip(self.data.qpos[:7] + dq, qmin, qmax)
            self.data.ctrl[:7] = q
            if gripper is not None:
                self.set_gripper(gripper)
            self.sim.step(1)
            if np.linalg.norm(pos_err) < 0.003 and np.linalg.norm(rot_err) < 0.02:
                break

    def reorient_keeping_peg(self, target_R, steps=220, gripper=None, pos_gain=2.5, rot_gain=0.25, offset_scale=1.0):
        hand_pos, hand_rot = self.hand_pose()
        peg_pos = self.sim.peg_position().copy()
        offset_local = hand_rot.T @ (peg_pos - hand_pos)
        target_pos = peg_pos - offset_scale * (target_R @ offset_local)
        self.ik_to_pose(target_pos, self.quat_from_mat(target_R), steps=steps, gripper=gripper, pos_gain=pos_gain, rot_gain=rot_gain)

    def run_attempt(self, name: str, grasp_z: float, lift_z: float, pre_slot_z: float, insert_x: float, push_x: float, settle: int = 240, side_scale: float = 1.0):
        top_quat = self.quat_from_mat(TARGET_R)
        peg = self.sim.peg_position()
        grasp = peg + np.array([0.0, 0.0, grasp_z])
        lift = np.array([peg[0] + 0.02, -0.03, lift_z])
        pre_slot = np.array([0.58, 0.08, pre_slot_z])
        slot = np.array([0.70, 0.08, pre_slot_z])
        push = np.array([push_x, 0.08, pre_slot_z])

        self.sim.reset()
        self.data.ctrl[:7] = self.qhome
        self.data.ctrl[7] = 255
        self.sim.step(20)

        self.ik_to_pose(np.array([0.50, -0.12, 0.60]), top_quat, steps=220, gripper=255, pos_gain=3.0, rot_gain=0.08)
        self.ik_to_pose(np.array([0.47, -0.12, 0.435]), top_quat, steps=220, gripper=255, pos_gain=3.0, rot_gain=0.08)
        self.reorient_keeping_peg(SIDE_R, steps=220, gripper=255, pos_gain=2.5, rot_gain=0.18, offset_scale=0.95)
        self.hold(220, gripper=0)
        self.save_best(name + "_after_grasp")
        self.ik_to_pose(lift, SIDE_R, steps=280, gripper=0, pos_gain=2.5, rot_gain=0.10)
        self.save_best(name + "_lift")
        self.reorient_keeping_peg(SIDE_R, steps=260, gripper=0, pos_gain=2.5, rot_gain=0.18, offset_scale=side_scale)
        self.save_best(name + "_side")
        self.ik_to_pose(pre_slot, SIDE_R, steps=320, gripper=0, pos_gain=2.5, rot_gain=0.20)
        self.ik_to_pose(slot, SIDE_R, steps=260, gripper=0, pos_gain=2.5, rot_gain=0.25)
        self.ik_to_pose(push, SIDE_R, steps=280, gripper=0, pos_gain=2.5, rot_gain=0.25)
        self.hold(settle, gripper=0)
        self.attempt_summaries.append(
            {
                "name": name,
                "grasp_z": grasp_z,
                "lift_z": lift_z,
                "pre_slot_z": pre_slot_z,
                "insert_x": insert_x,
                "push_x": push_x,
                "side_scale": side_scale,
                "final_peg_z": float(self.sim.peg_position()[2]),
                "final_peg_x": float(self.sim.peg_position()[0]),
                "final_peg_y": float(self.sim.peg_position()[1]),
                "score": float(self.evaluate(name).score),
            }
        )
        self.save_best(name)
        return self.best


def main():
    attempts = [
        ("attempt1", 0.016, 0.58, 0.48, 0.705, 0.72, 240, 0.8),
        ("attempt2", 0.018, 0.62, 0.52, 0.725, 0.76, 260, 0.9),
        ("attempt3", 0.020, 0.66, 0.56, 0.740, 0.80, 260, 1.0),
        ("attempt4", 0.022, 0.68, 0.58, 0.760, 0.84, 280, 1.1),
        ("attempt5", 0.020, 0.70, 0.60, 0.780, 0.86, 280, 1.2),
        ("attempt6", 0.018, 0.64, 0.54, 0.750, 0.82, 300, 0.95),
    ]
    controller = Controller()
    best = None
    for params in attempts:
        result = controller.run_attempt(*params)
        print(result)
        if best is None or result.score > best.score:
            best = result
    print("best", best, flush=True)


if __name__ == "__main__":
    main()
