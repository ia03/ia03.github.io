from pathlib import Path

import mujoco
import numpy as np

from sim import Sim


HAND_BODY = "hand"
SAVE_PATH = "/work/final_state.npz"


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_error(current, target):
    dq = quat_mul(target, quat_conj(current))
    if dq[0] < 0:
        dq = -dq
    return 2.0 * dq[1:]


class Planner:
    def __init__(self):
        self.sim = Sim()
        self.model = self.sim.model
        self.data = self.sim.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.q = self.home.copy()
        self.data.qpos[:7] = self.q
        self.data.qpos[7:9] = 0.04
        mujoco.mj_forward(self.model, self.data)
        self.hand_quat_target = self.data.xquat[self.hand_id].copy()

    def solve(self, pos, quat=None, q_init=None, max_iters=200):
        q = self.q.copy() if q_init is None else np.array(q_init, dtype=float).copy()
        quat = self.hand_quat_target if quat is None else np.array(quat, dtype=float)
        for _ in range(max_iters):
            self.data.qpos[:7] = q
            self.data.qpos[7:9] = 0.04
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            pos_err = np.array(pos) - self.data.xpos[self.hand_id]
            rot_err = quat_error(self.data.xquat[self.hand_id], quat)
            err = np.concatenate([pos_err, 0.2 * rot_err])
            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 1e-3:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_id)
            J = np.vstack([jacp[:, :7], 0.2 * jacr[:, :7]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = np.clip(
                q + 0.7 * dq,
                self.model.actuator_ctrlrange[:7, 0],
                self.model.actuator_ctrlrange[:7, 1],
            )
        self.q = q.copy()
        return q


class Executor:
    def __init__(self):
        self.sim = Sim()
        self.model = self.sim.model
        self.data = self.sim.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.data.ctrl[:7] = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.data.ctrl[7] = 255.0
        self.sim.step(1500)

    def cup_pos(self):
        return self.sim.cup_position().copy()

    def hand_pos(self):
        return self.data.xpos[self.hand_id].copy()

    def move_q(self, q_target, grip, chunks=20, steps_per_chunk=40):
        q_start = self.data.ctrl[:7].copy()
        for alpha in np.linspace(0.0, 1.0, chunks + 1)[1:]:
            self.data.ctrl[:7] = q_start * (1.0 - alpha) + q_target * alpha
            self.data.ctrl[7] = grip
            self.sim.step(steps_per_chunk)

    def hold(self, q_target, grip, steps):
        self.data.ctrl[:7] = q_target
        self.data.ctrl[7] = grip
        self.sim.step(steps)

    def score(self):
        cup = self.cup_pos()
        hand = self.hand_pos()
        return float(cup[2] - 0.1 * np.linalg.norm(cup[:2] - hand[:2]))


def attempt(params):
    planner = Planner()
    cup = planner.sim.cup_position().copy()
    x, y = cup[:2] + np.array(params["lateral"])
    hover_hi = planner.solve([x, y, 0.68])
    hover = planner.solve([x, y, 0.58], q_init=hover_hi)
    grasp = planner.solve([x, y, params["grasp_z"]], q_init=hover)
    lift1 = planner.solve([x, y, 0.56], q_init=grasp)
    lift2 = planner.solve([x, y, 0.62], q_init=lift1)

    exe = Executor()
    exe.move_q(hover_hi, 255.0, chunks=24, steps_per_chunk=40)
    exe.move_q(hover, 255.0, chunks=12, steps_per_chunk=40)
    exe.move_q(grasp, 255.0, chunks=10, steps_per_chunk=45)
    exe.hold(grasp, 255.0, 200)

    for grip in np.linspace(255.0, params["preclose"], 6):
        exe.hold(grasp, float(grip), 80)
    exe.sim.save_final_state(SAVE_PATH)
    for grip in np.linspace(params["preclose"], 0.0, 6):
        exe.hold(grasp, float(grip), 100)

    exe.move_q(lift1, 0.0, chunks=8, steps_per_chunk=50)
    exe.move_q(lift2, 0.0, chunks=12, steps_per_chunk=50)
    exe.hold(lift2, 0.0, 500)
    exe.sim.save_final_state(SAVE_PATH)
    return exe


def main():
    attempts = [
        {"lateral": (0.0, 0.0), "grasp_z": 0.495, "preclose": 70.0},
        {"lateral": (0.0, 0.0), "grasp_z": 0.490, "preclose": 40.0},
        {"lateral": (0.003, 0.0), "grasp_z": 0.493, "preclose": 40.0},
        {"lateral": (-0.003, 0.0), "grasp_z": 0.493, "preclose": 40.0},
        {"lateral": (0.0, 0.003), "grasp_z": 0.493, "preclose": 40.0},
        {"lateral": (0.0, -0.003), "grasp_z": 0.493, "preclose": 40.0},
    ]

    best = None
    for i, params in enumerate(attempts, 1):
        exe = attempt(params)
        cup = exe.cup_pos()
        hand = exe.hand_pos()
        score = exe.score()
        print(
            f"attempt {i}: cup={cup} hand={hand} fingers={exe.data.qpos[7:9]} score={score:.4f} params={params}"
        )
        if best is None or score > best[0]:
            best = (score, cup.copy(), params)
            exe.sim.save_final_state(SAVE_PATH)
    print("best", best)
    print("saved", Path(SAVE_PATH).exists(), SAVE_PATH)


if __name__ == "__main__":
    main()
