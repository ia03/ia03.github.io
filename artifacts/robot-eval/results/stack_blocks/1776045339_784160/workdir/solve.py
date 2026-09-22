import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import BLOCK_HALF, Sim


ARM_DOF = 7
GRIP_OPEN = 255.0
GRIP_CLOSE = 0.0


@dataclass
class EvalResult:
    dx: float
    dy: float
    dz: float
    steps: int

    @property
    def passed(self) -> bool:
        return abs(self.dx) <= 0.018 and abs(self.dy) <= 0.018 and 0.040 <= self.dz <= 0.070


class PandaController:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.home_q = np.array([0.0, 0.15, 0.0, -1.7, 0.0, 1.9, 0.78])
        self.target_rot = self.data.xmat[self.hand_id].reshape(3, 3).copy()

    def orientation_error(self) -> np.ndarray:
        current = self.data.xmat[self.hand_id].reshape(3, 3)
        return 0.5 * (
            np.cross(current[:, 0], self.target_rot[:, 0])
            + np.cross(current[:, 1], self.target_rot[:, 1])
            + np.cross(current[:, 2], self.target_rot[:, 2])
        )

    def ik_step(self, target_pos: np.ndarray, grip: float, pos_gain: float = 4.0, rot_gain: float = 1.5):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_id)
        jpos = jacp[:, :ARM_DOF]
        jrot = jacr[:, :ARM_DOF]

        pos_err = target_pos - self.data.xpos[self.hand_id]
        rot_err = self.orientation_error()
        task_j = np.vstack([jpos, 0.35 * jrot])
        task_err = np.concatenate([pos_gain * pos_err, rot_gain * rot_err])

        damping = 1e-3
        lhs = task_j.T @ task_j + damping * np.eye(ARM_DOF)
        rhs = task_j.T @ task_err + 0.01 * (self.home_q - self.data.qpos[:ARM_DOF])
        dq = np.linalg.solve(lhs, rhs)
        dq = np.clip(dq, -0.08, 0.08)

        ctrl = self.data.ctrl.copy()
        ctrl[:ARM_DOF] = np.clip(
            self.data.qpos[:ARM_DOF] + dq,
            self.model.actuator_ctrlrange[:ARM_DOF, 0],
            self.model.actuator_ctrlrange[:ARM_DOF, 1],
        )
        ctrl[7] = grip
        self.data.ctrl[:] = ctrl

    def move_to(self, target_pos, steps, grip, pos_gain=4.0):
        target_pos = np.array(target_pos, dtype=float)
        for _ in range(steps):
            self.ik_step(target_pos, grip, pos_gain=pos_gain)
            self.sim.step(1)

    def hold(self, steps, grip):
        target = self.data.xpos[self.hand_id].copy()
        self.move_to(target, steps, grip, pos_gain=5.0)


def evaluate_trace(ctrl_trace: np.ndarray) -> EvalResult:
    sim = Sim()
    for ctrl in ctrl_trace:
        sim.data.ctrl[:] = ctrl
        sim.step(1)
    sim.data.ctrl[:] = 0
    sim.step(500)
    pos = sim.block_positions()
    delta = pos["red"] - pos["green"]
    return EvalResult(dx=float(delta[0]), dy=float(delta[1]), dz=float(delta[2]), steps=len(ctrl_trace))


def run_attempt(save_path="/work/final_state.npz", render_path=None):
    sim = Sim()
    ctrl = PandaController(sim)

    red = sim.block_positions()["red"].copy()
    green = sim.block_positions()["green"].copy()

    hover_red = red + np.array([0.0, 0.0, 0.20])
    pre_grasp = red + np.array([0.0, 0.0, 0.095])
    grasp = red + np.array([0.0, 0.0, 0.058])
    lift = red + np.array([0.0, 0.0, 0.22])
    hover_green = green + np.array([0.0, 0.0, 0.20])
    pre_place = green + np.array([0.0, 0.0, 0.105])
    place = green + np.array([0.0, 0.0, 0.081])
    retreat = green + np.array([0.0, 0.0, 0.20])

    ctrl.move_to(ctrl.data.xpos[ctrl.hand_id] + np.array([0.0, 0.0, 0.02]), 100, GRIP_OPEN)
    ctrl.move_to(hover_red, 650, GRIP_OPEN)
    ctrl.move_to(pre_grasp, 350, GRIP_OPEN)
    ctrl.move_to(grasp, 260, GRIP_OPEN, pos_gain=5.0)
    ctrl.hold(40, GRIP_CLOSE)
    ctrl.hold(180, GRIP_CLOSE)
    ctrl.move_to(lift, 500, GRIP_CLOSE)

    sim.save_final_state(save_path)

    ctrl.move_to(hover_green, 700, GRIP_CLOSE)
    ctrl.move_to(pre_place, 280, GRIP_CLOSE)
    ctrl.move_to(place, 260, GRIP_CLOSE, pos_gain=5.0)
    ctrl.hold(60, GRIP_CLOSE)
    ctrl.hold(120, GRIP_OPEN)
    ctrl.move_to(retreat, 450, GRIP_OPEN)
    ctrl.hold(500, GRIP_OPEN)

    sim.save_final_state(save_path)

    if render_path:
        import imageio.v3 as iio

        iio.imwrite(render_path, sim.render())

    payload = np.load(save_path)
    result = evaluate_trace(payload["ctrl_trace"])
    return result


if __name__ == "__main__":
    result = run_attempt(render_path="/work/final.png")
    print(
        {
            "passed": result.passed,
            "dx": result.dx,
            "dy": result.dy,
            "dz": result.dz,
            "steps": result.steps,
        }
    )
