import math
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


FINGER_GEOM_IDS = (69, 77)


@dataclass
class EvalResult:
    contact_seen: bool
    reached_x: bool
    reached_z: bool
    final_x: float
    final_z: float
    settle_contact_fraction: float
    passed: bool


class CupRetriever:
    def __init__(self):
        self.sim = Sim()
        self.model = self.sim.model
        self.data = self.sim.data
        self.arm_limits = self.model.jnt_range[:7].copy()
        self.q_home = self.data.qpos[:7].copy()
        self.best_score = -1e9

    def pinch_pos(self):
        return np.mean([self.data.geom_xpos[g] for g in FINGER_GEOM_IDS], axis=0)

    def pinch_jac(self):
        jac = np.zeros((3, self.model.nv))
        for geom_id in FINGER_GEOM_IDS:
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, geom_id)
            jac += jacp
        return 0.5 * jac[:, :7]

    def solve_ik(self, target, q_seed=None, steps=200, alpha=0.2, wrist7=None):
        q = self.data.qpos[:7].copy() if q_seed is None else q_seed.copy()
        if wrist7 is not None:
            q[6] = wrist7
        for _ in range(steps):
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)
            err = target - self.pinch_pos()
            if np.linalg.norm(err) < 1e-4:
                break
            jac = self.pinch_jac()
            if wrist7 is not None:
                jac[:, 6] = 0.0
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(3), err)
            q = np.clip(q + alpha * dq, self.arm_limits[:, 0], self.arm_limits[:, 1])
            if wrist7 is not None:
                q[6] = wrist7
        self.data.qpos[:7] = q
        mujoco.mj_forward(self.model, self.data)
        return q

    def set_arm_target(self, q_target, grip):
        self.data.ctrl[:7] = q_target
        self.data.ctrl[7] = grip

    def track_target(self, target, grip, q_seed, hold_steps=10, max_iters=180, tol=0.01, wrist7=None):
        q_target = self.solve_ik(target, q_seed=q_seed, wrist7=wrist7)
        for _ in range(max_iters):
            self.set_arm_target(q_target, grip)
            self.sim.step(4)
            if np.linalg.norm(self.pinch_pos() - target) < tol:
                break
        for _ in range(hold_steps):
            self.set_arm_target(q_target, grip)
            self.sim.step(4)
        return q_target

    def close_gripper(self, q_hold, start=255.0, end=0.0, steps=80):
        for grip in np.linspace(start, end, steps):
            self.set_arm_target(q_hold, grip)
            self.sim.step(3)
        for _ in range(50):
            self.set_arm_target(q_hold, end)
            self.sim.step(3)

    def hold(self, q_hold, grip=0.0, steps=240):
        for _ in range(steps):
            self.set_arm_target(q_hold, grip)
            self.sim.step(2)

    def replay_eval(self):
        ctrl_trace = np.array(self.sim._ctrl_trace, dtype=float).reshape(-1, self.model.nu)
        replay = Sim()
        contact_seen = False
        reached_x = False
        reached_z = False
        for ctrl in ctrl_trace:
            replay.data.ctrl[:] = ctrl
            replay.step(1)
            cup = replay.cup_position()
            contact_seen |= replay.has_gripper_cup_contact()
            reached_x |= cup[0] <= 0.50
            reached_z |= cup[2] >= 0.56
        settle_contacts = 0
        for _ in range(500):
            replay.step(1)
            settle_contacts += float(replay.has_gripper_cup_contact())
        final_cup = replay.cup_position()
        fraction = settle_contacts / 500.0
        passed = (
            len(ctrl_trace) >= 20
            and contact_seen
            and reached_x
            and reached_z
            and final_cup[0] <= 0.42
            and final_cup[2] >= 0.56
            and fraction >= 0.5
        )
        return EvalResult(
            contact_seen=contact_seen,
            reached_x=reached_x,
            reached_z=reached_z,
            final_x=float(final_cup[0]),
            final_z=float(final_cup[2]),
            settle_contact_fraction=float(fraction),
            passed=passed,
        )

    def maybe_save(self, result):
        score = (
            1000.0 * float(result.passed)
            + 100.0 * float(result.contact_seen)
            + 50.0 * float(result.reached_x)
            + 50.0 * float(result.reached_z)
            - 100.0 * max(result.final_x - 0.42, 0.0)
            + 100.0 * min(result.final_z, 0.56)
            + 100.0 * result.settle_contact_fraction
        )
        if score > self.best_score:
            self.best_score = score
            self.sim.save_final_state("/work/final_state.npz")
            print("saved best", score, result)

    def run_attempt(self, side_y=0.24, grasp_z=0.438, lift_z=0.78, retreat_x=0.37, retreat_z=0.66, wrist7=1.0):
        self.sim.reset()
        q = self.data.qpos[:7].copy()
        open_grip = 255.0
        cup = self.sim.cup_position()

        waypoints = [
            np.array([0.42, side_y, lift_z]),
            np.array([0.58, side_y, lift_z]),
            np.array([cup[0], side_y, lift_z]),
            np.array([cup[0], cup[1], lift_z]),
            np.array([cup[0], cup[1], 0.58]),
            np.array([cup[0], cup[1], grasp_z]),
        ]
        for target in waypoints:
            q = self.track_target(target, open_grip, q, wrist7=wrist7)

        self.close_gripper(q)

        post_grasp = [
            np.array([cup[0], cup[1], 0.58]),
            np.array([cup[0], cup[1], lift_z]),
            np.array([cup[0], side_y, lift_z]),
            np.array([retreat_x, side_y, lift_z]),
            np.array([retreat_x, 0.04, retreat_z]),
            np.array([retreat_x, 0.04, 0.64]),
        ]
        for target in post_grasp:
            q = self.track_target(target, 0.0, q, hold_steps=15, max_iters=220, tol=0.015, wrist7=wrist7)
        self.hold(q, steps=280)
        result = self.replay_eval()
        print("attempt", side_y, grasp_z, lift_z, retreat_x, retreat_z, result)
        self.maybe_save(result)
        return result


def main():
    solver = CupRetriever()
    params = [
        (0.24, 0.438, 0.78, 0.37, 0.66, 1.0),
        (0.22, 0.438, 0.80, 0.35, 0.68, 1.0),
        (0.24, 0.442, 0.80, 0.36, 0.66, 1.1),
        (0.26, 0.438, 0.79, 0.38, 0.65, 0.9),
    ]
    best = None
    for p in params:
        result = solver.run_attempt(*p)
        if result.passed:
            best = result
            break
        best = result
    print("best result", best)


if __name__ == "__main__":
    main()
