import math
import os
import numpy as np
import mujoco

from sim import Sim, BLOCK_HALF


HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
HOME_GRIPPER = 255.0
ARM_DOF = 7
SETTLE_STEPS = 500


def mat_from_body(data, body_id):
    return data.xmat[body_id].reshape(3, 3).copy()


def orientation_error(target_R, current_R):
    return 0.5 * (
        np.cross(current_R[:, 0], target_R[:, 0])
        + np.cross(current_R[:, 1], target_R[:, 1])
        + np.cross(current_R[:, 2], target_R[:, 2])
    )


class Policy:
    def __init__(self, sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.hand_id = self.model.body("hand").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.home_R = mat_from_body(self.data, self.hand_id)
        sim.reset()

    def set_ctrl(self, q_target, grip):
        self.data.ctrl[:ARM_DOF] = q_target
        self.data.ctrl[7] = grip

    def solve_ik(self, target_pos, target_R=None, q_seed=None, pos_weight=1.0, rot_weight=0.35, iters=80):
        target_R = self.home_R if target_R is None else target_R
        q = self.data.qpos[:ARM_DOF].copy() if q_seed is None else q_seed.copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            self.data.qpos[:ARM_DOF] = q
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            hand_pos = self.data.xpos[self.hand_id].copy()
            hand_R = mat_from_body(self.data, self.hand_id)
            pos_err = target_pos - hand_pos
            rot_err = orientation_error(target_R, hand_R)
            err = np.concatenate([pos_weight * pos_err, rot_weight * rot_err])
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 3e-4:
                break
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_id)
            J = np.vstack([pos_weight * jacp[:, :ARM_DOF], rot_weight * jacr[:, :ARM_DOF]])
            damp = 1e-4
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
            q += dq
            q = np.clip(q, self.model.actuator_ctrlrange[:ARM_DOF, 0], self.model.actuator_ctrlrange[:ARM_DOF, 1])
        self.data.qpos[:] = self.sim._initial_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return q

    def move_to_pose(self, target_pos, grip, steps, q_hint=None):
        q_target = self.solve_ik(target_pos, q_seed=q_hint)
        q_start = self.data.ctrl[:ARM_DOF].copy()
        grip_start = float(self.data.ctrl[7])
        for i in range(steps):
            a = (i + 1) / steps
            q_cmd = (1 - a) * q_start + a * q_target
            g_cmd = (1 - a) * grip_start + a * grip
            self.set_ctrl(q_cmd, g_cmd)
            self.sim.step()
        return q_target

    def hold(self, steps, grip=None):
        q_target = self.data.ctrl[:ARM_DOF].copy()
        g = float(self.data.ctrl[7] if grip is None else grip)
        for _ in range(steps):
            self.set_ctrl(q_target, g)
            self.sim.step()


def compute_metrics(sim):
    pos = sim.block_positions()
    delta = pos["red"] - pos["green"]
    return {
        "red": pos["red"],
        "green": pos["green"],
        "delta": delta,
        "max_xy": float(max(abs(delta[0]), abs(delta[1]))),
        "dz": float(delta[2]),
    }


def run_attempt(
    grasp_z=0.485,
    carry_z=0.60,
    place_xy=(0.63, 0.08),
    place_z=0.535,
    preplace_z=0.60,
    open_grip=255.0,
    close_grip=0.0,
    release_grip=180.0,
):
    sim = Sim()
    policy = Policy(sim)
    policy.set_ctrl(HOME_QPOS, HOME_GRIPPER)
    policy.hold(500, HOME_GRIPPER)

    red = sim.block_positions()["red"].copy()
    green = sim.block_positions()["green"].copy()
    pregrasp = np.array([red[0], red[1], carry_z])
    grasp = np.array([red[0], red[1], grasp_z])
    preplace = np.array([place_xy[0], place_xy[1], preplace_z])
    place = np.array([place_xy[0], place_xy[1], place_z])

    q = policy.move_to_pose(pregrasp, open_grip, 500, q_hint=HOME_QPOS)
    q = policy.move_to_pose(grasp, open_grip, 450, q_hint=q)
    policy.hold(120, open_grip)
    policy.hold(80, close_grip)
    policy.hold(260, close_grip)
    q = policy.move_to_pose(pregrasp, close_grip, 500, q_hint=q)
    q = policy.move_to_pose(preplace, close_grip, 700, q_hint=q)
    q = policy.move_to_pose(place, close_grip, 450, q_hint=q)
    policy.hold(80, close_grip)
    policy.hold(120, release_grip)
    policy.hold(220, open_grip)
    q = policy.move_to_pose(preplace, open_grip, 500, q_hint=q)
    policy.hold(SETTLE_STEPS, open_grip)

    sim.save_final_state("/work/final_state.npz")
    metrics = compute_metrics(sim)
    metrics["green_start"] = green
    metrics["steps"] = len(sim._ctrl_trace)
    return metrics


if __name__ == "__main__":
    metrics = run_attempt()
    print(metrics)
    print("saved", os.path.exists("/work/final_state.npz"))
