import math
import numpy as np
import mujoco

from sim import Sim


ARM_DOF = 7
HAND_BODY = "hand"
PINCH_OFFSET = np.array([0.053, 0.0, -0.090])
HOME_Q = np.array([0.0, 0.15, 0.0, -1.35, 0.0, 1.6, 0.785])


def rotation_error(target_mat, current_mat):
    return 0.5 * (
        np.cross(current_mat[:, 0], target_mat[:, 0])
        + np.cross(current_mat[:, 1], target_mat[:, 1])
        + np.cross(current_mat[:, 2], target_mat[:, 2])
    )


def damped_ls(jac, err, damping=1e-3):
    jj_t = jac @ jac.T
    return jac.T @ np.linalg.solve(jj_t + damping * np.eye(jj_t.shape[0]), err)


class Controller:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.target_rot = sim.data.xmat[self.hand_id].reshape(3, 3).copy()
        self.sim.data.ctrl[:ARM_DOF] = self.sim.data.qpos[:ARM_DOF]
        self.sim.data.ctrl[7] = 255.0
        self.sim.step(20)

    def hand_pose(self):
        pos = self.sim.data.xpos[self.hand_id].copy()
        rot = self.sim.data.xmat[self.hand_id].reshape(3, 3).copy()
        return pos, rot

    def step_ik(self, target_pos, grip=255.0, pos_gain=4.0, rot_gain=1.5):
        pos, rot = self.hand_pose()
        err_pos = target_pos - pos
        err_rot = rotation_error(self.target_rot, rot)

        jacp = np.zeros((3, self.sim.model.nv))
        jacr = np.zeros((3, self.sim.model.nv))
        mujoco.mj_jacBody(self.sim.model, self.sim.data, jacp, jacr, self.hand_id)
        jac = np.vstack([pos_gain * jacp[:, :ARM_DOF], rot_gain * jacr[:, :ARM_DOF]])
        err = np.concatenate([pos_gain * err_pos, rot_gain * err_rot])
        dq = damped_ls(jac, err, damping=2e-3)

        q = self.sim.data.qpos[:ARM_DOF].copy()
        dq += 0.08 * (HOME_Q - q)
        q_target = np.clip(q + 0.35 * dq, self.sim.model.actuator_ctrlrange[:ARM_DOF, 0], self.sim.model.actuator_ctrlrange[:ARM_DOF, 1])
        self.sim.data.ctrl[:ARM_DOF] = q_target
        self.sim.data.ctrl[7] = grip
        self.sim.step(1)

    def move_to(self, target_pos, steps, grip=255.0, pos_gain=4.0, rot_gain=1.5):
        for _ in range(steps):
            self.step_ik(target_pos, grip=grip, pos_gain=pos_gain, rot_gain=rot_gain)

    def move_linear(self, start, end, steps, grip=255.0, pos_gain=4.0, rot_gain=1.5):
        for alpha in np.linspace(0.0, 1.0, steps):
            target = (1.0 - alpha) * start + alpha * end
            self.step_ik(target, grip=grip, pos_gain=pos_gain, rot_gain=rot_gain)


def contact_fraction(sim: Sim, steps=500):
    contact = 0
    for _ in range(steps):
        sim.step(1)
        if sim.has_gripper_cup_contact():
            contact += 1
    return contact / steps


def replay_metrics(sim: Sim):
    trace = sim._trace
    xs = np.array([entry["cup_pos"][0] for entry in trace])
    zs = np.array([entry["cup_pos"][2] for entry in trace])
    contacts = np.array([entry["cup_contact"] for entry in trace])
    final = sim.cup_position().copy()
    return {
        "best_x": float(xs.min()),
        "best_z": float(zs.max()),
        "ever_contact": bool(np.any(contacts > 0.5)),
        "final_pos": final,
    }


def run_attempt():
    sim = Sim()
    ctrl = Controller(sim)

    cup = sim.cup_position().copy()
    grasp_hand = cup - PINCH_OFFSET
    above_side = np.array([0.44, 0.22, 0.66])
    near_side = np.array([0.49, 0.205, 0.58])
    pregrasp = grasp_hand + np.array([0.0, 0.03, 0.02])
    grasp = grasp_hand + np.array([0.0, 0.0, 0.01])
    squeeze = grasp + np.array([0.0, -0.005, -0.002])
    lift = np.array([0.47, 0.16, 0.68])
    retreat = np.array([0.39, 0.10, 0.70])
    settle = np.array([0.36, 0.08, 0.69])

    ctrl.move_to(np.array([0.20, 0.08, 0.82]), 120, grip=255.0)
    ctrl.move_linear(ctrl.hand_pose()[0], above_side, 180, grip=255.0)
    ctrl.move_linear(above_side, near_side, 140, grip=255.0)
    ctrl.move_linear(near_side, pregrasp, 120, grip=255.0)
    ctrl.move_linear(pregrasp, grasp, 90, grip=255.0, pos_gain=5.0)
    ctrl.move_linear(grasp, squeeze, 50, grip=255.0, pos_gain=5.0)

    for grip in np.linspace(255.0, 0.0, 140):
        ctrl.step_ik(squeeze, grip=float(grip), pos_gain=5.0, rot_gain=1.8)

    ctrl.move_to(squeeze, 80, grip=0.0, pos_gain=5.0, rot_gain=1.8)
    sim.save_final_state("/work/final_state.npz")

    ctrl.move_linear(squeeze, lift, 180, grip=0.0, pos_gain=5.0, rot_gain=1.8)
    ctrl.move_linear(lift, retreat, 220, grip=0.0, pos_gain=4.5, rot_gain=1.8)
    ctrl.move_linear(retreat, settle, 160, grip=0.0, pos_gain=4.0, rot_gain=1.8)
    ctrl.move_to(settle, 220, grip=0.0, pos_gain=4.0, rot_gain=1.8)

    metrics = replay_metrics(sim)
    settle_cf = contact_fraction(sim, steps=500)
    metrics["settle_contact_fraction"] = settle_cf
    metrics["post_settle_pos"] = sim.cup_position().copy()
    sim.save_final_state("/work/final_state.npz")
    return metrics


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    result = run_attempt()
    for key, value in result.items():
        print(key, value)
