import math
import numpy as np
import mujoco

from sim import Sim, BIN_CENTER, CUP_INIT_POS


HAND_BODY = "hand"
HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
GRASP_BIAS = np.array([0.009, -0.008, 0.0], dtype=float)


def rot_error(target, current):
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )


class Controller:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.hand_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.ctrl_lo = sim.model.actuator_ctrlrange[:7, 0].copy()
        self.ctrl_hi = sim.model.actuator_ctrlrange[:7, 1].copy()
        self.nominal_q = HOME_Q.copy()
        self.target_rot = None
        self.hand_offset_local = None

    def hand_pose(self):
        pos = self.sim.data.xpos[self.hand_body_id].copy()
        rot = self.sim.data.xmat[self.hand_body_id].reshape(3, 3).copy()
        return pos, rot

    def desired_hand_pos(self, pinch_target):
        return pinch_target - self.target_rot @ self.hand_offset_local

    def ik_step(self, hand_target, orient_weight=0.35):
        pos, rot = self.hand_pose()
        pos_err = hand_target - pos
        ori_err = rot_error(self.target_rot, rot)
        err = np.concatenate([pos_err, orient_weight * ori_err])

        jacp = np.zeros((3, self.sim.model.nv))
        jacr = np.zeros((3, self.sim.model.nv))
        mujoco.mj_jacBody(self.sim.model, self.sim.data, jacp, jacr, self.hand_body_id)
        J = np.vstack([jacp[:, :7], orient_weight * jacr[:, :7]])
        damping = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        dq += 0.05 * (self.nominal_q - self.sim.data.qpos[:7])
        q_des = self.sim.data.qpos[:7] + 0.8 * dq
        return np.clip(q_des, self.ctrl_lo, self.ctrl_hi), np.linalg.norm(pos_err), np.linalg.norm(ori_err)

    def drive(self, pinch_target, grip, steps, substeps=8):
        hand_target = self.desired_hand_pos(np.array(pinch_target, dtype=float))
        for i in range(steps):
            q_des, pos_err, _ = self.ik_step(hand_target)
            self.sim.data.ctrl[:7] = q_des
            self.sim.data.ctrl[7] = grip(i) if callable(grip) else grip
            self.sim.step(substeps)
            if pos_err < 0.002 and i > 10:
                break


def evaluate_ctrl_trace(ctrl_trace):
    sim = Sim()
    cup_id = sim.cup_body_id
    contact_hist = []
    best_z = sim.cup_position()[2]
    for ctrl in ctrl_trace:
        sim.data.ctrl[:] = ctrl
        mujoco.mj_step(sim.model, sim.data)
        contact = 1.0 if sim.has_gripper_cup_contact() else 0.0
        contact_hist.append(contact)
        best_z = max(best_z, float(sim.data.xpos[cup_id][2]))

    replay_contact_tail = np.mean(contact_hist[-10:]) if len(contact_hist) >= 10 else 1.0
    for _ in range(500):
        mujoco.mj_step(sim.model, sim.data)
    settle_contact = []
    for _ in range(50):
        settle_contact.append(1.0 if sim.has_gripper_cup_contact() else 0.0)
        mujoco.mj_step(sim.model, sim.data)
    settle_contact_frac = float(np.mean(settle_contact))
    cup_pos = sim.cup_position()
    cup_rot = sim.data.xmat[cup_id].reshape(3, 3)
    up_z = float(cup_rot[2, 2])
    bin_dx = abs(cup_pos[0] - BIN_CENTER[0])
    bin_dy = abs(cup_pos[1] - BIN_CENTER[1])
    passed = (
        len(ctrl_trace) >= 20
        and max(contact_hist, default=0.0) > 0.5
        and best_z >= 0.50
        and replay_contact_tail < 0.5
        and settle_contact_frac < 0.1
        and bin_dx <= 0.046
        and bin_dy <= 0.046
        and cup_pos[2] >= 0.42
        and up_z >= 0.85
    )
    return {
        "passed": passed,
        "best_z": best_z,
        "replay_contact_tail": replay_contact_tail,
        "settle_contact_frac": settle_contact_frac,
        "final_cup_pos": cup_pos,
        "up_z": up_z,
        "bin_dx": bin_dx,
        "bin_dy": bin_dy,
    }


def main():
    sim = Sim()
    sim.data.ctrl[:7] = HOME_Q
    sim.data.ctrl[7] = 255
    sim.step(400)
    ctl = Controller(sim)
    ctl.target_rot = sim.data.xmat[ctl.hand_body_id].reshape(3, 3).copy()
    finger_pts = np.array([sim.data.geom_xpos[gid].copy() for gid in sorted(sim.finger_geom_ids)])
    hand_pos, hand_rot = ctl.hand_pose()
    ctl.hand_offset_local = hand_rot.T @ (finger_pts.mean(axis=0) - hand_pos)

    cup = np.array(CUP_INIT_POS, dtype=float)
    bin_center = np.array([BIN_CENTER[0], BIN_CENTER[1], 0.445], dtype=float)
    grasp = cup + GRASP_BIAS

    ctl.drive(grasp + np.array([0.0, 0.0, 0.11]), 255, steps=80)
    ctl.drive(grasp + np.array([0.0, 0.0, 0.035]), 255, steps=100)
    ctl.drive(grasp + np.array([0.0, 0.0, 0.01]), lambda i: max(0, 255 - 3 * i), steps=90)
    ctl.drive(grasp + np.array([0.0, 0.0, 0.01]), 0, steps=40)
    ctl.drive(grasp + np.array([0.0, 0.0, 0.04]), 0, steps=50)
    ctl.drive(grasp + np.array([0.0, 0.0, 0.07]), 0, steps=60)
    ctl.drive(grasp + np.array([0.0, 0.0, 0.10]), 0, steps=60)
    ctl.drive(grasp + np.array([0.0, 0.0, 0.14]), 0, steps=70)

    sim.save_final_state("/work/final_state.npz")

    ctl.drive(np.array([bin_center[0], bin_center[1], 0.62]), 0, steps=120)
    ctl.drive(np.array([bin_center[0], bin_center[1], 0.56]), 0, steps=100)
    ctl.drive(bin_center + np.array([0.0, 0.0, 0.03]), 0, steps=90)
    ctl.drive(bin_center, lambda i: min(255, 4 * i), steps=80)
    ctl.drive(bin_center, 255, steps=40)
    ctl.drive(bin_center + np.array([-0.04, 0.0, 0.14]), 255, steps=120)
    sim.step(120)

    sim.save_final_state("/work/final_state.npz")
    metrics = evaluate_ctrl_trace(sim._ctrl_trace)
    print(metrics)


if __name__ == "__main__":
    main()
