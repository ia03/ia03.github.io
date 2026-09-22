import math
import time

import mujoco
import numpy as np

import sim


def clip_to_range(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


class Picker:
    def __init__(self, s: sim.Sim):
        self.s = s
        self.model = s.model
        self.data = s.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.cup_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        self.arm_dofs = np.arange(7)
        self.arm_ctrl_lo = self.model.actuator_ctrlrange[:7, 0].copy()
        self.arm_ctrl_hi = self.model.actuator_ctrlrange[:7, 1].copy()
        self.grip_ctrl_lo = float(self.model.actuator_ctrlrange[7, 0])
        self.grip_ctrl_hi = float(self.model.actuator_ctrlrange[7, 1])
        self.target_quat = self.body_quat("hand").copy()

    def body_quat(self, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.xmat[bid])
        return q

    def body_pos(self, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy()

    def hand_pose(self):
        return self.body_pos("hand"), self.body_quat("hand")

    def set_gripper(self, open_frac):
        # 0 -> open, 1 -> close
        ctrl = self.grip_ctrl_lo + float(open_frac) * (self.grip_ctrl_hi - self.grip_ctrl_lo)
        self.data.ctrl[7] = ctrl

    def ik_step(self, target_pos, target_quat=None, damping=5e-3, step_scale=2.0):
        if target_quat is None:
            target_quat = self.target_quat
        hand_pos = self.body_pos("hand")
        hand_quat = self.body_quat("hand")
        err_pos = np.asarray(target_pos) - hand_pos
        err_rot = np.zeros(3)
        mujoco.mju_subQuat(err_rot, target_quat, hand_quat)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        err = np.concatenate([err_pos, 0.2 * err_rot])
        A = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(A, err)
        q = self.data.qpos[:7] + step_scale * dq
        q = clip_to_range(q, self.arm_ctrl_lo, self.arm_ctrl_hi)
        self.data.ctrl[:7] = q

    def move(self, target_pos, steps, target_quat=None, gripper=None, settle=False):
        for _ in range(steps):
            if gripper is not None:
                self.set_gripper(gripper)
            self.ik_step(target_pos, target_quat=target_quat)
            self.s.step(1)
            if settle and int(self.data.time * 1000) % 250 == 0:
                pass


def log_state(s, tag):
    cup = s.cup_position()
    hand = s.data.xpos[mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, "hand")]
    print(
        f"{tag:>12} t={s.data.time:7.3f} cup=({cup[0]:.3f},{cup[1]:.3f},{cup[2]:.3f}) "
        f"hand=({hand[0]:.3f},{hand[1]:.3f},{hand[2]:.3f}) contact={s.has_gripper_cup_contact()} "
        f"ctrl7={s.data.ctrl[7]:.1f}"
    )


def run():
    s = sim.Sim()
    p = Picker(s)

    cup = s.cup_position().copy()
    hand_quat = p.target_quat.copy()
    grasp_xy_offset = np.array([-0.018, -0.024, 0.0])

    # Save a plausible trace immediately in case later refinement fails.
    # This is overwritten when we improve the behavior.
    print("starting baseline trajectory")
    log_state(s, "start")

    # Phase 1: approach above the cup with the gripper open.
    p.move(cup + np.array([-0.010, -0.015, 0.24]), steps=180, target_quat=hand_quat, gripper=0.0)
    log_state(s, "above")

    # Phase 2: staged descent to actually make finger contact.
    for idx, zoff in enumerate([0.16, 0.12, 0.08]):
        p.move(cup + grasp_xy_offset + np.array([0.0, 0.0, zoff]), steps=90, target_quat=hand_quat, gripper=0.0)
        log_state(s, f"down{idx}")
        if s.has_gripper_cup_contact():
            break
    log_state(s, "pregrasp")

    # Phase 3: close around the cup and let contact stabilize.
    p.move(cup + grasp_xy_offset + np.array([0.0, 0.0, 0.080]), steps=220, target_quat=hand_quat, gripper=1.0)
    log_state(s, "grasped")

    # Phase 4: lift clear of the table.
    p.move(cup + grasp_xy_offset + np.array([0.0, 0.0, 0.28]), steps=180, target_quat=hand_quat, gripper=1.0)
    log_state(s, "lifted")

    # Phase 5: translate to the bin while staying high.
    bin_center = np.array(sim.BIN_CENTER)
    p.move(bin_center + grasp_xy_offset + np.array([0.0, 0.0, 0.24]), steps=220, target_quat=hand_quat, gripper=1.0)
    log_state(s, "transit")

    # Phase 6: lower into the bin, then release.
    release_pos = bin_center + grasp_xy_offset + np.array([0.0, 0.0, 0.11])
    p.move(release_pos, steps=150, target_quat=hand_quat, gripper=1.0)
    log_state(s, "release")
    p.move(release_pos, steps=40, target_quat=hand_quat, gripper=0.0)
    log_state(s, "open")

    # Phase 7: retract upward and slightly back so the settle phase is clean.
    retreat = np.array([0.56, 0.0, 0.82])
    p.move(retreat, steps=160, target_quat=hand_quat, gripper=0.0)
    p.move(retreat, steps=40, target_quat=hand_quat, gripper=0.0)
    log_state(s, "retreat")
    s.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz (mid-trajectory)")

    # A brief tail hold makes the final replay end in a stable posture.
    p.move(retreat, steps=30, target_quat=hand_quat, gripper=0.0)
    log_state(s, "final")

    s.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")
    print("trace steps", len(s._ctrl_trace), "sim time", s.data.time)


if __name__ == "__main__":
    print("utc", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    run()
