import numpy as np
import mujoco

from sim import Sim


def _axis_angle_from_R(R):
    # Robust axis-angle from rotation matrix (world frame).
    tr = np.trace(R)
    cos_angle = (tr - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-6:
        return np.zeros(3)
    if np.pi - angle < 1e-5:
        # Near pi: use diagonal for axis.
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) * 0.5, 0.0))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / (2.0 * np.sin(angle) + 1e-12)
    return axis * angle


class PandaStackPolicy:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.m = sim.model
        self.d = sim.data
        self.hand_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.lf_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.rf_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        # Keep initial wrist orientation as a soft prior; controller below is position-only.
        self.R_des = self.d.xmat[self.hand_id].reshape(3, 3).copy()

        self.ctrl_low = self.m.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.m.actuator_ctrlrange[:, 1].copy()
        self.q_cmd = np.clip(self.d.qpos[:7].copy(), self.ctrl_low[:7], self.ctrl_high[:7])

    def hand_pose(self):
        pos = self.d.xpos[self.hand_id].copy()
        R = self.d.xmat[self.hand_id].reshape(3, 3).copy()
        return pos, R

    def mid_pos(self):
        return ((self.d.xpos[self.lf_id] + self.d.xpos[self.rf_id]) / 2.0).copy()

    def _ik_step_mid(self, target_mid, kp_pos=5.0, damping=1.2e-1, max_dq=0.025, step_scale=0.35):
        cur_mid = self.mid_pos()
        pos_err = (target_mid - cur_mid) * kp_pos

        jacp_l = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBody(self.m, self.d, jacp_l, jacr, self.lf_id)
        jacp_r = np.zeros((3, self.m.nv))
        mujoco.mj_jacBody(self.m, self.d, jacp_r, jacr, self.rf_id)
        J = 0.5 * (jacp_l[:, :7] + jacp_r[:, :7])

        A = J @ J.T + (damping**2) * np.eye(3)
        x = np.linalg.solve(A, pos_err)
        dq = J.T @ x
        dq = dq * step_scale
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > max_dq:
            dq = dq * (max_dq / (dq_norm + 1e-12))

        self.q_cmd = np.clip(self.q_cmd + dq, self.ctrl_low[:7], self.ctrl_high[:7])
        return self.q_cmd

    def _set_ctrl(self, q_cmd, grip_cmd):
        self.d.ctrl[:7] = q_cmd
        self.d.ctrl[7] = float(np.clip(grip_cmd, self.ctrl_low[7], self.ctrl_high[7]))

    def move_mid_linear(self, target_mid, grip_cmd, steps, kp_pos=5.0):
        start_mid = self.mid_pos()
        for t in range(steps):
            u = (t + 1) / steps
            desired_mid = (1 - u) * start_mid + u * target_mid
            q_cmd = self._ik_step_mid(desired_mid, kp_pos=kp_pos)
            self._set_ctrl(q_cmd, grip_cmd)
            self.sim.step(1)

    def hold(self, mid_target, grip_cmd, steps, kp_pos=5.0):
        for _ in range(steps):
            q_cmd = self._ik_step_mid(mid_target, kp_pos=kp_pos)
            self._set_ctrl(q_cmd, grip_cmd)
            self.sim.step(1)

    def move_mid_until(self, target_mid, grip_cmd, max_steps, tol=0.01, kp_pos=5.0, stable_steps=50):
        target_mid = np.array(target_mid, dtype=float)
        ok = 0
        for _ in range(max_steps):
            q_cmd = self._ik_step_mid(target_mid, kp_pos=kp_pos)
            self._set_ctrl(q_cmd, grip_cmd)
            self.sim.step(1)
            err = float(np.linalg.norm(self.mid_pos() - target_mid))
            if err <= tol:
                ok += 1
                if ok >= stable_steps:
                    break
            else:
                ok = 0

    def move_joints_linear(self, q_target, grip_cmd, steps):
        q_target = np.clip(np.array(q_target, dtype=float), self.ctrl_low[:7], self.ctrl_high[:7])
        q_start = self.q_cmd.copy()
        for t in range(steps):
            u = (t + 1) / steps
            self.q_cmd = (1 - u) * q_start + u * q_target
            self._set_ctrl(self.q_cmd, grip_cmd)
            self.sim.step(1)


def run(out_path="/work/final_state.npz", render_debug=False):
    sim = Sim()
    pol = PandaStackPolicy(sim)

    def dbg(tag):
        pos = sim.block_positions()
        mid = pol.mid_pos()
        print(f"{tag}: t={sim.data.time:.3f} mid={mid} red={pos['red']} green={pos['green']} grip={sim.data.ctrl[7]:.1f}")

    blocks = sim.block_positions()
    red0 = blocks["red"].copy()
    green0 = blocks["green"].copy()

    # Start from a safer, typical Panda posture before doing task-space IK.
    q_home = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 2.0, 0.8])
    pol.move_joints_linear(q_home, grip_cmd=255, steps=1200)
    dbg("home")

    # Use high-clearance waypoints, then vertical descents to reduce link sweeps.
    z_high = 0.75
    z_lift = 0.82
    pregrasp = np.array([red0[0], red0[1], z_high])
    grasp = np.array([red0[0], red0[1], red0[2] + 0.015])
    lift = np.array([red0[0], red0[1], z_lift])

    preplace = np.array([green0[0], green0[1], z_high])
    place = np.array([green0[0], green0[1], green0[2] + 0.055])
    retreat = np.array([0.40, 0.00, 0.70])

    # Phase 1: approach red with open gripper.
    pol.move_mid_until(pregrasp, grip_cmd=255, max_steps=2600, tol=0.015, kp_pos=4.0)
    dbg("pregrasp")
    pol.move_mid_until(grasp, grip_cmd=255, max_steps=2200, tol=0.010, kp_pos=4.0)
    dbg("at_grasp")

    # Phase 2: close gripper while holding.
    for t in range(700):
        grip = 255 * (1 - (t + 1) / 700)
        pol.hold(grasp, grip_cmd=grip, steps=1, kp_pos=4.0)
    pol.hold(grasp, grip_cmd=0, steps=700, kp_pos=4.0)
    dbg("closed")

    # Phase 3: lift red.
    pol.move_mid_until(lift, grip_cmd=0, max_steps=2200, tol=0.020, kp_pos=3.5)
    dbg("lifted")

    # Phase 4: move above green and place.
    pol.move_mid_until(preplace, grip_cmd=0, max_steps=3200, tol=0.020, kp_pos=3.5)
    dbg("preplace")
    pol.move_mid_until(place, grip_cmd=0, max_steps=2600, tol=0.010, kp_pos=4.0)
    dbg("at_place")

    # Phase 5: release.
    for t in range(600):
        grip = 255 * ((t + 1) / 600)
        pol.hold(place, grip_cmd=grip, steps=1, kp_pos=4.5)
    pol.hold(place, grip_cmd=255, steps=350, kp_pos=4.0)
    dbg("released")

    # Phase 6: retreat and settle.
    pol.move_mid_until(preplace, grip_cmd=255, max_steps=1600, tol=0.025, kp_pos=3.5)
    pol.move_mid_until(retreat, grip_cmd=255, max_steps=2200, tol=0.030, kp_pos=3.0)
    dbg("retreat")
    pol.hold(retreat, grip_cmd=255, steps=1600, kp_pos=5.0)
    dbg("settled")

    # Local estimate of score metrics after an extra settle (mirrors grader).
    for _ in range(500):
        sim.step(1)
    pos = sim.block_positions()
    dxyz = pos["red"] - pos["green"]
    print(f"after_settle: dx={dxyz[0]:.4f} dy={dxyz[1]:.4f} dz={dxyz[2]:.4f}")

    if render_debug:
        frame = sim.render()
        from PIL import Image
        Image.fromarray(frame).save("/work/debug_final.png")

    sim.save_final_state(out_path)
    print(f"saved: {out_path} steps={len(sim._ctrl_trace)}")


if __name__ == "__main__":
    run()
