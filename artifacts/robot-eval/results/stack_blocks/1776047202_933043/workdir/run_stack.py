import numpy as np
import mujoco

from sim import Sim, BLOCK_HALF


def _mat_from_xmat(xmat):
    return np.array(xmat, dtype=float).reshape(3, 3)


def _rot_err(R_cur, R_tgt):
    # orientation error in so(3): 0.5 * vee(R_tgt^T R_cur - R_cur^T R_tgt)
    R_err = R_tgt.T @ R_cur - R_cur.T @ R_tgt
    return 0.5 * np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])


def _damped_ls(J, err, damping=0.05):
    # Solve dq = J^T (J J^T + λ^2 I)^-1 err
    A = J @ J.T + (damping**2) * np.eye(J.shape[0])
    x = np.linalg.solve(A, err)
    return J.T @ x


class ScriptedStacker:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.hand_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.left_finger_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        self.left_tip_geom = self._find_finger_tip_geom(self.left_finger_body)
        self.right_tip_geom = self._find_finger_tip_geom(self.right_finger_body)
        self.arm_n = 7
        self.ctrl_low = self.model.actuator_ctrlrange[: self.arm_n, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[: self.arm_n, 1].copy()
        self.grip_open = 255.0
        self.grip_closed = 0.0

        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self._jacp_l = np.zeros((3, self.model.nv))
        self._jacp_r = np.zeros((3, self.model.nv))
        self._ik_data = mujoco.MjData(self.model)

        self.sim.reset()
        self._hold_current_ctrl()
        self.R_hand0 = _mat_from_xmat(self.data.xmat[self.hand_body])

    def _find_finger_tip_geom(self, finger_body_id: int) -> int:
        geoms = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == finger_body_id]
        if not geoms:
            raise RuntimeError("No geoms found for finger body.")
        return max(geoms, key=lambda i: float(self.model.geom_pos[i][2]))

    def _hold_current_ctrl(self):
        q = self.data.qpos[: self.arm_n].copy()
        self.data.ctrl[: self.arm_n] = np.clip(q, self.ctrl_low, self.ctrl_high)

    def _ee_pose(self, data=None):
        if data is None:
            data = self.data
        pos_l = np.array(data.geom_xpos[self.left_tip_geom], dtype=float)
        pos_r = np.array(data.geom_xpos[self.right_tip_geom], dtype=float)
        pos = 0.5 * (pos_l + pos_r)
        R = _mat_from_xmat(data.xmat[self.hand_body])
        return pos, R

    def _move_joints(self, q_tgt, steps=400, settle=50):
        q_tgt = np.clip(np.array(q_tgt, dtype=float), self.ctrl_low, self.ctrl_high)
        q0 = self.data.qpos[: self.arm_n].copy()
        for i in range(steps):
            a = (i + 1) / steps
            self.data.ctrl[: self.arm_n] = (1 - a) * q0 + a * q_tgt
            self.sim.step(1)
        # Hold until convergence (actuators can be slow).
        if settle:
            for _ in range(settle):
                self.data.ctrl[: self.arm_n] = q_tgt
                self.sim.step(1)

    def _track_joints(self, q_tgt, max_steps=1200, q_tol=0.01, v_tol=0.2):
        q_tgt = np.clip(np.array(q_tgt, dtype=float), self.ctrl_low, self.ctrl_high)
        for _ in range(max_steps):
            self.data.ctrl[: self.arm_n] = q_tgt
            self.sim.step(1)
            if np.max(np.abs(self.data.qpos[: self.arm_n] - q_tgt)) < q_tol and np.max(
                np.abs(self.data.qvel[: self.arm_n])
            ) < v_tol:
                break

    def _solve_ik(
        self,
        pos_tgt,
        R_tgt,
        iters=80,
        pos_tol=0.002,
        rot_tol=0.06,
        pos_gain=1.0,
        rot_gain=0.6,
        damping=0.06,
    ):
        pos_tgt = np.array(pos_tgt, dtype=float)
        d = self._ik_data
        d.qpos[:] = self.data.qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)

        for _ in range(iters):
            pos, R = self._ee_pose(d)
            e_pos = pos_tgt - pos
            e_rot = _rot_err(R, R_tgt)
            if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_rot) < rot_tol:
                break

            mujoco.mj_jacGeom(self.model, d, self._jacp_l, None, self.left_tip_geom)
            mujoco.mj_jacGeom(self.model, d, self._jacp_r, None, self.right_tip_geom)
            Jp_full = 0.5 * (self._jacp_l + self._jacp_r)
            mujoco.mj_jacBody(self.model, d, self._jacp, self._jacr, self.hand_body)

            Jp = Jp_full[:, : self.arm_n]
            Jr = self._jacr[:, : self.arm_n]
            J = np.vstack([Jp, Jr])
            err = np.hstack([pos_gain * e_pos, rot_gain * e_rot])
            dq = _damped_ls(J, err, damping=damping)

            max_dq = 0.08
            dq = np.clip(dq, -max_dq, max_dq)
            d.qpos[: self.arm_n] = np.clip(d.qpos[: self.arm_n] + dq, self.ctrl_low, self.ctrl_high)
            mujoco.mj_forward(self.model, d)

        return d.qpos[: self.arm_n].copy()

    def _goto_pose(self, pos_tgt, R_tgt, max_steps=1400):
        q_tgt = self._solve_ik(pos_tgt, R_tgt)
        self._track_joints(q_tgt, max_steps=max_steps)

    def run(self):
        # Start from a reasonable pose above the table.
        self.data.ctrl[7] = self.grip_open
        home = np.array([0.0, -0.6, 0.0, -2.1, 0.0, 1.6, 0.8])
        self._track_joints(home, max_steps=1200)
        # Use the reachable hand orientation at the working pose.
        _, self.R_task = self._ee_pose()

        blocks = self.sim.block_positions()
        red = blocks["red"]
        green = blocks["green"]

        hover = red + np.array([0.0, 0.0, 0.30])
        pregrasp = red + np.array([0.0, 0.0, 0.16])
        self._goto_pose(hover, self.R_task, max_steps=1600)
        self._goto_pose(pregrasp, self.R_task, max_steps=1400)

        # Put fingertip midpoint around the block center height.
        grasp = red + np.array([0.0, 0.0, 0.002])
        self._goto_pose(grasp, self.R_task, max_steps=1400)

        # Close gripper and let contacts settle.
        self.data.ctrl[7] = self.grip_closed
        self.sim.step(350)

        # Lift.
        lift = red + np.array([0.0, 0.0, 0.30])
        self._goto_pose(lift, self.R_task, max_steps=1800)
        self.sim.step(100)

        # Estimate hand->red offset while grasped.
        blocks = self.sim.block_positions()
        red = blocks["red"]
        ee_pos, _ = self._ee_pose()
        ee_to_red = red - ee_pos

        # Move above green.
        above_green = green + np.array([0.0, 0.0, 0.30]) - ee_to_red
        self._goto_pose(above_green, self.R_task, max_steps=2000)

        # Lower for placement: aim for dz ~ 0.05.
        place_red = green + np.array([0.0, 0.0, 0.05])
        place_hand = place_red - ee_to_red
        self._goto_pose(place_hand, self.R_task, max_steps=2000)
        self.sim.step(120)

        # Release and retreat.
        self.data.ctrl[7] = self.grip_open
        self.sim.step(200)
        retreat = place_hand + np.array([0.0, 0.0, 0.12])
        self._goto_pose(retreat, self.R_task, max_steps=1800)

        # Let the stack settle before saving.
        self.sim.step(900)


def main():
    sim = Sim()
    ScriptedStacker(sim).run()
    sim.save_final_state("/work/final_state.npz")
    pos = sim.block_positions()
    dxdyz = pos["red"] - pos["green"]
    print("saved /work/final_state.npz")
    print("final red", pos["red"], "green", pos["green"], "dxdyz", dxdyz)


if __name__ == "__main__":
    main()
