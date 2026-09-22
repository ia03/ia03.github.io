import numpy as np
import mujoco

from sim import Sim, SUCCESS_CUP_Z


def _mat_to_rotvec(R):
    # Returns axis-angle vector (axis * angle) from a rotation matrix.
    trace = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(trace))
    if angle < 1e-8:
        return np.zeros(3)
    denom = 2.0 * np.sin(angle)
    axis = np.array(
        [
            (R[2, 1] - R[1, 2]) / denom,
            (R[0, 2] - R[2, 0]) / denom,
            (R[1, 0] - R[0, 1]) / denom,
        ],
        dtype=float,
    )
    return axis * angle


def _pose_error(p, R, p_des, R_des):
    dp = p_des - p
    if R_des is None:
        dr = np.zeros(3)
    else:
        R_err = R_des @ R.T
        dr = _mat_to_rotvec(R_err)
    return dp, dr


class PandaPick:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.m = sim.model
        self.d = sim.data

        self.hand_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.cup_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "cup")
        self.left_finger_body = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        self.right_finger_body = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )

        self.arm_joint_ids = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ]
        self.arm_jnt_range = np.array(self.m.jnt_range[self.arm_joint_ids], dtype=float)

        self.finger_geom_ids = self._finger_geom_ids()
        self.cup_geom_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")

        self.R0 = self.d.xmat[self.hand_id].reshape(3, 3).copy()
        self.q_target = self.d.qpos[:7].copy()
        self.gripper_cmd = 255.0  # open

    def _finger_geom_ids(self):
        ids = []
        for gid in range(self.m.ngeom):
            b = int(self.m.geom_bodyid[gid])
            if b in (self.left_finger_body, self.right_finger_body):
                ids.append(gid)
        return set(ids)

    def _set_ctrl(self):
        self.d.ctrl[:7] = self.q_target
        self.d.ctrl[7] = self.gripper_cmd

    def _step(self, n=1):
        for _ in range(n):
            self._set_ctrl()
            self.sim.step(1)

    def _hand_pose(self):
        p = self.d.xpos[self.hand_id].copy()
        R = self.d.xmat[self.hand_id].reshape(3, 3).copy()
        return p, R

    def _jacobian_body_pos(self, body_id):
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBody(self.m, self.d, jacp, jacr, body_id)
        return jacp[:, :7]

    def finger_midpoint(self):
        return 0.5 * (
            self.d.xpos[self.left_finger_body] + self.d.xpos[self.right_finger_body]
        )

    def _jacobian_finger_midpoint(self):
        Jl = self._jacobian_body_pos(self.left_finger_body)
        Jr = self._jacobian_body_pos(self.right_finger_body)
        return 0.5 * (Jl + Jr)

    def _ik_step(
        self,
        p_des,
        R_des,
        pos_gain=4.0,
        rot_gain=2.0,
        damping=0.10,
        max_step=0.015,
    ):
        # Track from the robot's actual current configuration to avoid target drift.
        self.q_target = self.d.qpos[:7].copy()
        p, R = self._hand_pose()
        dp, dr = _pose_error(p, R, p_des, R_des)

        jacp = self._jacobian_body_pos(self.hand_id)
        if R_des is None:
            err = pos_gain * dp
            A = jacp @ jacp.T + (damping**2) * np.eye(3)
            dq = jacp.T @ np.linalg.solve(A, err)
        else:
            err = np.concatenate([pos_gain * dp, rot_gain * dr])
            jacp_full = np.zeros((3, self.m.nv))
            jacr_full = np.zeros((3, self.m.nv))
            mujoco.mj_jacBody(self.m, self.d, jacp_full, jacr_full, self.hand_id)
            jacr = jacr_full[:, :7]
            J = np.vstack([jacp, jacr])
            A = J @ J.T + (damping**2) * np.eye(6)
            dq = J.T @ np.linalg.solve(A, err)
        dq = np.clip(dq, -max_step, max_step)

        self.q_target = np.clip(
            self.q_target + dq, self.arm_jnt_range[:, 0], self.arm_jnt_range[:, 1]
        )

    def move_pose(
        self,
        p_des,
        R_des=None,
        steps=1500,
        tol_pos=0.005,
        tol_rot=0.15,
        **ik_kwargs,
    ):
        for _ in range(int(steps)):
            self._ik_step(p_des, R_des, **ik_kwargs)
            self._step(1)
            if _ % 50 == 0:
                p, R = self._hand_pose()
                dp, dr = _pose_error(p, R, p_des, R_des)
                if np.linalg.norm(dp) < tol_pos and (
                    R_des is None or np.linalg.norm(dr) < tol_rot
                ):
                    break

    def settle(self, steps=200):
        self._step(int(steps))

    def move_finger_midpoint(
        self,
        p_des,
        steps=2000,
        pos_gain=3.0,
        damping=0.12,
        max_step=0.012,
        tol_pos=0.004,
    ):
        for t in range(int(steps)):
            self.q_target = self.d.qpos[:7].copy()
            p = self.finger_midpoint().copy()
            dp = p_des - p
            J = self._jacobian_finger_midpoint()
            A = J @ J.T + (damping**2) * np.eye(3)
            dq = J.T @ np.linalg.solve(A, pos_gain * dp)
            dq = np.clip(dq, -max_step, max_step)
            self.q_target = np.clip(
                self.q_target + dq,
                self.arm_jnt_range[:, 0],
                self.arm_jnt_range[:, 1],
            )
            self._step(1)
            if t % 50 == 0 and np.linalg.norm(dp) < tol_pos:
                break

    def cup_pos(self):
        return self.d.xpos[self.cup_id].copy()

    def cup_z(self):
        return float(self.cup_pos()[2])

    def contact_with_cup(self):
        # True if any finger geom is in contact with the cup geom.
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.geom1 == self.cup_geom_id and c.geom2 in self.finger_geom_ids:
                return True
            if c.geom2 == self.cup_geom_id and c.geom1 in self.finger_geom_ids:
                return True
        return False

    def contact_fraction(self, steps=500):
        count = 0
        for _ in range(int(steps)):
            self._step(1)
            if self.contact_with_cup():
                count += 1
        return count / float(steps)


def run(out_path="/work/final_state.npz", render_debug=False):
    sim = Sim()
    bot = PandaPick(sim)

    # Phase 0: open gripper
    bot.gripper_cmd = 255.0
    bot.settle(200)

    cup0 = bot.cup_pos()

    # Phase 1: move finger midpoint above cup
    pre_mid = cup0 + np.array([0.0, 0.0, 0.25])
    bot.move_finger_midpoint(pre_mid, steps=2600, pos_gain=3.0, damping=0.14, max_step=0.015)

    # Phase 2: descend midpoint to near cup center height
    cup1 = bot.cup_pos()
    grasp_mid = cup1 + np.array([0.0, 0.0, 0.01])
    bot.move_finger_midpoint(grasp_mid, steps=2600, pos_gain=2.5, damping=0.16, max_step=0.010, tol_pos=0.003)

    # Phase 3: close gripper while keeping midpoint fixed
    cup2 = bot.cup_pos()
    hold_mid = cup2 + np.array([0.0, 0.0, 0.005])
    for k in range(650):
        bot.q_target = bot.d.qpos[:7].copy()
        p = bot.finger_midpoint().copy()
        dp = hold_mid - p
        J = bot._jacobian_finger_midpoint()
        A = J @ J.T + (0.18**2) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, 2.0 * dp)
        dq = np.clip(dq, -0.008, 0.008)
        bot.q_target = np.clip(
            bot.q_target + dq,
            bot.arm_jnt_range[:, 0],
            bot.arm_jnt_range[:, 1],
        )
        bot.gripper_cmd = float(max(0.0, 255.0 * (1.0 - (k + 1) / 520.0)))
        bot._step(1)

    # Phase 4: lift by raising finger midpoint
    mid = bot.finger_midpoint().copy()
    lift_mid = mid + np.array([0.0, 0.0, 0.30])
    bot.gripper_cmd = 0.0
    bot.move_finger_midpoint(lift_mid, steps=3200, pos_gain=2.2, damping=0.16, max_step=0.010, tol_pos=0.004)

    # Phase 5: hold at top
    bot.move_finger_midpoint(lift_mid, steps=1600, pos_gain=1.8, damping=0.18, max_step=0.008, tol_pos=0.004)

    # Quick internal settle diagnostic (adds to trace, but also reduces risk).
    contact_frac = bot.contact_fraction(steps=500)
    final_z = bot.cup_z()
    print(f"cup_z={final_z:.3f} (target {SUCCESS_CUP_Z}) contact_frac~{contact_frac:.2f}")

    sim.save_final_state(out_path)
    print(f"saved {out_path}")
    return final_z, contact_frac


if __name__ == "__main__":
    run()
