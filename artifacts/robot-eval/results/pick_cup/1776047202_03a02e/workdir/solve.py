import numpy as np
import mujoco

from sim import Sim


def _orientation_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    # Both are 3x3, body->world rotation matrices.
    return 0.5 * (
        np.cross(R_cur[:, 0], R_des[:, 0])
        + np.cross(R_cur[:, 1], R_des[:, 1])
        + np.cross(R_cur[:, 2], R_des[:, 2])
    )


class PandaCupPolicy:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.m = sim.model
        self.d = sim.data

        self.hand_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.left_finger_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        self.right_finger_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )
        self.cup_geom_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom"
        )

        # Arm joint ids / dof indices.
        arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.arm_jids = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in arm_joint_names
        ]
        self.arm_dofs = np.array([self.m.jnt_dofadr[jid] for jid in self.arm_jids], dtype=int)

        # Control ranges for actuators 0..6 correspond to arm joints.
        self.arm_ctrl_min = self.m.actuator_ctrlrange[:7, 0].copy()
        self.arm_ctrl_max = self.m.actuator_ctrlrange[:7, 1].copy()

        # Desired hand orientation: z-axis down, x-axis forward, y-axis left.
        self.R_des = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

        # Internal target state.
        self.q_target = np.zeros(7, dtype=float)
        self.grip_target = 255.0

    def reset_targets_from_state(self):
        self.q_target = self.d.qpos[:7].copy()
        # Keep within actuator ranges.
        self.q_target = np.clip(self.q_target, self.arm_ctrl_min, self.arm_ctrl_max)

    def finger_midpoint(self) -> np.ndarray:
        return 0.5 * (self.d.xpos[self.left_finger_id] + self.d.xpos[self.right_finger_id])

    def _jac_body(self, body_id: int):
        jacp = np.zeros((3, self.m.nv), dtype=float)
        jacr = np.zeros((3, self.m.nv), dtype=float)
        mujoco.mj_jacBody(self.m, self.d, jacp, jacr, body_id)
        return jacp, jacr

    def ik_update(
        self,
        p_mid_des: np.ndarray,
        *,
        kp_pos: float = 2.5,
        kp_rot: float = 0.6,
        damping: float = 0.20,
        step_scale: float = 0.02,
        max_dq: float = 0.6,
    ):
        # Position target is for the midpoint between fingers.
        p_mid = self.finger_midpoint()
        ep = p_mid_des - p_mid

        R_cur = self.d.xmat[self.hand_id].reshape(3, 3)
        er = _orientation_error(R_cur, self.R_des)

        jacp_l, _ = self._jac_body(self.left_finger_id)
        jacp_r, _ = self._jac_body(self.right_finger_id)
        jacp_mid = 0.5 * (jacp_l + jacp_r)

        _, jacr_hand = self._jac_body(self.hand_id)

        J = np.vstack([jacp_mid[:, self.arm_dofs], jacr_hand[:, self.arm_dofs]])
        e = np.concatenate([kp_pos * ep, kp_rot * er])

        # Damped least squares.
        A = J @ J.T + (damping**2) * np.eye(6)
        dq = J.T @ np.linalg.solve(A, e)
        dq = np.clip(dq, -max_dq, max_dq)

        self.q_target = np.clip(
            self.q_target + step_scale * dq, self.arm_ctrl_min, self.arm_ctrl_max
        )

    def set_gripper(self, value: float):
        self.grip_target = float(np.clip(value, 0.0, 255.0))

    def apply_ctrl(self):
        ctrl = self.d.ctrl
        ctrl[:7] = self.q_target
        ctrl[7] = self.grip_target

    def step_with_ik(self, p_mid_des: np.ndarray, n: int, gripper: float | None = None):
        if gripper is not None:
            self.set_gripper(gripper)
        for _ in range(n):
            self.ik_update(p_mid_des)
            self.apply_ctrl()
            self.sim.step(1)


def _finger_cup_contact(sim: Sim) -> bool:
    m, d = sim.model, sim.data
    cup_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")
    left_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    finger_geoms = set(np.where((m.geom_bodyid == left_body) | (m.geom_bodyid == right_body))[0].tolist())
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 == cup_geom and g2 in finger_geoms:
            return True
        if g2 == cup_geom and g1 in finger_geoms:
            return True
    return False


def evaluate_after_settle(sim: Sim, settle_steps: int = 500):
    contacts = 0
    for _ in range(settle_steps):
        sim.step(1)
        if _finger_cup_contact(sim):
            contacts += 1
    cup_z = float(sim.cup_position()[2])
    return cup_z, contacts / float(settle_steps)


def run():
    sim = Sim()
    pol = PandaCupPolicy(sim)

    # Start in a reasonable "ready" configuration (smoothly).
    q_home = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8], dtype=float)
    pol.reset_targets_from_state()
    pol.set_gripper(255)

    # Blend to home over ~3s.
    for t in range(1500):
        alpha = (t + 1) / 1500.0
        pol.q_target = np.clip(
            (1 - alpha) * pol.q_target + alpha * q_home, pol.arm_ctrl_min, pol.arm_ctrl_max
        )
        pol.apply_ctrl()
        sim.step(1)

    cup = sim.cup_position()
    # Waypoints for finger-midpoint position.
    above = np.array([cup[0], cup[1], 0.60], dtype=float)
    pregrasp = np.array([cup[0], cup[1], 0.54], dtype=float)
    grasp = np.array([cup[0], cup[1], 0.48], dtype=float)
    lift = np.array([cup[0], cup[1], 0.72], dtype=float)

    # Move above cup, descend, close, lift, then hold.
    pol.step_with_ik(above, n=2200, gripper=255)
    pol.step_with_ik(pregrasp, n=1600, gripper=255)
    pol.step_with_ik(grasp, n=1400, gripper=255)

    # Close gripper gradually while staying at grasp.
    for k in range(800):
        g = 255 - (255 - 25) * (k + 1) / 800.0
        pol.step_with_ik(grasp, n=1, gripper=g)

    # Lift while keeping grip closed.
    pol.step_with_ik(lift, n=2600, gripper=25)
    pol.step_with_ik(lift, n=1200, gripper=25)

    # Quick self-eval (includes additional steps, but preserves ctrl trace).
    cup_z, contact_frac = evaluate_after_settle(sim, settle_steps=500)
    print(f"after-settle: cup_z={cup_z:.3f} contact_frac={contact_frac:.3f}")

    sim.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")


if __name__ == "__main__":
    run()
