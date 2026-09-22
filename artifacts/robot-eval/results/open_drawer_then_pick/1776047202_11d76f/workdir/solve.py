"""Controller for the 'Open Drawer Then Pick' task.

Runs the MuJoCo sim from the canonical start state, applies a simple staged
controller (open drawer -> reach -> grasp -> lift), and saves the control trace
to /work/final_state.npz for the grader to replay.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to wxyz quaternion."""
    # Robust-ish conversion for proper rotations.
    m00, m01, m02 = float(R[0, 0]), float(R[0, 1]), float(R[0, 2])
    m10, m11, m12 = float(R[1, 0]), float(R[1, 1]), float(R[1, 2])
    m20, m21, m22 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=float)
    # Normalize and enforce qw >= 0 for a stable sign convention.
    q = q / (np.linalg.norm(q) + 1e-12)
    if q[0] < 0:
        q *= -1.0
    return q


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to rotation vector (axis * angle)."""
    q = q / (np.linalg.norm(q) + 1e-12)
    if q[0] < 0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    v = q[1:4]
    vnorm = float(np.linalg.norm(v))
    if vnorm < 1e-10:
        return np.zeros(3, dtype=float)
    angle = 2.0 * float(np.arctan2(vnorm, w))
    axis = v / vnorm
    return axis * angle


@dataclass
class ControllerConfig:
    drawer_target_open: float = 0.135
    drawer_kp: float = 10.0
    drawer_kd: float = 1.2
    drawer_settle_steps: int = 220
    approach_steps: int = 700
    descend_steps: int = 800
    close_steps: int = 420
    lift_steps: int = 900
    hold_steps: int = 450
    gripper_open: float = 255.0
    gripper_closed: float = 0.0
    above_dz: float = 0.16
    grasp_dz: float = -0.01
    lift_dz: float = 0.26
    ik_pos_gain: float = 0.9
    ik_rot_gain: float = 0.0
    ik_damping: float = 0.02
    ik_step_scale: float = 0.55


class Controller:
    def __init__(self, sim: Sim, cfg: ControllerConfig):
        self.sim = sim
        self.cfg = cfg
        m = sim.model
        self.hand_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.left_finger_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        # Keep the initial hand frame around as a weak preference (optional).
        self.R_des = sim.data.xmat[self.hand_body].reshape(3, 3).copy()
        self.q_des = _mat_to_quat(self.R_des)

        # Compute the offset between the hand origin and the finger midpoint (in the hand frame)
        # at a known gripper opening. We'll keep orientation fixed so this stays useful.
        self._finger_mid_offset_hand = self._compute_finger_mid_offset_hand()

        # Arm joint limits from actuators (actuator1..7 match joints1..7).
        self._q_min = sim.model.actuator_ctrlrange[:7, 0].copy()
        self._q_max = sim.model.actuator_ctrlrange[:7, 1].copy()

        self._jacp = np.zeros((3, sim.model.nv), dtype=float)
        self._jacr = np.zeros((3, sim.model.nv), dtype=float)
        self._drawer_dofadr = int(sim.model.jnt_dofadr[sim.drawer_joint_id])

    def _finger_mid_world(self) -> np.ndarray:
        d = self.sim.data
        return 0.5 * (d.xpos[self.left_finger_body] + d.xpos[self.right_finger_body])

    def _compute_finger_mid_offset_hand(self) -> np.ndarray:
        # Set a stable pose, open gripper, and forward once.
        self.sim.data.ctrl[:7] = self.sim.data.qpos[:7]
        self.sim.data.ctrl[7] = self.cfg.gripper_open
        self.sim.data.ctrl[8] = 0.0
        self.sim.step(10)
        hand_pos = self.sim.data.xpos[self.hand_body].copy()
        R = self.sim.data.xmat[self.hand_body].reshape(3, 3).copy()
        mid = self._finger_mid_world()
        return R.T @ (mid - hand_pos)

    def _desired_hand_pos_for_mid(self, mid_world: np.ndarray) -> np.ndarray:
        # Use the *current* hand orientation so we can let IK rotate freely while
        # still targeting the finger midpoint consistently.
        R_cur = self.sim.data.xmat[self.hand_body].reshape(3, 3)
        return mid_world - R_cur @ self._finger_mid_offset_hand

    def _ik_arm_target(self, hand_pos_des: np.ndarray) -> np.ndarray:
        """One damped least-squares IK update producing an arm joint target."""
        m, d = self.sim.model, self.sim.data
        cur_pos = d.xpos[self.hand_body].copy()
        pos_err = hand_pos_des - cur_pos

        mujoco.mj_jacBody(m, d, self._jacp, self._jacr, self.hand_body)
        Jp = self._jacp[:, :7]

        if self.cfg.ik_rot_gain > 0.0:
            cur_R = d.xmat[self.hand_body].reshape(3, 3).copy()
            cur_q = _mat_to_quat(cur_R)
            q_err = _quat_mul(self.q_des, _quat_conj(cur_q))
            rot_err = _quat_to_rotvec(q_err)
            e = np.concatenate(
                [self.cfg.ik_pos_gain * pos_err, self.cfg.ik_rot_gain * rot_err], dtype=float
            )
            Jr = self._jacr[:, :7]
            J = np.vstack([Jp, Jr])  # 6x7
            JJt = J @ J.T
            A = JJt + (self.cfg.ik_damping**2) * np.eye(6, dtype=float)
            dq = J.T @ np.linalg.solve(A, e)
        else:
            e = self.cfg.ik_pos_gain * pos_err
            JJt = Jp @ Jp.T
            A = JJt + (self.cfg.ik_damping**2) * np.eye(3, dtype=float)
            dq = Jp.T @ np.linalg.solve(A, e)

        dq = np.clip(dq, -0.35, 0.35)
        q_next = d.qpos[:7].copy() + self.cfg.ik_step_scale * dq
        return np.clip(q_next, self._q_min, self._q_max)

    def _set_ctrl(self, q_arm: np.ndarray, grip: float, drawer_ctrl: float):
        self.sim.data.ctrl[:7] = q_arm
        self.sim.data.ctrl[7] = float(np.clip(grip, 0.0, 255.0))
        self.sim.data.ctrl[8] = float(np.clip(drawer_ctrl, -1.0, 1.0))

    def _drawer_pd(self, target_open: float) -> float:
        q = self.sim.drawer_open_amount()
        qd = float(self.sim.data.qvel[self._drawer_dofadr])
        u = self.cfg.drawer_kp * (target_open - q) - self.cfg.drawer_kd * qd
        return float(np.clip(u, -1.0, 1.0))

    def run(self, save_path: str = "/work/final_state.npz", early_save: bool = True):
        cfg = self.cfg

        def log(tag: str, step: int):
            if step % 50 != 0:
                return
            drawer = self.sim.drawer_open_amount()
            block = self.sim.block_position()
            contact = self.sim.has_gripper_block_contact()
            mid = self._finger_mid_world()
            print(
                f"{tag:>10} step={step:4d} t={self.sim.data.time:6.3f} "
                f"drawer={drawer:6.3f} block_z={block[2]:6.3f} mid_z={mid[2]:6.3f} contact={int(contact)}"
            )

        # Stage 1: open drawer to a moderate amount (keeps block reachable).
        for i in range(cfg.drawer_settle_steps):
            drawer_u = self._drawer_pd(cfg.drawer_target_open)
            self._set_ctrl(self.sim.data.qpos[:7].copy(), cfg.gripper_open, drawer_u)
            self.sim.step(1)
            log("open", i)

        # Optional early save to guarantee a replay file exists.
        if early_save:
            self.sim.save_final_state(save_path)
            print(f"Early save -> {save_path}")

        # Stage 2: reach above the block.
        for i in range(cfg.approach_steps):
            block = self.sim.block_position()
            mid_des = block + np.array([0.0, 0.0, cfg.above_dz], dtype=float)
            hand_pos_des = self._desired_hand_pos_for_mid(mid_des)
            q_arm = self._ik_arm_target(hand_pos_des)
            self._set_ctrl(q_arm, cfg.gripper_open, self._drawer_pd(cfg.drawer_target_open))
            self.sim.step(1)
            log("approach", i)

        # Stage 3: descend to grasp height (smoothly).
        block0 = self.sim.block_position()
        mid_hi = block0 + np.array([0.0, 0.0, cfg.above_dz], dtype=float)
        mid_lo = block0 + np.array([0.0, 0.0, cfg.grasp_dz], dtype=float)
        for i in range(cfg.descend_steps):
            alpha = (i + 1) / cfg.descend_steps
            block = self.sim.block_position()
            # Track the block in x/y while interpolating z.
            mid_des = np.array([block[0], block[1], (1 - alpha) * mid_hi[2] + alpha * mid_lo[2]], dtype=float)
            hand_pos_des = self._desired_hand_pos_for_mid(mid_des)
            q_arm = self._ik_arm_target(hand_pos_des)
            self._set_ctrl(q_arm, cfg.gripper_open, self._drawer_pd(cfg.drawer_target_open))
            self.sim.step(1)
            log("descend", i)

        # Stage 4: close gripper while holding.
        for i in range(cfg.close_steps):
            block = self.sim.block_position()
            mid_des = block + np.array([0.0, 0.0, cfg.grasp_dz], dtype=float)
            hand_pos_des = self._desired_hand_pos_for_mid(mid_des)
            q_arm = self._ik_arm_target(hand_pos_des)
            grip = (1 - (i + 1) / cfg.close_steps) * cfg.gripper_open + ((i + 1) / cfg.close_steps) * cfg.gripper_closed
            self._set_ctrl(q_arm, grip, self._drawer_pd(cfg.drawer_target_open))
            self.sim.step(1)
            log("close", i)

        # Stage 5: lift.
        lift_mid0 = self._finger_mid_world().copy()
        for i in range(cfg.lift_steps):
            dz = cfg.lift_dz * ((i + 1) / cfg.lift_steps)
            mid_des = lift_mid0 + np.array([0.0, 0.0, dz], dtype=float)
            hand_pos_des = self._desired_hand_pos_for_mid(mid_des)
            q_arm = self._ik_arm_target(hand_pos_des)
            self._set_ctrl(q_arm, cfg.gripper_closed, self._drawer_pd(cfg.drawer_target_open))
            self.sim.step(1)
            log("lift", i)

        # Stage 6: hold to let things settle (still under our final controls).
        for i in range(cfg.hold_steps):
            mid_des = lift_mid0 + np.array([0.0, 0.0, cfg.lift_dz], dtype=float)
            hand_pos_des = self._desired_hand_pos_for_mid(mid_des)
            q_arm = self._ik_arm_target(hand_pos_des)
            self._set_ctrl(q_arm, cfg.gripper_closed, self._drawer_pd(cfg.drawer_target_open))
            self.sim.step(1)
            log("hold", i)

        self.sim.save_final_state(save_path)
        print(f"Final save -> {save_path}")


def main():
    t0 = time.time()
    sim = Sim()
    cfg = ControllerConfig()
    ctrl = Controller(sim, cfg)
    ctrl.run("/work/final_state.npz", early_save=True)
    print(f"Done in {time.time() - t0:.2f}s. drawer={sim.drawer_open_amount():.3f} block_z={sim.block_position()[2]:.3f}")


if __name__ == "__main__":
    main()
