import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


def utc_now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def rot_error(R_cur, R_des):
    # Small-angle orientation error (world frame).
    return 0.5 * (
        np.cross(R_cur[:, 0], R_des[:, 0])
        + np.cross(R_cur[:, 1], R_des[:, 1])
        + np.cross(R_cur[:, 2], R_des[:, 2])
    )


@dataclass
class ControllerGains:
    k_pos: float = 10.0
    k_ori: float = 3.0
    damping: float = 0.06
    dq_max: float = 0.05
    substeps: int = 10


class PandaStacker:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.m = sim.model
        self.d = sim.data

        self.hand_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.lf_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.rf_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        # Offset from hand body origin to finger midpoint along hand +Z axis (hand frame),
        # estimated at the default configuration.
        self.grip_offset = 0.0584

        self.ctrl_lo = self.m.actuator_ctrlrange[:7, 0].copy()
        self.ctrl_hi = self.m.actuator_ctrlrange[:7, 1].copy()

        self.R_des = self._hand_R().copy()

    def _hand_pos(self):
        return np.array(self.d.xpos[self.hand_id])

    def _hand_R(self):
        return np.array(self.d.xmat[self.hand_id]).reshape(3, 3)

    def grip_mid_world(self):
        R = self._hand_R()
        return self._hand_pos() + R[:, 2] * self.grip_offset

    def _jac_hand(self):
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBody(self.m, self.d, jacp, jacr, self.hand_id)
        return jacp[:, :7].copy(), jacr[:, :7].copy()

    def _jac_grip_mid(self):
        # Linear Jacobian at the grip midpoint (between fingers).
        # v_point = v_hand + omega_hand x r
        Jp_hand, Jr = self._jac_hand()
        r = self._hand_R()[:, 2] * self.grip_offset
        Jp = Jp_hand.copy()
        for i in range(7):
            Jp[:, i] += np.cross(Jr[:, i], r)
        return Jp, Jr

    def _solve_dls(self, J, err, damping):
        # dq = J^T (J J^T + λ^2 I)^-1 err
        A = J @ J.T + (damping * damping) * np.eye(J.shape[0])
        return J.T @ np.linalg.solve(A, err)

    def servo_to_mid(self, target_mid, gripper, n_ticks, gains: ControllerGains):
        for _ in range(n_ticks):
            R = self._hand_R()
            mid = self.grip_mid_world()
            pos_err = target_mid - mid
            ori_err = rot_error(R, self.R_des)

            err = np.concatenate([gains.k_pos * pos_err, gains.k_ori * ori_err])
            Jp, Jr = self._jac_grip_mid()
            J = np.vstack([Jp, Jr])
            dq = self._solve_dls(J, err, gains.damping)
            dq = clamp(dq, -gains.dq_max, gains.dq_max)

            q = self.d.qpos[:7].copy()
            q_tgt = clamp(q + dq, self.ctrl_lo, self.ctrl_hi)

            self.d.ctrl[:7] = q_tgt
            self.d.ctrl[7] = float(gripper)
            self.sim.step(gains.substeps)

    def hold(self, gripper, steps, q_target=None):
        if q_target is None:
            q_target = self.d.qpos[:7].copy()
        self.d.ctrl[:7] = clamp(q_target, self.ctrl_lo, self.ctrl_hi)
        self.d.ctrl[7] = float(gripper)
        self.sim.step(steps)

    def goto_joints(self, q_target, gripper, steps=1500):
        q_target = np.asarray(q_target, dtype=float).reshape(7)
        self.hold(gripper, steps=steps, q_target=q_target)


def run_attempt(out_path="/work/final_state.npz", render_debug=False):
    sim = Sim()
    stacker = PandaStacker(sim)

    # Use a consistent desired hand orientation from the canonical start pose.
    sim.reset()
    open_g = 255.0
    close_g = 0.0

    # Move to a safer "home" pose to avoid scraping the table.
    q_home = np.array([0.0, -0.6, 0.0, -2.3, 0.0, 1.7, 0.8], dtype=float)
    stacker.goto_joints(q_home, open_g, steps=1800)
    stacker.R_des = stacker._hand_R().copy()

    gains_fast = ControllerGains(k_pos=12.0, k_ori=4.0, damping=0.07, dq_max=0.06, substeps=10)
    gains_slow = ControllerGains(k_pos=7.0, k_ori=3.0, damping=0.08, dq_max=0.04, substeps=10)

    red0 = sim.block_positions()["red"].copy()
    green0 = sim.block_positions()["green"].copy()

    # 1) Open and move above red.
    above_red = red0 + np.array([0.0, 0.0, 0.20])
    stacker.servo_to_mid(above_red, open_g, n_ticks=80, gains=gains_fast)

    # 2) Pregrasp: align above, then descend to grasp height.
    pregrasp_mid = red0 + np.array([0.0, 0.0, 0.08])
    stacker.servo_to_mid(pregrasp_mid, open_g, n_ticks=70, gains=gains_slow)
    grasp_mid = red0 + np.array([0.0, 0.0, 0.0])
    stacker.servo_to_mid(grasp_mid, open_g, n_ticks=80, gains=gains_slow)

    # 3) Close to grasp and wait.
    stacker.servo_to_mid(grasp_mid, close_g, n_ticks=40, gains=gains_slow)
    stacker.hold(close_g, steps=200)

    # 4) Lift.
    lift_mid = red0 + np.array([0.0, 0.0, 0.22])
    stacker.servo_to_mid(lift_mid, close_g, n_ticks=90, gains=gains_fast)
    stacker.hold(close_g, steps=100)

    # Estimate grasp offset between commanded mid and actual red center while held.
    red_pos = sim.block_positions()["red"].copy()
    mid_pos = stacker.grip_mid_world().copy()
    grasp_offset = red_pos - mid_pos

    # 5) Move above green.
    above_green = green0 + np.array([0.0, 0.0, 0.24])
    stacker.servo_to_mid(above_green - grasp_offset, close_g, n_ticks=120, gains=gains_fast)
    stacker.hold(close_g, steps=100)

    # 6) Place: target red center at green + 0.055m.
    place_red = green0 + np.array([0.0, 0.0, 0.055])
    place_mid = place_red - grasp_offset
    stacker.servo_to_mid(place_mid, close_g, n_ticks=120, gains=gains_slow)
    stacker.hold(close_g, steps=150)

    # 7) Release cleanly: open, then rise.
    stacker.hold(open_g, steps=250)
    retreat_mid = place_mid + np.array([0.0, 0.0, 0.18])
    stacker.servo_to_mid(retreat_mid, open_g, n_ticks=80, gains=gains_fast)

    # 8) Let blocks settle.
    stacker.hold(open_g, steps=2200)

    # Mimic grader settle window for diagnostics.
    stacker.hold(open_g, steps=500)
    pos = sim.block_positions()
    dx, dy, dz = (pos["red"] - pos["green"]).tolist()
    print(f"[{utc_now_str()}] dx={dx:+.4f} dy={dy:+.4f} dz={dz:+.4f} steps={len(sim._ctrl_trace)}")

    sim.save_final_state(out_path)
    print(f"saved {out_path}")

    if render_debug:
        # Save a couple debug frames for quick sanity-checking.
        import imageio.v2 as iio

        frame = sim.render(640, 480)
        iio.imwrite("/work/debug_frame.png", frame)
        print("wrote /work/debug_frame.png")


if __name__ == "__main__":
    run_attempt()
