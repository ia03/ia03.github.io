import math
from pathlib import Path

import mujoco
import numpy as np

from sim import Sim


CUP_TARGET_Z = 0.54
SAVE_PATH = "/work/final_state.npz"


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def contact_fraction(sim, steps=500):
    model, data = sim.model, sim.data
    cup_bid = body_id(model, "cup")
    finger_bids = {body_id(model, "left_finger"), body_id(model, "right_finger")}
    hits = 0
    for _ in range(steps):
        touching = False
        for i in range(data.ncon):
            con = data.contact[i]
            b1 = model.geom_bodyid[con.geom1]
            b2 = model.geom_bodyid[con.geom2]
            if cup_bid in (b1, b2) and ({b1, b2} & finger_bids):
                touching = True
                break
        hits += int(touching)
        sim.step()
    return hits / steps


def evaluate_ctrl_trace(ctrl_trace):
    sim = Sim()
    for ctrl in ctrl_trace:
        sim.data.ctrl[:] = ctrl
        sim.step()
    frac = contact_fraction(sim, steps=500)
    cup_z = float(sim.cup_position()[2])
    return cup_z, frac


class PandaController:
    def __init__(self):
        self.sim = Sim()
        self.model = self.sim.model
        self.data = self.sim.data
        self.hand_bid = body_id(self.model, "hand")
        self.left_bid = body_id(self.model, "left_finger")
        self.right_bid = body_id(self.model, "right_finger")
        self.arm_q = np.zeros(7)
        self.open_gripper()
        self.set_arm(self.arm_q, steps=5)

    def open_gripper(self):
        self.data.ctrl[7] = 255.0

    def close_gripper(self):
        self.data.ctrl[7] = 0.0

    def finger_midpoint(self):
        return 0.5 * (self.data.xpos[self.left_bid] + self.data.xpos[self.right_bid])

    def hand_pos(self):
        return self.data.xpos[self.hand_bid].copy()

    def finger_gap(self):
        return float(np.linalg.norm(self.data.xpos[self.left_bid] - self.data.xpos[self.right_bid]))

    def finger_mid_jac(self):
        jacp_l = np.zeros((3, self.model.nv))
        jacp_r = np.zeros((3, self.model.nv))
        mujoco.mj_jacBodyCom(self.model, self.data, jacp_l, None, self.left_bid)
        mujoco.mj_jacBodyCom(self.model, self.data, jacp_r, None, self.right_bid)
        return 0.5 * (jacp_l[:, :7] + jacp_r[:, :7])

    def ik_to_midpoint(self, target, iters=200, tol=1e-4):
        q = self.data.qpos[:7].copy()
        q_lo = self.model.jnt_range[:7, 0]
        q_hi = self.model.jnt_range[:7, 1]
        for _ in range(iters):
            self.data.qpos[:7] = q
            self.data.qpos[7:9] = self.data.qpos[7:9]
            mujoco.mj_forward(self.model, self.data)
            err = target - self.finger_midpoint()
            if np.linalg.norm(err) < tol:
                break
            jac = self.finger_mid_jac()
            damping = 1e-3
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), err)
            dq += 0.02 * (self.arm_q - q)
            step = np.clip(dq, -0.08, 0.08)
            q = np.clip(q + step, q_lo, q_hi)
        self.data.qpos[:7] = q
        mujoco.mj_forward(self.model, self.data)
        self.arm_q = q.copy()
        return np.linalg.norm(target - self.finger_midpoint())

    def set_arm(self, target_q, steps=120):
        start = self.data.qpos[:7].copy()
        for alpha in np.linspace(0.0, 1.0, steps):
            q = start * (1.0 - alpha) + target_q * alpha
            self.data.ctrl[:7] = q
            self.sim.step()
        self.arm_q = target_q.copy()

    def hold(self, steps):
        self.data.ctrl[:7] = self.arm_q
        self.sim.step(steps)

    def save_if_best(self, best):
        cup_z, frac = evaluate_ctrl_trace(np.array(self.sim._ctrl_trace))
        score = 0.5 * np.clip((cup_z - 0.435) / (0.52 - 0.435), 0.0, 1.0) + 0.5 * np.clip(frac / 0.5, 0.0, 1.0)
        improved = best is None or score > best[0]
        print(f"eval score={score:.3f} cup_z={cup_z:.3f} contact={frac:.3f} gap={self.finger_gap():.3f}")
        if improved:
            self.sim.save_final_state(SAVE_PATH)
            print(f"saved {SAVE_PATH}")
            return (score, cup_z, frac)
        return best


def run_attempt(descend_z=0.446, close_steps=220, lift_z=0.57, settle_hold=200):
    ctrl = PandaController()
    cup = ctrl.sim.cup_position()

    waypoints = [
        np.array([cup[0], cup[1], cup[2] + 0.14]),
        np.array([cup[0], cup[1], descend_z + 0.05]),
        np.array([cup[0], cup[1], descend_z]),
    ]
    for pt in waypoints:
        err = ctrl.ik_to_midpoint(pt)
        ctrl.set_arm(ctrl.arm_q, steps=160)
        print("move", pt, "err", err, "mid", ctrl.finger_midpoint(), "cup", ctrl.sim.cup_position())

    ctrl.close_gripper()
    ctrl.hold(close_steps)
    print("after close cup", ctrl.sim.cup_position(), "gap", ctrl.finger_gap())

    lift_points = [
        np.array([cup[0], cup[1], cup[2] + 0.18]),
        np.array([cup[0], cup[1], lift_z]),
    ]
    for pt in lift_points:
        err = ctrl.ik_to_midpoint(pt)
        ctrl.set_arm(ctrl.arm_q, steps=200)
        print("lift", pt, "err", err, "mid", ctrl.finger_midpoint(), "cup", ctrl.sim.cup_position())

    ctrl.hold(settle_hold)
    return ctrl


def search():
    best = None
    Path(SAVE_PATH).unlink(missing_ok=True)
    configs = [
        (0.446, 220, 0.56, 200),
        (0.440, 260, 0.57, 250),
        (0.452, 260, 0.56, 250),
        (0.448, 320, 0.58, 300),
        (0.442, 320, 0.60, 300),
    ]
    for cfg in configs:
        print("config", cfg)
        ctrl = run_attempt(*cfg)
        best = ctrl.save_if_best(best)
    return best


if __name__ == "__main__":
    best = search()
    print("best", best)
