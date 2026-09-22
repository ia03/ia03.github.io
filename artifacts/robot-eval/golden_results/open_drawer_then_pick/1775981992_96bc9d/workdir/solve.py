import math
import os

import imageio.v2 as iio
import mujoco
import numpy as np

from sim import Sim


TIP_GEOM_IDS = (70, 78)


class Controller:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.q = np.zeros(7, dtype=float)
        self.ctrl = np.zeros(8, dtype=float)
        self.frames = []
        self.step_idx = 0
        self.reset()

    def reset(self):
        self.sim.reset()
        self.q[:] = self.data.qpos[:7]
        self.ctrl[:7] = self.q
        self.ctrl[7] = 255.0
        self.data.ctrl[:] = self.ctrl
        mujoco.mj_forward(self.model, self.data)
        self.frames = []
        self.step_idx = 0

    def tcp_pos(self):
        return np.mean(self.data.geom_xpos[list(TIP_GEOM_IDS)], axis=0)

    def tcp_jac(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        acc = np.zeros((3, 7))
        for geom_id in TIP_GEOM_IDS:
            jacp[:] = 0
            mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, geom_id)
            acc += jacp[:, :7]
        return acc / len(TIP_GEOM_IDS)

    def set_grip(self, open_amount):
        self.ctrl[7] = float(np.clip(open_amount, 0.0, 255.0))

    def arm_step_toward(self, target, gain=3.0, damp=1e-3, max_dq=0.06):
        err = np.asarray(target) - self.tcp_pos()
        jac = self.tcp_jac()
        lhs = jac @ jac.T + damp * np.eye(3)
        dq = jac.T @ np.linalg.solve(lhs, gain * err)
        dq = np.clip(dq, -max_dq, max_dq)
        self.q += dq
        lo = self.model.actuator_ctrlrange[:7, 0]
        hi = self.model.actuator_ctrlrange[:7, 1]
        self.q = np.clip(self.q, lo, hi)
        self.ctrl[:7] = self.q
        self.data.ctrl[:] = self.ctrl
        self.sim.step()
        self.step_idx += 1
        if self.step_idx % 25 == 0:
            self.frames.append(self.sim.render(320, 240))
        return np.linalg.norm(err)

    def hold(self, steps, grip=None):
        if grip is not None:
            self.set_grip(grip)
        for _ in range(steps):
            self.data.ctrl[:] = self.ctrl
            self.sim.step()
            self.step_idx += 1
            if self.step_idx % 25 == 0:
                self.frames.append(self.sim.render(320, 240))

    def move_tcp(self, target, steps=500, tol=0.01, grip=None, gain=3.0, max_dq=0.06):
        if grip is not None:
            self.set_grip(grip)
        err = None
        for _ in range(steps):
            err = self.arm_step_toward(target, gain=gain, max_dq=max_dq)
            if err < tol:
                break
        return err


def stage_metrics(sim):
    return {
        "time": sim.data.time,
        "drawer": sim.drawer_open_amount(),
        "block": sim.block_position().copy(),
        "contact": sim.has_gripper_block_contact(),
    }


def attempt(sim, handle_y_offset=0.0, pull_x=0.88, pre_q=None, block_offset=(0.0, 0.0, 0.0)):
    ctl = Controller(sim)
    if pre_q is not None:
        ctl.q[:] = np.array(pre_q, dtype=float)
        ctl.ctrl[:7] = ctl.q
        ctl.data.ctrl[:] = ctl.ctrl
        ctl.hold(400, grip=255)

    handle = np.array([0.753, -0.02 + handle_y_offset, 0.44])
    block = np.array([0.615, -0.02, 0.435]) + np.array(block_offset)

    ctl.move_tcp([0.58, -0.02, 0.60], steps=500, tol=0.015, grip=255)
    ctl.move_tcp(handle + [0.00, 0.0, 0.11], steps=700, tol=0.012, grip=255)
    ctl.move_tcp(handle + [0.00, 0.0, 0.04], steps=700, tol=0.009, grip=255)
    ctl.move_tcp(handle + [0.004, 0.0, 0.01], steps=500, tol=0.008, grip=255, gain=2.5, max_dq=0.04)
    ctl.hold(150, grip=30)
    for x in np.linspace(handle[0] + 0.01, pull_x, 7):
        ctl.move_tcp([x, handle[1], handle[2] + 0.01], steps=260, tol=0.01, grip=20, gain=2.2, max_dq=0.04)
    ctl.hold(200, grip=20)
    sim.save_final_state("/work/final_state.npz")

    ctl.move_tcp([0.68, -0.02, 0.60], steps=500, tol=0.02, grip=255)
    ctl.move_tcp(block + [0.0, 0.0, 0.13], steps=700, tol=0.012, grip=255)
    ctl.move_tcp(block + [0.0, 0.0, 0.045], steps=700, tol=0.01, grip=255)
    ctl.move_tcp(block + [0.0, 0.0, 0.005], steps=600, tol=0.008, grip=255, gain=2.5, max_dq=0.035)
    ctl.hold(220, grip=0)
    ctl.move_tcp(block + [0.0, 0.0, 0.22], steps=900, tol=0.012, grip=0, gain=2.5, max_dq=0.03)
    ctl.move_tcp([0.58, -0.02, 0.68], steps=700, tol=0.015, grip=0, gain=2.0, max_dq=0.03)
    ctl.hold(350, grip=0)

    if ctl.frames:
        iio.mimsave("/work/attempt.mp4", ctl.frames, fps=20)
    sim.save_final_state("/work/final_state.npz")
    return stage_metrics(sim)


def main():
    sim = Sim()
    configs = [
        {},
        {"pre_q": [0.0, 0.35, 0.0, -1.9, 0.0, 2.2, 0.8]},
        {"pre_q": [0.2, 0.5, 0.1, -1.8, 0.0, 2.0, 0.7], "handle_y_offset": 0.01},
        {"pre_q": [-0.2, 0.45, -0.2, -1.85, 0.1, 2.05, 0.6], "handle_y_offset": -0.01},
    ]
    best = None
    for i, cfg in enumerate(configs):
        metrics = attempt(sim, **cfg)
        print("attempt", i, metrics)
        score = 4.0 * metrics["drawer"] + metrics["block"][2] + 0.1 * float(metrics["contact"])
        if best is None or score > best[0]:
            best = (score, i, metrics)
            sim.save_final_state("/work/final_state.npz")
            if os.path.exists("/work/attempt.mp4"):
                os.replace("/work/attempt.mp4", "/work/best_attempt.mp4")
    print("best", best)


if __name__ == "__main__":
    main()
