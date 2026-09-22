"""Heuristic controller for Pick Cup With Wall.

Runs a simple damped-least-squares IK loop on the Panda hand body, executes a
few waypoints (approach -> grasp -> lift -> retract -> hold), and saves
`/work/final_state.npz` via Sim.save_final_state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


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


def _quat_from_mat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def _ang_err(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Return orientation error as axis-angle (3,) in world frame."""
    q_cur = _quat_from_mat(R_cur)
    q_des = _quat_from_mat(R_des)
    q_err = _quat_mul(q_des, _quat_conj(q_cur))
    if q_err[0] < 0:
        q_err = -q_err
    vec = q_err[1:]
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return np.zeros(3, dtype=float)
    w = float(np.clip(q_err[0], -1.0, 1.0))
    angle = 2.0 * float(np.arctan2(n, w))
    axis = vec / n
    return axis * angle


@dataclass
class IKConfig:
    pos_gain: float = 6.0
    rot_gain: float = 3.0
    damping: float = 0.08
    step_scale: float = 0.8


class PandaIK:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, body_name: str = "hand"):
        self.m = model
        self.d = data
        self.body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.arm_jids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.arm_joint_names]
        self.arm_dofs = [int(self.m.jnt_dofadr[j]) for j in self.arm_jids]
        self.arm_qposadr = [int(self.m.jnt_qposadr[j]) for j in self.arm_jids]
        self.qmin = self.m.actuator_ctrlrange[:7, 0].copy()
        self.qmax = self.m.actuator_ctrlrange[:7, 1].copy()

        self.q_des = self.d.qpos[self.arm_qposadr].copy()

    def _jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.m.nv), dtype=float)
        jacr = np.zeros((3, self.m.nv), dtype=float)
        mujoco.mj_jacBody(self.m, self.d, jacp, jacr, self.body_id)
        Jp = jacp[:, self.arm_dofs]
        Jr = jacr[:, self.arm_dofs]
        return Jp, Jr

    def step_towards(self, pos_des: np.ndarray, R_des: np.ndarray, cfg: IKConfig) -> None:
        pos_cur = self.d.xpos[self.body_id].copy()
        R_cur = self.d.xmat[self.body_id].reshape(3, 3).copy()

        ep = (pos_des - pos_cur) * cfg.pos_gain
        er = _ang_err(R_cur, R_des) * cfg.rot_gain
        e = np.concatenate([ep, er], axis=0)

        Jp, Jr = self._jacobian()
        J = np.vstack([Jp, Jr])

        # Damped least squares: dq = J^T (J J^T + l^2 I)^-1 e
        JJt = J @ J.T
        JJt.flat[:: JJt.shape[0] + 1] += cfg.damping * cfg.damping
        dq = J.T @ np.linalg.solve(JJt, e)

        self.q_des = np.clip(self.q_des + cfg.step_scale * dq, self.qmin, self.qmax)
        self.d.ctrl[:7] = self.q_des


def _R_des_y_aligned() -> np.ndarray:
    """Hand orientation: z down, y along world +y."""
    x = np.array([-1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, y, z])


def solve_ik_kinematic(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_R: np.ndarray,
    q_init: np.ndarray,
    *,
    iters: int = 200,
    cfg: IKConfig | None = None,
) -> np.ndarray:
    """Solve IK by directly updating qpos (no dynamics), returning joint targets."""
    if cfg is None:
        cfg = IKConfig(pos_gain=1.0, rot_gain=0.6, damping=0.12, step_scale=0.6)
    ik = PandaIK(model, data, body_name="hand")
    q = q_init.copy()
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]

    for _ in range(iters):
        data.qpos[ik.arm_qposadr] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        ik.q_des = q
        ik.step_towards(target_pos, target_R, cfg)
        q = np.clip(ik.q_des, qmin, qmax)

    return q


def run() -> None:
    sim = Sim()
    m = sim.model
    d = sim.data
    R_des = _R_des_y_aligned()

    # Gripper: ctrl[7] in [0,255], where 255=open, 0=closed.
    d.ctrl[7] = 255.0

    # Planning data for kinematic IK (keeps execution sim clean).
    plan_d = mujoco.MjData(m)
    mujoco.mj_resetData(m, plan_d)
    mujoco.mj_forward(m, plan_d)

    fast_cfg = IKConfig(pos_gain=1.2, rot_gain=0.5, damping=0.14, step_scale=0.7)

    def exec_cartesian_segment(
        p0: np.ndarray,
        p1: np.ndarray,
        steps: int,
        *,
        grip: float | None = None,
        ik_iters: int = 12,
    ) -> None:
        if grip is not None:
            d.ctrl[7] = float(grip)
        q = d.qpos[:7].copy()
        for i in range(steps):
            a = (i + 1) / steps
            p = (1 - a) * p0 + a * p1
            q = solve_ik_kinematic(m, plan_d, p, R_des, q, iters=ik_iters, cfg=fast_cfg)
            d.ctrl[:7] = q
            sim.step(1)

    cup0 = sim.cup_position().copy()

    # Cartesian waypoints (keeps near-straight paths over the wall).
    wp0 = plan_d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")].copy()
    wp1 = np.array([0.35, 0.00, 0.80])
    wp2 = np.array([0.46, 0.18, 0.78])
    wp3 = np.array([cup0[0], cup0[1], 0.78])
    wp4 = np.array([cup0[0], cup0[1], 0.52])
    wp5 = np.array([cup0[0], cup0[1], 0.485])
    wp6 = np.array([cup0[0], cup0[1], 0.72])
    wp7 = np.array([0.38, cup0[1], 0.72])
    wp8 = np.array([0.32, cup0[1], 0.70])

    exec_cartesian_segment(wp0, wp1, 450, grip=255)
    exec_cartesian_segment(wp1, wp2, 550)
    exec_cartesian_segment(wp2, wp3, 450)

    # Save an early plausible attempt to ensure the file exists.
    sim.save_final_state("/work/final_state.npz")

    # Descend to pre-grasp and grasp.
    exec_cartesian_segment(wp3, wp4, 500, ik_iters=14)
    exec_cartesian_segment(wp4, wp5, 450, ik_iters=18)

    # Close gripper while maintaining pose.
    for g in np.linspace(255, 0, 200):
        d.ctrl[7] = float(g)
        q = solve_ik_kinematic(m, plan_d, wp5, R_des, d.qpos[:7].copy(), iters=8, cfg=fast_cfg)
        d.ctrl[:7] = q
        sim.step(1)

    # Brief squeeze/hold (let fingers settle on the cup).
    for _ in range(300):
        d.ctrl[7] = 0.0
        q = solve_ik_kinematic(m, plan_d, wp5, R_des, d.qpos[:7].copy(), iters=6, cfg=fast_cfg)
        d.ctrl[:7] = q
        sim.step(1)

    # Lift.
    exec_cartesian_segment(wp5, wp6, 650, grip=0, ik_iters=14)

    # Retract to robot side.
    exec_cartesian_segment(wp6, wp7, 900, ik_iters=14)
    exec_cartesian_segment(wp7, wp8, 700, ik_iters=12)
    exec_cartesian_segment(wp8, wp8, 900, grip=0, ik_iters=6)

    # Overwrite with best-so-far.
    sim.save_final_state("/work/final_state.npz")

    # Lightweight diagnostics.
    cup = sim.cup_position()
    best_x = float(np.min([e["cup_pos"][0] for e in sim._trace]))
    best_z = float(np.max([e["cup_pos"][2] for e in sim._trace]))
    print(
        "final time=%.3f steps=%d cup=(%.3f,%.3f,%.3f) contact=%s best_x=%.3f best_z=%.3f"
        % (sim.data.time, len(sim._ctrl_trace), cup[0], cup[1], cup[2], sim.has_gripper_cup_contact(), best_x, best_z)
    )


if __name__ == "__main__":
    t0 = time.time()
    run()
    print("wallclock_sec=%.2f" % (time.time() - t0))
