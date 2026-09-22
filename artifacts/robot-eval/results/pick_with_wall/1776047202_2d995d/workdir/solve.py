import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim, SUCCESS_CUP_Z


def _utc_now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def _hand_body_id(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")


PAD_GEOM_LEFT = 69
PAD_GEOM_RIGHT = 77


def _pinch_pos(data):
    return 0.5 * (data.geom_xpos[PAD_GEOM_LEFT] + data.geom_xpos[PAD_GEOM_RIGHT])


def ik_pinch_to_pos(model, qpos_seed, target_pos, *, n_iters=120, damping=3e-3, step_size=0.9):
    """
    Solve for arm joint positions (first 7 DOF) so that the gripper pinch point
    (midpoint between the two pad geoms) matches target_pos.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_seed
    data.qvel[:] = 0

    jacp_l = np.zeros((3, model.nv), dtype=float)
    jacr_l = np.zeros((3, model.nv), dtype=float)
    jacp_r = np.zeros((3, model.nv), dtype=float)
    jacr_r = np.zeros((3, model.nv), dtype=float)

    q = data.qpos[:7].copy()
    for _ in range(n_iters):
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        pinch = _pinch_pos(data)
        err = np.asarray(target_pos, dtype=float) - pinch
        if float(np.linalg.norm(err)) < 2e-3:
            break

        mujoco.mj_jacGeom(model, data, jacp_l, jacr_l, PAD_GEOM_LEFT)
        mujoco.mj_jacGeom(model, data, jacp_r, jacr_r, PAD_GEOM_RIGHT)
        J = 0.5 * (jacp_l[:, :7] + jacp_r[:, :7])
        JJt = J @ J.T
        JJt.flat[::4] += damping * damping
        dq = J.T @ np.linalg.solve(JJt, err)
        q = q + step_size * dq

    return q


def clamp_to_ctrlrange(model, q):
    q = np.asarray(q, dtype=float).copy()
    for i in range(7):
        lo, hi = model.actuator_ctrlrange[i]
        q[i] = float(np.clip(q[i], lo, hi))
    return q


@dataclass(frozen=True)
class Stage:
    name: str
    pinch_pos: tuple[float, float, float]
    grip: float
    steps: int


def run_episode(sim: Sim):
    m = sim.model
    # Current state seed for IK.
    qpos_seed = sim.data.qpos.copy()

    # A simple over-the-top plan:
    # - move high above wall toward cup
    # - descend behind wall to grasp height
    # - close gripper and let settle
    # - lift above success height
    # - carry back over wall to robot side
    # - hold still for stability
    stages = [
        Stage("open_gripper", (0.30, 0.00, 0.75), grip=1.0, steps=140),
        Stage("approach_high", (0.48, 0.15, 0.70), grip=1.0, steps=260),
        Stage("pregrasp", (0.55, 0.15, 0.55), grip=1.0, steps=260),
        Stage("descend", (0.55, 0.15, 0.455), grip=1.0, steps=300),
        Stage("close_grip", (0.55, 0.15, 0.455), grip=0.10, steps=280),
        Stage("squeeze", (0.55, 0.15, 0.455), grip=0.00, steps=240),
        Stage("lift", (0.55, 0.15, 0.70), grip=0.00, steps=360),
        Stage("return_high", (0.38, 0.15, 0.70), grip=0.00, steps=420),
        Stage("tuck", (0.34, 0.15, 0.72), grip=0.00, steps=520),
        Stage("final_hold", (0.34, 0.15, 0.72), grip=0.00, steps=700),
    ]

    last_q_des = sim.data.qpos[:7].copy()
    for stage in stages:
        q_des = ik_pinch_to_pos(m, qpos_seed, stage.pinch_pos)
        q_des = clamp_to_ctrlrange(m, q_des)
        for t in range(stage.steps):
            blend = min(1.0, (t + 1) / 60.0)
            sim.data.ctrl[:7] = (1 - blend) * last_q_des + blend * q_des
            sim.data.ctrl[7] = stage.grip * 255.0
            sim.step(1)
        last_q_des = q_des
        qpos_seed = sim.data.qpos.copy()
        pinch = _pinch_pos(sim.data).copy()
        err = np.linalg.norm(pinch - np.asarray(stage.pinch_pos))
        print(
            f"stage={stage.name:>12s} pinch={pinch} err={err:.3f} "
            f"cup={sim.cup_position()} contact={sim.has_gripper_cup_contact()}"
        )


def summarize(sim: Sim):
    cup = sim.cup_position()
    best_z = max(float(entry["cup_pos"][2]) for entry in sim._trace)
    best_x = min(float(entry["cup_pos"][0]) for entry in sim._trace)
    ever_contact = any(bool(entry["cup_contact"]) for entry in sim._trace)
    print(
        f"final cup pos {cup} | best_z={best_z:.3f} best_x={best_x:.3f} ever_contact={ever_contact}"
    )


def main():
    print(f"[{_utc_now_str()}] starting")
    sim = Sim()
    run_episode(sim)
    summarize(sim)
    sim.save_final_state("/work/final_state.npz")
    print(f"[{_utc_now_str()}] saved /work/final_state.npz (steps={len(sim._ctrl_trace)})")


if __name__ == "__main__":
    main()
