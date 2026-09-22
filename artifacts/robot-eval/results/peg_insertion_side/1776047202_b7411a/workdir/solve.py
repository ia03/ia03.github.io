import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=float)
GRIPPER_OPEN = 255.0


@dataclass
class Metrics:
    peg_pos: np.ndarray
    peg_x_axis_alignment: float


def now_utc():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def peg_metrics(sim: Sim) -> Metrics:
    peg_id = sim.peg_body_id
    R = sim.data.xmat[peg_id].reshape(3, 3)
    x_axis = R[:, 0]
    alignment = float(abs(x_axis[0]))
    return Metrics(peg_pos=sim.peg_position(), peg_x_axis_alignment=alignment)


def solve_ik_pos(
    model: mujoco.MjModel,
    qpos_seed: np.ndarray,
    body_id: int,
    target_pos: np.ndarray,
    *,
    max_iters: int = 80,
    tol: float = 5e-4,
    damping: float = 2e-2,
    step_scale: float = 0.8,
) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_seed
    mujoco.mj_forward(model, data)

    nv = model.nv
    jacp = np.zeros((3, nv), dtype=float)

    for _ in range(max_iters):
        cur = np.array(data.xpos[body_id])
        err = target_pos - cur
        if float(np.linalg.norm(err)) < tol:
            break

        mujoco.mj_jacBody(model, data, jacp, None, body_id)
        J = jacp[:, :7]  # arm dofs
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping**2) * np.eye(3), err)
        dq = np.clip(dq, -0.2, 0.2)
        data.qpos[:7] = data.qpos[:7] + step_scale * dq

        # Clip to joint limits.
        for j in range(7):
            jnt_id = j  # joint1..joint7 are first 7 joints
            if model.jnt_limited[jnt_id]:
                lo, hi = model.jnt_range[jnt_id]
                data.qpos[j] = float(np.clip(data.qpos[j], lo, hi))

        mujoco.mj_forward(model, data)

    return data.qpos[:7].copy()


def solve_ik_point(
    model: mujoco.MjModel,
    qpos_seed: np.ndarray,
    body_id: int,
    local_point: np.ndarray,
    target_pos: np.ndarray,
    *,
    max_iters: int = 100,
    tol: float = 7e-4,
    damping: float = 2e-2,
    step_scale: float = 0.9,
) -> np.ndarray:
    """IK for a point fixed in a body frame (position-only)."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos_seed
    mujoco.mj_forward(model, data)

    nv = model.nv
    jacp = np.zeros((3, nv), dtype=float)

    for _ in range(max_iters):
        R = data.xmat[body_id].reshape(3, 3)
        cur = np.array(data.xpos[body_id]) + R @ local_point
        err = target_pos - cur
        if float(np.linalg.norm(err)) < tol:
            break

        mujoco.mj_jac(model, data, jacp, None, cur, body_id)
        J = jacp[:, :7]
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping**2) * np.eye(3), err)
        dq = np.clip(dq, -0.25, 0.25)
        data.qpos[:7] = data.qpos[:7] + step_scale * dq

        for j in range(7):
            jnt_id = j
            if model.jnt_limited[jnt_id]:
                lo, hi = model.jnt_range[jnt_id]
                data.qpos[j] = float(np.clip(data.qpos[j], lo, hi))

        mujoco.mj_forward(model, data)

    return data.qpos[:7].copy()

def hold_ctrl(sim: Sim, q: np.ndarray, steps: int, gripper: float = GRIPPER_OPEN):
    sim.data.ctrl[:7] = q
    sim.data.ctrl[7] = gripper
    sim.step(steps)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def main():
    t0 = time.time()
    sim = Sim()

    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    rf_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    # Fingertip pad center in finger body frame (from panda.xml defaults).
    rf_tip_local = np.array([0.0, 0.0055, 0.0445], dtype=float)

    print(f"[{now_utc()}] start")
    print("initial peg:", peg_metrics(sim).peg_pos)
    print("initial hand:", np.array(sim.data.xpos[hand_id]))

    # Phase 1: move to home-ish pose and let the peg settle on the guide.
    for i in range(5):
        q = lerp(sim.data.qpos[:7], HOME_Q, (i + 1) / 5.0)
        hold_ctrl(sim, q, 120)

    # Let the gripper open fully before insertion.
    hold_ctrl(sim, sim.data.ctrl[:7].copy(), 300)

    def goto_tip(target_pos: np.ndarray, *, steps: int, iters: int = 220) -> np.ndarray:
        q_tgt = solve_ik_point(sim.model, sim.data.qpos.copy(), rf_id, rf_tip_local, target_pos, max_iters=iters)
        hold_ctrl(sim, q_tgt, steps)
        return q_tgt

    # Phase 2: approach above the peg, then descend behind the push tab.
    peg = sim.peg_position()
    above = np.array([peg[0] - 0.085, peg[1], peg[2] + 0.12], dtype=float)
    goto_tip(above, steps=650)

    peg = sim.peg_position()
    behind = np.array([peg[0] - 0.085, peg[1], peg[2] + 0.01], dtype=float)
    goto_tip(behind, steps=900)

    # Phase 3: push into the slot along +x by reducing the behind-offset.
    offsets = [-0.070, -0.055, -0.040, -0.025, -0.010, 0.005]
    for off in offsets:
        peg = sim.peg_position()
        tgt = np.array([peg[0] + off, peg[1], peg[2] + 0.01], dtype=float)
        goto_tip(tgt, steps=600)

    # Phase 4: keep holding while the peg settles inside the slot.
    hold_ctrl(sim, sim.data.ctrl[:7].copy(), 1000)

    m = peg_metrics(sim)
    print(f"[{now_utc()}] pre-save peg pos: {m.peg_pos}, alignment: {m.peg_x_axis_alignment:.3f}")
    sim.save_final_state("/work/final_state.npz")
    print(f"[{now_utc()}] saved /work/final_state.npz (elapsed {time.time() - t0:.2f}s, steps {len(sim._ctrl_trace)})")


if __name__ == "__main__":
    main()
