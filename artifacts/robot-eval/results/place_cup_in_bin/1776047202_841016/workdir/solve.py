import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import BIN_CENTER, CUP_INIT_POS, Sim


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, R.reshape(9))
    return quat


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q.astype(float)
    if q[0] < 0:
        q = -q
    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:]
    nv = float(np.linalg.norm(v))
    if nv < 1e-10:
        return 2.0 * v
    angle = 2.0 * np.arctan2(nv, w)
    return angle * (v / nv)


@dataclass
class IKParams:
    kp_pos: float = 6.0
    kp_rot: float = 3.5
    damping: float = 0.08
    max_qvel: float = 1.8
    k_null: float = 0.15


class PandaIK:
    def __init__(
        self,
        sim: Sim,
        body_name: str = "hand",
        params: IKParams | None = None,
        q_neutral: np.ndarray | None = None,
    ):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.params = params or IKParams()
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.q_target = self.data.qpos[:7].copy()
        self.q_neutral = q_neutral.copy() if q_neutral is not None else self.q_target.copy()

        self._jacp = np.zeros((3, self.model.nv), dtype=float)
        self._jacr = np.zeros((3, self.model.nv), dtype=float)

        self.ctrl_min = self.model.actuator_ctrlrange[:7, 0].copy()
        self.ctrl_max = self.model.actuator_ctrlrange[:7, 1].copy()

    def _hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.xpos[self.body_id].copy()
        R = self.data.xmat[self.body_id].reshape(3, 3).copy()
        return pos, R

    def step_towards_position(self, pos_des: np.ndarray):
        pos_cur, _ = self._hand_pose()
        e_p = (pos_des - pos_cur) * self.params.kp_pos
        mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, self.body_id)
        J = self._jacp[:, :7]

        # Damped least-squares: qdot = J^T (J J^T + λI)^-1 e
        JJt = J @ J.T + (self.params.damping**2) * np.eye(3)
        qdot_task = J.T @ np.linalg.solve(JJt, e_p)
        qdot = qdot_task + self.params.k_null * (self.q_neutral - self.q_target)

        qdot_norm = float(np.linalg.norm(qdot))
        if qdot_norm > self.params.max_qvel:
            qdot *= self.params.max_qvel / (qdot_norm + 1e-12)

        dt = float(self.model.opt.timestep)
        self.q_target = self.q_target + qdot * dt
        self.q_target = np.clip(self.q_target, self.ctrl_min, self.ctrl_max)
        self.data.ctrl[:7] = self.q_target

    def step_towards_pose(self, pos_des: np.ndarray, R_des: np.ndarray):
        pos_cur, R_cur = self._hand_pose()
        e_p = (pos_des - pos_cur) * self.params.kp_pos

        q_des = _mat_to_quat(R_des)
        q_cur = _mat_to_quat(R_cur)
        q_err = _quat_mul(q_des, _quat_conj(q_cur))
        e_r = _quat_to_rotvec(q_err) * self.params.kp_rot

        mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, self.body_id)
        J = np.vstack([self._jacp[:, :7], self._jacr[:, :7]])
        e = np.concatenate([e_p, e_r], axis=0)

        JJt = J @ J.T + (self.params.damping**2) * np.eye(6)
        qdot_task = J.T @ np.linalg.solve(JJt, e)
        qdot = qdot_task + self.params.k_null * (self.q_neutral - self.q_target)

        qdot_norm = float(np.linalg.norm(qdot))
        if qdot_norm > self.params.max_qvel:
            qdot *= self.params.max_qvel / (qdot_norm + 1e-12)

        dt = float(self.model.opt.timestep)
        self.q_target = self.q_target + qdot * dt
        self.q_target = np.clip(self.q_target, self.ctrl_min, self.ctrl_max)
        self.data.ctrl[:7] = self.q_target


def run_script(sim: Sim, render_every: int = 0):
    q_neutral = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 2.0, 0.8], dtype=float)
    ik = PandaIK(sim, q_neutral=q_neutral)
    left_finger_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    def pose(x, y, z):
        return np.array([x, y, z], dtype=float)

    cup_xy = np.array(CUP_INIT_POS[:2], dtype=float)
    bin_xy = np.array(BIN_CENTER[:2], dtype=float)

    def finger_midpoint() -> np.ndarray:
        return 0.5 * (sim.data.xpos[left_finger_id] + sim.data.xpos[right_finger_id])

    # Phases (counts are physics steps; dt=0.002s).
    # We command the finger midpoint to follow these waypoints.
    phases: list[tuple[int, np.ndarray, tuple[float, float]]] = [
        (500, pose(0.45, 0.0, 0.80), (255.0, 255.0)),  # lift to safe height
        (900, pose(cup_xy[0], cup_xy[1], 0.80), (255.0, 255.0)),  # above cup (high)
        (900, pose(cup_xy[0], cup_xy[1], 0.445), (255.0, 255.0)),  # descend to grasp height
        (900, pose(cup_xy[0], cup_xy[1], 0.445), (255.0, 0.0)),  # close while holding pose
        (1000, pose(cup_xy[0], cup_xy[1], 0.80), (0.0, 0.0)),  # lift (cup z>=0.5)
        (1300, pose(bin_xy[0], bin_xy[1], 0.80), (0.0, 0.0)),  # carry above bin (high)
        (900, pose(bin_xy[0], bin_xy[1], 0.445), (0.0, 0.0)),  # lower to place
        (900, pose(bin_xy[0], bin_xy[1], 0.445), (0.0, 255.0)),  # open to release
        (900, pose(bin_xy[0], bin_xy[1], 0.80), (255.0, 255.0)),  # retreat upward
        (700, pose(bin_xy[0] - 0.18, bin_xy[1] - 0.18, 0.80), (255.0, 255.0)),  # move away
        (1200, pose(bin_xy[0] - 0.18, bin_xy[1] - 0.18, 0.80), (255.0, 255.0)),  # idle settle
    ]

    frames = []
    # Move to neutral first, then freeze orientation reference.
    for t in range(1400):
        sim.data.ctrl[:7] = q_neutral
        sim.data.ctrl[7] = 255.0
        sim.step(1)
        if render_every and (t % render_every == 0):
            frames.append(sim.render(width=320, height=240))

    R_des = sim.data.xmat[ik.body_id].reshape(3, 3).copy()

    def step_mid_to(mid_des: np.ndarray):
        hand_pos, _ = ik._hand_pose()
        offset = finger_midpoint() - hand_pos
        hand_des = mid_des - offset
        ik.step_towards_pose(hand_des, R_des)

    for phase_idx, (steps, mid_des, (g0, g1)) in enumerate(phases):
        for t in range(steps):
            step_mid_to(mid_des)
            if g0 == g1:
                sim.data.ctrl[7] = float(g1)
            else:
                sim.data.ctrl[7] = float(g0 + (g1 - g0) * (t / max(1, steps - 1)))
            sim.step(1)
            if render_every and (t % render_every == 0) and (phase_idx % 2 == 0):
                frames.append(sim.render(width=320, height=240))

    return frames


def quick_diagnostics(sim: Sim):
    cup = sim.cup_position()
    if sim._trace:
        best_z = float(max(float(entry["cup_pos"][2]) for entry in sim._trace))
        last_contact = float(sim._trace[-1]["cup_contact"])
    else:
        best_z = float(cup[2])
        last_contact = 0.0
    print("final cup pos:", cup, "best_recent_z:", best_z, "last_contact:", last_contact)


def main():
    start = time.time()
    sim = Sim()
    run_script(sim, render_every=0)
    quick_diagnostics(sim)
    sim.save_final_state("/work/final_state.npz")
    print(f"saved /work/final_state.npz (sim_time={sim.data.time:.3f}s, wall={time.time()-start:.2f}s)")


if __name__ == "__main__":
    main()
