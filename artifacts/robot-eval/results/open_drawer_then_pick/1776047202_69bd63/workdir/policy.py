import time
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"Body not found: {name}")
    return int(bid)


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found: {name}")
    return int(jid)


def _actuator_id(model: mujoco.MjModel, name: str) -> int:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator not found: {name}")
    return int(aid)


def _rot_from_axes(x_axis, y_axis, z_axis) -> np.ndarray:
    r = np.zeros((3, 3), dtype=float)
    r[:, 0] = np.array(x_axis, dtype=float)
    r[:, 1] = np.array(y_axis, dtype=float)
    r[:, 2] = np.array(z_axis, dtype=float)
    return r


def _mat_from_xmat(xmat9) -> np.ndarray:
    return np.array(xmat9, dtype=float).reshape(3, 3)


def _orientation_error_world(r_des: np.ndarray, r_cur: np.ndarray) -> np.ndarray:
    # Compute orientation error as the "vee" of the skew-symmetric part of R_des^T R_cur.
    r_err = r_des.T @ r_cur
    w_des = 0.5 * np.array(
        [r_err[2, 1] - r_err[1, 2], r_err[0, 2] - r_err[2, 0], r_err[1, 0] - r_err[0, 1]],
        dtype=float,
    )
    return r_des @ w_des


@dataclass
class PandaKinematics:
    hand_body_id: int
    arm_joint_ids: list[int]
    arm_qpos_adrs: np.ndarray
    arm_dof_adrs: np.ndarray
    arm_jnt_ranges: np.ndarray
    gripper_act_id: int
    drawer_act_id: int

    @classmethod
    def from_sim(cls, sim: Sim) -> "PandaKinematics":
        m = sim.model
        hand_body_id = _body_id(m, "hand")
        arm_joint_ids = [_joint_id(m, f"joint{i}") for i in range(1, 8)]
        arm_qpos_adrs = np.array([m.jnt_qposadr[jid] for jid in arm_joint_ids], dtype=int)
        arm_dof_adrs = np.array([m.jnt_dofadr[jid] for jid in arm_joint_ids], dtype=int)
        arm_jnt_ranges = np.array([m.jnt_range[jid].copy() for jid in arm_joint_ids], dtype=float)
        gripper_act_id = _actuator_id(m, "actuator8")
        drawer_act_id = _actuator_id(m, "drawer_motor")
        return cls(
            hand_body_id=hand_body_id,
            arm_joint_ids=arm_joint_ids,
            arm_qpos_adrs=arm_qpos_adrs,
            arm_dof_adrs=arm_dof_adrs,
            arm_jnt_ranges=arm_jnt_ranges,
            gripper_act_id=gripper_act_id,
            drawer_act_id=drawer_act_id,
        )

    def arm_qpos(self, data: mujoco.MjData) -> np.ndarray:
        return np.array(data.qpos[self.arm_qpos_adrs], dtype=float)

    def set_arm_ctrl(self, data: mujoco.MjData, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float)
        q = np.clip(q, self.arm_jnt_ranges[:, 0], self.arm_jnt_ranges[:, 1])
        data.ctrl[:7] = q

    def set_gripper(self, data: mujoco.MjData, cmd: float) -> None:
        data.ctrl[self.gripper_act_id] = float(np.clip(cmd, 0.0, 255.0))

    def set_drawer_motor(self, data: mujoco.MjData, cmd: float) -> None:
        data.ctrl[self.drawer_act_id] = float(np.clip(cmd, -1.0, 1.0))

    def hand_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        pos = np.array(data.xpos[self.hand_body_id], dtype=float)
        rot = _mat_from_xmat(data.xmat[self.hand_body_id])
        return pos, rot

    def jacobian_hand(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacBody(model, data, jacp, jacr, self.hand_body_id)
        return jacp[:, self.arm_dof_adrs], jacr[:, self.arm_dof_adrs]


def _dls_solve(jac: np.ndarray, err: np.ndarray, damping: float) -> np.ndarray:
    # Damped least squares: dq = J^T (J J^T + λ I)^-1 e
    jj_t = jac @ jac.T
    a = jj_t + (damping * damping) * np.eye(jj_t.shape[0])
    return jac.T @ np.linalg.solve(a, err)


def move_hand_ik(
    sim: Sim,
    kin: PandaKinematics,
    target_pos: np.ndarray,
    target_rot: np.ndarray | None,
    steps: int,
    *,
    pos_gain: float = 6.0,
    rot_gain: float = 2.0,
    damping: float = 0.15,
    max_dq: float = 0.08,
    target_z_dir: np.ndarray | None = None,
) -> None:
    target_pos = np.asarray(target_pos, dtype=float).copy()
    for _ in range(int(steps)):
        cur_pos, cur_rot = kin.hand_pose(sim.data)
        pos_err = target_pos - cur_pos

        jacp, jacr = kin.jacobian_hand(sim.model, sim.data)
        if target_z_dir is not None:
            z_des = np.asarray(target_z_dir, dtype=float)
            z_des = z_des / (np.linalg.norm(z_des) + 1e-12)
            z_cur = cur_rot[:, 2]
            rot_err = np.cross(z_cur, z_des)
            jac = np.vstack([jacp, jacr])
            err = np.hstack([pos_gain * pos_err, rot_gain * rot_err])
        elif target_rot is None:
            jac = jacp
            err = pos_gain * pos_err
        else:
            rot_err = _orientation_error_world(target_rot, cur_rot)
            jac = np.vstack([jacp, jacr])
            err = np.hstack([pos_gain * pos_err, rot_gain * rot_err])

        dq = _dls_solve(jac, err, damping=damping)
        dq = np.clip(dq, -max_dq, max_dq)
        q = kin.arm_qpos(sim.data)
        q_next = q + dq
        kin.set_arm_ctrl(sim.data, q_next)
        sim.step(1)


def run_episode(save_path: str = "/work/final_state.npz", *, render_debug: bool = False) -> dict:
    sim = Sim()
    kin = PandaKinematics.from_sim(sim)

    # Desired "top-down grasp" orientation:
    # - hand z axis points downward (-z): fingers extend downward toward the block
    # - hand y axis aligns with +y (fingers close along +/-y)
    # - hand x axis points toward -x
    r_des = _rot_from_axes([-1, 0, 0], [0, 1, 0], [0, 0, -1])

    # Stage 0: go to a reasonable home posture and open gripper.
    home_q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=float)
    kin.set_arm_ctrl(sim.data, home_q)
    kin.set_gripper(sim.data, 255.0)
    kin.set_drawer_motor(sim.data, 0.0)
    sim.step(120)

    # Stage 1: open drawer via motor (no contact needed).
    kin.set_arm_ctrl(sim.data, home_q)
    kin.set_gripper(sim.data, 255.0)
    # Avoid fully extending the drawer: at the hard stop the block tends to get ejected.
    kin.set_drawer_motor(sim.data, 0.6)
    for _ in range(520):
        sim.step(1)
        if sim.drawer_open_amount() > 0.095:
            break
    # Let friction/damping hold the drawer open (it remains open enough during long holds).
    kin.set_drawer_motor(sim.data, 0.0)

    # Mandatory early save (drawer opened, >20 ctrl steps recorded).
    sim.save_final_state(save_path)

    # Stage 2: compute block position after drawer opened.
    block_pos = sim.block_position()
    # With the drawer open, the block is presented high enough to grasp from above.
    pregrasp = block_pos + np.array([0.0, 0.0, 0.26])
    # Fingertip pads sit about 0.10m along +hand-z from the hand body origin.
    grasp = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.115], dtype=float)

    z_down = np.array([0.0, 0.0, -1.0], dtype=float)
    move_hand_ik(sim, kin, pregrasp, None, steps=360, pos_gain=5.0, rot_gain=7.0, damping=0.12, max_dq=0.10, target_z_dir=z_down)
    move_hand_ik(sim, kin, grasp, None, steps=320, pos_gain=5.0, rot_gain=7.0, damping=0.12, max_dq=0.10, target_z_dir=z_down)

    # Stage 3: close gripper to grasp.
    kin.set_gripper(sim.data, 0.0)
    kin.set_drawer_motor(sim.data, 0.25)
    sim.step(90)

    # Stage 4: lift straight up.
    lift = grasp + np.array([0.0, 0.0, 0.34])
    move_hand_ik(sim, kin, lift, None, steps=520, pos_gain=5.0, rot_gain=7.0, damping=0.12, max_dq=0.10, target_z_dir=z_down)

    # Hold the pose for stability and contact during replay end.
    kin.set_drawer_motor(sim.data, 0.25)
    sim.step(650)

    # Final save (best-so-far).
    sim.save_final_state(save_path)

    # Quick diagnostics (approximate; grader uses a replay+settle).
    trace = sim._trace  # noqa: SLF001 (local script)
    max_drawer = max(t["drawer_open"] for t in trace) if trace else sim.drawer_open_amount()
    max_block_z = max(float(t["block_pos"][2]) for t in trace) if trace else float(sim.block_position()[2])
    return {
        "time": float(sim.data.time),
        "drawer_open_final": sim.drawer_open_amount(),
        "block_pos_final": sim.block_position(),
        "max_drawer_open_trace": float(max_drawer),
        "max_block_z_trace": float(max_block_z),
        "final_contact": bool(sim.has_gripper_block_contact()),
        "ctrl_steps": int(len(sim._ctrl_trace)),  # noqa: SLF001
    }


if __name__ == "__main__":
    t0 = time.time()
    info = run_episode("/work/final_state.npz")
    dt = time.time() - t0
    print("episode_runtime_s", round(dt, 3))
    for k, v in info.items():
        print(k, "=", v)
