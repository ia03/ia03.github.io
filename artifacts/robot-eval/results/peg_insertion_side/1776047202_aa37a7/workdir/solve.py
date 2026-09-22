import numpy as np
import mujoco

from sim import Sim


def _body_pos(model, data, body_name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return np.array(data.xpos[bid])


def _peg_axes(model, data, peg_body_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = data.xmat[peg_body_id].reshape(3, 3)
    return R[:, 0].copy(), R[:, 1].copy(), R[:, 2].copy()


def _dls_step(J: np.ndarray, err: np.ndarray, damping: float = 1e-2) -> np.ndarray:
    # J: 3x7, err: 3
    A = J @ J.T + (damping**2) * np.eye(3)
    return J.T @ np.linalg.solve(A, err)


def _clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def run(seed: int = 0, save_path: str = "/work/final_state.npz", render_debug: bool = False):
    rng = np.random.default_rng(seed)
    sim = Sim()
    m, d = sim.model, sim.data

    # Arm joints are the first 7 qpos entries; controls map 1:1 to desired joint angles.
    arm_qpos_idx = np.arange(7)
    arm_ctrl_idx = np.arange(7)
    gripper_ctrl_idx = 7

    # Actuator ctrl ranges (for first 7 are joint position targets).
    ctrl_lo = m.actuator_ctrlrange[arm_ctrl_idx, 0].copy()
    ctrl_hi = m.actuator_ctrlrange[arm_ctrl_idx, 1].copy()
    # Some actuators don't declare ctrlrange; treat as unbounded.
    ctrl_lo = np.where(np.isfinite(ctrl_lo), ctrl_lo, -np.inf)
    ctrl_hi = np.where(np.isfinite(ctrl_hi), ctrl_hi, np.inf)

    hand_body = "hand"
    hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, hand_body)
    peg_id = sim.peg_body_id

    # Finger geom ids for crude contact-point estimation.
    left_finger_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    finger_geom_ids = [i for i in range(m.ngeom) if m.geom_bodyid[i] in (left_finger_id, right_finger_id)]

    # Start with gripper closed for a stiffer pushing surface.
    d.ctrl[:] = 0.0
    d.ctrl[gripper_ctrl_idx] = 0.0

    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    ik_data = mujoco.MjData(m)

    def ik_solve_hand_pos(target_pos: np.ndarray, q_seed: np.ndarray, iters: int = 60) -> np.ndarray:
        # Kinematic-only damped least squares on a scratch MjData.
        q = q_seed.copy()
        for _ in range(iters):
            ik_data.qpos[:] = d.qpos
            ik_data.qvel[:] = 0.0
            ik_data.qpos[arm_qpos_idx] = q
            mujoco.mj_forward(m, ik_data)
            err = target_pos - ik_data.xpos[hand_id].copy()
            if np.linalg.norm(err) < 5e-4:
                break
            mujoco.mj_jacBody(m, ik_data, jacp, jacr, hand_id)
            dq = _dls_step(jacp[:, :7], err, damping=2e-2)
            dq = np.clip(dq, -0.25, 0.25)
            q = _clamp(q + 0.9 * dq, ctrl_lo, ctrl_hi)
        return q

    def goto_hand_pos(target_pos: np.ndarray, settle_steps: int = 350):
        q0 = d.qpos[arm_qpos_idx].copy()
        q_tgt = ik_solve_hand_pos(target_pos, q0, iters=80)
        d.ctrl[arm_ctrl_idx] = q_tgt
        d.ctrl[gripper_ctrl_idx] = 0.0
        sim.step(settle_steps)

    def hold(steps: int):
        for _ in range(steps):
            d.ctrl[gripper_ctrl_idx] = 0.0
            sim.step(1)

    def peg_metrics():
        pos = sim.peg_position()
        peg_x, _, _ = _peg_axes(m, d, peg_id)
        align = float(abs(np.dot(peg_x, np.array([1.0, 0.0, 0.0]))))
        return pos, align

    # Phase 0: Move to a reasonable "home-ish" joint posture to avoid singularity.
    home = np.array([0.0, 0.0, 0.0, -1.55, 0.0, 1.55, -0.78])
    home = _clamp(home, ctrl_lo, ctrl_hi)
    d.ctrl[arm_ctrl_idx] = home
    hold(400)

    # Compute tab position from peg pose (tab is at -0.05m along peg local x).
    def tab_world_pos():
        peg_pos = sim.peg_position()
        peg_x, _, _ = _peg_axes(m, d, peg_id)
        return peg_pos + (-0.050) * peg_x

    def hand_to_finger_offsets() -> tuple[float, float]:
        # Returns (hand_x - finger_front_x, hand_z - finger_bottom_z) in world coordinates.
        hand_pos = d.xpos[hand_id].copy()
        xs = [d.geom_xpos[i][0] for i in finger_geom_ids]
        zs = [d.geom_xpos[i][2] for i in finger_geom_ids]
        finger_front_x = float(max(xs))
        finger_bottom_z = float(min(zs))
        return float(hand_pos[0] - finger_front_x), float(hand_pos[2] - finger_bottom_z)

    # Approach from above and slightly behind the tab (keep hand higher than contact point).
    tab0 = tab_world_pos()
    x_off_hand_to_front, z_off_hand_to_bottom = hand_to_finger_offsets()
    # Put finger front slightly behind tab, with a gentle downward bias (bottom a touch below tab center).
    desired_finger_front_x = tab0[0] - 0.030
    desired_finger_bottom_z = tab0[2] - 0.002
    approach_high = np.array(
        [
            desired_finger_front_x + x_off_hand_to_front,
            tab0[1],
            desired_finger_bottom_z + z_off_hand_to_bottom + 0.08,
        ]
    )
    approach_low = np.array(
        [
            desired_finger_front_x + x_off_hand_to_front,
            tab0[1],
            desired_finger_bottom_z + z_off_hand_to_bottom + 0.02,
        ]
    )

    goto_hand_pos(approach_high, settle_steps=450)
    # Recompute offsets after large motion for better z placement.
    x_off_hand_to_front, z_off_hand_to_bottom = hand_to_finger_offsets()
    approach_low[2] = desired_finger_bottom_z + z_off_hand_to_bottom + 0.01
    goto_hand_pos(approach_low, settle_steps=450)

    # Phase 1: Push forward in +x along the guide lane.
    # Use an absolute finger-front x ramp to avoid accidentally "following" the tab and pulling backwards.
    tab_lane = tab_world_pos()
    lane_y = float(tab_lane[1])
    lane_z = float(tab_lane[2])
    x_off_hand_to_front, z_off_hand_to_bottom = hand_to_finger_offsets()
    finger_front_x_start = float(tab_lane[0] - 0.030)
    finger_front_x_end = float(tab_lane[0] + 0.095)
    n_waypoints = 140
    for i in range(n_waypoints):
        frac = (i + 1) / n_waypoints
        y_dither = 0.0010 * np.sin(2 * np.pi * frac * 1.2) + rng.normal(0.0, 0.00010)
        desired_finger_front_x = finger_front_x_start + (finger_front_x_end - finger_front_x_start) * frac
        desired_finger_bottom_z = lane_z - 0.003
        target = np.array(
            [
                desired_finger_front_x + x_off_hand_to_front,
                lane_y + y_dither,
                desired_finger_bottom_z + z_off_hand_to_bottom,
            ]
        )
        q_tgt = ik_solve_hand_pos(target, d.qpos[arm_qpos_idx].copy(), iters=50)
        d.ctrl[arm_ctrl_idx] = q_tgt
        d.ctrl[gripper_ctrl_idx] = 0.0
        sim.step(25)

        pos, align = peg_metrics()
        if pos[0] >= 0.520 and abs(pos[1] - (-0.10)) <= 0.03 and abs(pos[2] - 0.52) <= 0.03 and align >= 0.5:
            break

    # Phase 2: Hold to let contacts settle with the peg inserted.
    hold(900)

    # Save final state (includes ctrl_trace).
    sim.save_final_state(save_path)
    return sim


if __name__ == "__main__":
    run(render_debug=False)
