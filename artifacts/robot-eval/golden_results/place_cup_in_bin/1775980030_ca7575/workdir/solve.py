from pathlib import Path

import mujoco
import numpy as np

from sim import BIN_CENTER, Sim


OUT_PATH = Path("/work/final_state.npz")
ARM_DOF = 7
OPEN_GRIPPER = 255.0
CLOSED_GRIPPER = 0.0
LEFT_PAD_GEOM = 69
RIGHT_PAD_GEOM = 77
GRASP_SEED = np.array(
    [1.95107168, 0.3300164, -2.32207346, -1.98667484, 0.24314222, 1.75163401, -0.37236781],
    dtype=float,
)
BIN_SEED = np.array([0.247, 0.775, -0.156, -0.467, 0.116, 1.273, 1.555], dtype=float)


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def weighted_subscore(*components):
    total_weight = sum(weight for _, _, weight in components)
    if total_weight <= 0:
        return 0.0
    return float(sum(clamp01(value) * weight for _, value, weight in components) / total_weight)


class Planner:
    def __init__(self, sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.hand_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.qmin = self.model.actuator_ctrlrange[:ARM_DOF, 0].copy()
        self.qmax = self.model.actuator_ctrlrange[:ARM_DOF, 1].copy()

    def fingertip_midpoint(self):
        return 0.5 * (
            self.data.geom_xpos[LEFT_PAD_GEOM].copy() + self.data.geom_xpos[RIGHT_PAD_GEOM].copy()
        )

    def solve_pose(self, q_init, target_midpoint, iters=240):
        q = q_init.copy()
        for _ in range(iters):
            self.data.qpos[:ARM_DOF] = q
            self.data.qpos[7] = 0.04
            self.data.qpos[8] = 0.04
            mujoco.mj_forward(self.model, self.data)

            midpoint = self.fingertip_midpoint()
            hand_mat = self.data.xmat[self.hand_body].reshape(3, 3)
            pos_err = target_midpoint - midpoint
            z_err = np.cross(hand_mat[:, 2], np.array([0.0, 0.0, -1.0], dtype=float))

            jacp_left = np.zeros((3, self.model.nv))
            jacr_left = np.zeros((3, self.model.nv))
            jacp_right = np.zeros((3, self.model.nv))
            jacr_right = np.zeros((3, self.model.nv))
            mujoco.mj_jac(
                self.model,
                self.data,
                jacp_left,
                jacr_left,
                self.data.geom_xpos[LEFT_PAD_GEOM],
                int(self.model.geom_bodyid[LEFT_PAD_GEOM]),
            )
            mujoco.mj_jac(
                self.model,
                self.data,
                jacp_right,
                jacr_right,
                self.data.geom_xpos[RIGHT_PAD_GEOM],
                int(self.model.geom_bodyid[RIGHT_PAD_GEOM]),
            )
            jacp = 0.5 * (jacp_left[:, :ARM_DOF] + jacp_right[:, :ARM_DOF])

            jacp_hand = np.zeros((3, self.model.nv))
            jacr_hand = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp_hand, jacr_hand, self.hand_body)

            jac = np.vstack((jacp, 0.1 * jacr_hand[:, :ARM_DOF]))
            twist = np.concatenate((2.5 * pos_err, 0.5 * z_err))
            damping = 1e-3 * np.eye(6)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping, twist)
            q = np.clip(q + 0.05 * dq, self.qmin, self.qmax)
        return q

    def goto(self, q_target, gripper, steps):
        for _ in range(steps):
            self.data.ctrl[:ARM_DOF] = q_target
            self.data.ctrl[7] = gripper
            self.sim.step()

    def hold(self, q_target, gripper, steps):
        self.goto(q_target, gripper, steps)


def cup_uprightness(sim):
    quat = sim.data.qpos[10:14].copy()
    quat = quat / np.linalg.norm(quat)
    return float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))


def evaluate_candidate(sim):
    trace = sim._trace
    ctrl_trace = np.array(sim._ctrl_trace, dtype=float).reshape(-1, sim.model.nu)
    if not trace or ctrl_trace.size == 0:
        return False, 0.0, {"error": "empty trace"}

    raw_cup = sim.cup_position().copy()
    raw_upright = cup_uprightness(sim)
    replay_contacts = np.array([entry["cup_contact"] for entry in trace], dtype=float)
    replay_cup = np.array([entry["cup_pos"] for entry in trace], dtype=float)
    replay_ok = len(ctrl_trace) >= 20
    replay_contact = bool(np.any(replay_contacts > 0.5))
    replay_lift = bool(np.max(replay_cup[:, 2]) >= 0.50)
    last10_contact = float(replay_contacts[-10:].mean()) if len(replay_contacts) >= 10 else float(replay_contacts.mean())

    settle = Sim()
    for ctrl in ctrl_trace:
        settle.data.ctrl[:] = ctrl
        settle.step(1)

    settle_contacts = []
    for _ in range(500):
        settle_contacts.append(float(settle.has_gripper_cup_contact()))
        settle.step(1)
    settle_contacts = np.array(settle_contacts, dtype=float)

    cup = settle.cup_position()
    inside = abs(cup[0] - 0.70) <= 0.046 and abs(cup[1] - 0.14) <= 0.046
    high = cup[2] >= 0.42
    upright = cup_uprightness(settle) >= 0.85
    release_replay = last10_contact < 0.5
    release_settle = settle_contacts.mean() < 0.1
    success = all([replay_ok, replay_contact, replay_lift, release_replay, release_settle, inside, high, upright])

    replay_cup_initial = np.array([0.48, -0.12, 0.435], dtype=float)
    final_bin_center_distance = float(np.linalg.norm(cup[:2] - np.array(BIN_CENTER[:2], dtype=float)))
    initial_bin_center_distance = float(np.linalg.norm(replay_cup_initial[:2] - np.array(BIN_CENTER[:2], dtype=float)))
    grasp_progress = 1.0 if replay_contact else 0.0
    lift_progress = clamp01((float(np.max(replay_cup[:, 2])) - 0.435) / (0.50 - 0.435))
    inside_bin_progress = 1.0 if inside else 0.0
    placement_progress = clamp01(1.0 - (final_bin_center_distance / initial_bin_center_distance))
    release_progress = clamp01(1.0 - (float(settle_contacts.mean()) / 0.1))
    upright_progress = clamp01(cup_uprightness(settle) / 0.85)
    efficiency_progress = clamp01(1400 / max(len(ctrl_trace), 1))
    progress_score = weighted_subscore(
        ("grasp_progress", grasp_progress, 0.05),
        ("lift_progress", lift_progress, 0.10),
        ("inside_bin_progress", inside_bin_progress, 0.35),
        ("placement_progress", placement_progress, 0.10),
        ("release_progress", release_progress, 0.10),
        ("upright_progress", upright_progress, 0.25),
        ("efficiency_progress", efficiency_progress, 0.05),
    )

    info = {
        "ctrl_steps": int(len(ctrl_trace)),
        "contact_seen": replay_contact,
        "best_z": float(np.max(replay_cup[:, 2])),
        "last_contact_frac": last10_contact,
        "raw_cup_pos": raw_cup,
        "raw_upright": raw_upright,
        "cup_pos": cup,
        "bin_xy_error": final_bin_center_distance,
        "cup_up_z": cup_uprightness(settle),
        "contact_now": bool(settle.has_gripper_cup_contact()),
        "replay_ok": replay_ok,
        "replay_lift": replay_lift,
        "release_replay": release_replay,
        "release_settle": release_settle,
        "inside": inside,
        "high": high,
        "upright": upright,
        "grasp_progress": grasp_progress,
        "lift_progress": lift_progress,
        "inside_bin_progress": inside_bin_progress,
        "placement_progress": placement_progress,
        "release_progress": release_progress,
        "upright_progress": upright_progress,
        "efficiency_progress": efficiency_progress,
        "progress_score": progress_score,
    }
    return success, progress_score, info


def run_attempt(params):
    sim = Sim()
    planner = Planner(sim)

    sim.data.ctrl[7] = OPEN_GRIPPER
    sim.step(150)

    q_pre = planner.solve_pose(GRASP_SEED.copy(), np.array([0.48, -0.12, 0.55], dtype=float))
    q_grasp = planner.solve_pose(q_pre.copy(), np.array([0.48, -0.12, params["grasp_z"]], dtype=float))
    q_lift = planner.solve_pose(q_grasp.copy(), np.array([0.52, -0.08, 0.56], dtype=float))
    q_mid = planner.solve_pose(BIN_SEED.copy(), np.array([0.60, 0.03, 0.60], dtype=float))
    q_over = planner.solve_pose(q_mid.copy(), np.array([0.70, 0.14, 0.56], dtype=float))
    q_place = planner.solve_pose(q_over.copy(), np.array([0.70, 0.14, params["place_z"]], dtype=float))
    q_retreat = planner.solve_pose(q_place.copy(), np.array([0.60, 0.14, 0.62], dtype=float))

    planner.goto(q_pre, OPEN_GRIPPER, 250)
    planner.goto(q_grasp, OPEN_GRIPPER, 220)
    planner.hold(q_grasp, params["close_grip"], 160)
    planner.goto(q_lift, params["close_grip"], 220)
    planner.goto(q_mid, params["close_grip"], 200)
    planner.goto(q_over, params["close_grip"], 220)
    planner.goto(q_place, params["close_grip"], 180)
    planner.hold(q_place, params["place_grip"], params["place_hold"])
    planner.goto(q_retreat, OPEN_GRIPPER, 220)
    planner.hold(q_retreat, OPEN_GRIPPER, 300)
    return sim


def main():
    candidates = []
    for close_grip in (0.0, 10.0, 20.0):
        for grasp_z in (0.435, 0.445, 0.455, 0.465):
            for place_z in (0.49, 0.51):
                candidates.append(
                    {
                        "close_grip": close_grip,
                        "place_grip": 20.0,
                        "place_hold": 180,
                        "place_z": place_z,
                        "grasp_z": grasp_z,
                    }
                )

    best_score = -1.0
    best_info = None
    best_success = False
    for idx, params in enumerate(candidates):
        sim = run_attempt(params)
        success, score, info = evaluate_candidate(sim)
        print(
            f"candidate={idx} success={success} score={score:.3f} "
            f"inside={info.get('inside')} upright={info.get('upright')} "
            f"raw_upright={info.get('raw_upright'):.3f} settle_upright={info.get('cup_up_z'):.3f} "
            f"best_z={info.get('best_z'):.3f} cup={np.round(info.get('cup_pos'), 4)} "
            f"params={params}"
        )
        if success:
            sim.save_final_state(str(OUT_PATH))
            print(f"saved passing candidate {idx} to {OUT_PATH}")
            return
        if score > best_score:
            best_score = score
            best_info = info
            best_success = success
            sim.save_final_state(str(OUT_PATH))

    print(f"best_success={best_success} best_score={best_score:.3f} best_info={best_info}")


if __name__ == "__main__":
    main()
