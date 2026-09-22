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


def evaluate(sim):
    cup_pos = sim.cup_position().copy()
    cup_mat = sim.data.xmat[sim.cup_body_id].reshape(3, 3)
    trace_contact = np.array([entry["cup_contact"] for entry in sim._trace], dtype=float)
    trace_cup = np.array([entry["cup_pos"] for entry in sim._trace], dtype=float)
    return {
        "ctrl_steps": len(sim._ctrl_trace),
        "contact_seen": bool(np.any(trace_contact > 0.5)),
        "best_z": float(np.max(trace_cup[:, 2])),
        "last_contact_frac": float(np.mean(trace_contact[-10:])),
        "cup_pos": cup_pos,
        "bin_xy_error": float(np.linalg.norm(cup_pos[:2] - np.array(BIN_CENTER[:2]))),
        "cup_up_z": float(cup_mat[2, 2]),
        "contact_now": bool(sim.has_gripper_cup_contact()),
    }


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


def run_attempt():
    sim = Sim()
    planner = Planner(sim)

    sim.data.ctrl[7] = OPEN_GRIPPER
    sim.step(150)
    sim.save_final_state(str(OUT_PATH))

    q_pre = planner.solve_pose(GRASP_SEED.copy(), np.array([0.48, -0.12, 0.55], dtype=float))
    q_grasp = planner.solve_pose(q_pre.copy(), np.array([0.48, -0.12, 0.435], dtype=float))
    q_lift = planner.solve_pose(q_grasp.copy(), np.array([0.52, -0.08, 0.56], dtype=float))
    q_mid = planner.solve_pose(BIN_SEED.copy(), np.array([0.60, 0.03, 0.60], dtype=float))
    q_over = planner.solve_pose(q_mid.copy(), np.array([0.70, 0.14, 0.56], dtype=float))
    q_place = planner.solve_pose(q_over.copy(), np.array([0.70, 0.14, 0.47], dtype=float))
    q_retreat = planner.solve_pose(q_place.copy(), np.array([0.60, 0.14, 0.62], dtype=float))

    planner.goto(q_pre, OPEN_GRIPPER, 250)
    planner.goto(q_grasp, OPEN_GRIPPER, 220)
    sim.save_final_state(str(OUT_PATH))

    planner.hold(q_grasp, CLOSED_GRIPPER, 160)
    planner.goto(q_lift, CLOSED_GRIPPER, 220)
    planner.goto(q_mid, CLOSED_GRIPPER, 200)
    planner.goto(q_over, CLOSED_GRIPPER, 220)
    planner.goto(q_place, CLOSED_GRIPPER, 180)
    planner.hold(q_place, OPEN_GRIPPER, 120)
    planner.goto(q_retreat, OPEN_GRIPPER, 220)
    planner.hold(q_retreat, OPEN_GRIPPER, 300)

    sim.save_final_state(str(OUT_PATH))
    return sim


if __name__ == "__main__":
    sim = run_attempt()
    print(evaluate(sim))
