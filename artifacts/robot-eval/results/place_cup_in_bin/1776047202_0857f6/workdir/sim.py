"""Panda arm + table + cup + bin MuJoCo simulation."""
from __future__ import annotations

import argparse
import os
import tempfile
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
PANDA_XML = os.path.join(MENAGERIE, "franka_emika_panda", "panda.xml")

TABLE_CENTER = (0.5, 0.0, 0.2)
TABLE_HALF_SIZE = (0.26, 0.26, 0.2)

CUP_RADIUS = 0.022
CUP_HALF_HEIGHT = 0.035
CUP_MASS = 0.05
CUP_FRICTION = (2.0, 0.5, 0.1)
CUP_INIT_POS = (0.48, -0.12, 0.4 + CUP_HALF_HEIGHT)

BIN_CENTER = (0.7, 0.14, 0.4)
BIN_INNER_HALF = (0.05, 0.05)
BIN_WALL_THICKNESS = 0.008
BIN_WALL_HEIGHT = 0.06
BIN_FLOOR_THICKNESS = 0.01
TRACE_EVERY_STEPS = 10


def build_spec():
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    world = spec.worldbody
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2, 2, 0.1],
        rgba=[0.85, 0.85, 0.85, 1],
    )
    table = world.add_body(name="table", pos=list(TABLE_CENTER))
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(TABLE_HALF_SIZE),
        rgba=[0.55, 0.35, 0.2, 1],
    )

    cup = world.add_body(name="cup", pos=list(CUP_INIT_POS))
    cup.add_freejoint()
    cup.add_geom(
        name="cup_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[CUP_RADIUS, CUP_HALF_HEIGHT, 0],
        mass=CUP_MASS,
        friction=list(CUP_FRICTION),
        rgba=[0.2, 0.5, 0.9, 1],
    )

    bx, by, bz = BIN_CENTER
    floor_z = bz + BIN_FLOOR_THICKNESS
    wall_z = bz + BIN_WALL_HEIGHT / 2.0
    half_x, half_y = BIN_INNER_HALF
    t = BIN_WALL_THICKNESS
    bin_body = world.add_body(name="bin")
    bin_body.add_geom(
        name="bin_floor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[bx, by, floor_z / 2.0],
        size=[half_x + t, half_y + t, BIN_FLOOR_THICKNESS / 2.0],
        rgba=[0.2, 0.2, 0.2, 1],
    )
    bin_body.add_geom(
        name="bin_wall_left",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[bx - half_x - t / 2.0, by, wall_z],
        size=[t / 2.0, half_y + t, BIN_WALL_HEIGHT / 2.0],
        rgba=[0.2, 0.2, 0.2, 1],
    )
    bin_body.add_geom(
        name="bin_wall_right",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[bx + half_x + t / 2.0, by, wall_z],
        size=[t / 2.0, half_y + t, BIN_WALL_HEIGHT / 2.0],
        rgba=[0.2, 0.2, 0.2, 1],
    )
    bin_body.add_geom(
        name="bin_wall_front",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[bx, by - half_y - t / 2.0, wall_z],
        size=[half_x, t / 2.0, BIN_WALL_HEIGHT / 2.0],
        rgba=[0.2, 0.2, 0.2, 1],
    )
    bin_body.add_geom(
        name="bin_wall_back",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[bx, by + half_y + t / 2.0, wall_z],
        size=[half_x, t / 2.0, BIN_WALL_HEIGHT / 2.0],
        rgba=[0.2, 0.2, 0.2, 1],
    )
    return spec


class Sim:
    def __init__(self):
        self.spec = build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.cup_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        self.cup_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_geom")
        self.finger_geom_ids = self._gripper_geom_ids()
        self.reset()
        self._initial_qpos = self.data.qpos.copy()
        self._trace = []
        self._step_counter = 0
        self._ctrl_trace = []

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._trace = []
        self._step_counter = 0
        self._ctrl_trace = []
        self._record_trace()

    def step(self, n=1):
        for _ in range(n):
            self._ctrl_trace.append(self.data.ctrl.copy())
            mujoco.mj_step(self.model, self.data)
            self._step_counter += 1
            if self._step_counter % TRACE_EVERY_STEPS == 0:
                self._record_trace()

    def cup_position(self):
        return np.array(self.data.xpos[self.cup_body_id])

    def _gripper_geom_ids(self):
        ids = set()
        for body_name in ("left_finger", "right_finger"):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for geom_id in range(self.model.ngeom):
                if int(self.model.geom_bodyid[geom_id]) == body_id:
                    ids.add(geom_id)
        return ids

    def has_gripper_cup_contact(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if self.cup_geom_id in (g1, g2) and (g1 in self.finger_geom_ids or g2 in self.finger_geom_ids):
                return True
        return False

    def _record_trace(self):
        self._trace.append(
            {
                "time": float(self.data.time),
                "cup_pos": self.cup_position(),
                "cup_contact": float(self.has_gripper_cup_contact()),
            }
        )

    def actuator_names(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]

    def joint_names(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]

    def render(self, width=640, height=480, camera=-1):
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def save_final_state(self, path="/work/final_state.npz"):
        if not self._trace or abs(self._trace[-1]["time"] - float(self.data.time)) > 1e-9:
            self._record_trace()
        trace_time = np.array([entry["time"] for entry in self._trace], dtype=float)
        trace_cup = np.array([entry["cup_pos"] for entry in self._trace], dtype=float)
        trace_contact = np.array([entry["cup_contact"] for entry in self._trace], dtype=float)
        payload = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "ctrl_trace": np.array(self._ctrl_trace, dtype=float).reshape(-1, self.model.nu),
            "initial_qpos": self._initial_qpos,
            "cup_pos": self.cup_position(),
            "trace_time": trace_time,
            "trace_cup": trace_cup,
            "trace_contact": trace_contact,
            "time": self.data.time,
        }
        target = os.path.abspath(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz", dir=os.path.dirname(target)) as tmp:
            tmp_path = tmp.name
        try:
            np.savez(tmp_path, **payload)
            os.replace(tmp_path, target)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def _dls_solve(J: np.ndarray, err: np.ndarray, damping: float) -> np.ndarray:
    # Solve dq = J^T (J J^T + λ^2 I)^-1 err
    JJt = J @ J.T
    A = JJt + (damping * damping) * np.eye(JJt.shape[0])
    x = np.linalg.solve(A, err)
    return J.T @ x


class PickPlacePolicy:
    def __init__(self, sim: Sim):
        self.sim = sim
        self.model = sim.model
        self.data = sim.data
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self.hand_id < 0:
            raise RuntimeError("Could not find body 'hand'")
        self.qpos_idx = np.arange(7, dtype=int)  # joints 1..7
        self.ctrl_min = self.model.actuator_ctrlrange[:7, 0].copy()
        self.ctrl_max = self.model.actuator_ctrlrange[:7, 1].copy()
        self.R_des = self.data.xmat[self.hand_id].reshape(3, 3).copy()
        self.hand_to_pinch = self._compute_hand_to_pinch_offset()

    def hand_pos(self) -> np.ndarray:
        return np.array(self.data.xpos[self.hand_id])

    def _compute_hand_to_pinch_offset(self) -> np.ndarray:
        # Approximate pinch point as the lowest-Z finger geometry positions (pads).
        finger_geom_ids: list[int] = []
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name in ("left_finger", "right_finger"):
                finger_geom_ids.append(geom_id)
        if not finger_geom_ids:
            return np.zeros(3, dtype=float)
        pts = np.array(self.data.geom_xpos[finger_geom_ids], dtype=float)
        min_z = float(pts[:, 2].min())
        sel = pts[pts[:, 2] < min_z + 1e-4]
        pinch = sel.mean(axis=0)
        return self.hand_pos() - pinch

    def hand_target_from_pinch(self, pinch_target_pos: np.ndarray) -> np.ndarray:
        return np.array(pinch_target_pos, dtype=float) + self.hand_to_pinch

    def _orientation_error(self, R_cur: np.ndarray) -> np.ndarray:
        # World-frame small-angle error that rotates R_cur toward R_des.
        # e = 0.5 * sum_i (r_cur_i x r_des_i)
        return 0.5 * (
            np.cross(R_cur[:, 0], self.R_des[:, 0])
            + np.cross(R_cur[:, 1], self.R_des[:, 1])
            + np.cross(R_cur[:, 2], self.R_des[:, 2])
        )

    def control_to_hand_target(
        self,
        target_pos: np.ndarray,
        finger_open: float,
        kp_pos: float = 10.0,
        kp_ori: float = 0.6,
        damping: float = 0.03,
        max_dq: float = 0.12,
    ) -> None:
        # Compute a joint-space update via Jacobian transpose DLS and set position targets.
        cur_pos = self.hand_pos()
        pos_err = target_pos - cur_pos

        R_cur = self.data.xmat[self.hand_id].reshape(3, 3)
        ori_err = self._orientation_error(R_cur)

        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])

        err6 = np.concatenate([kp_pos * pos_err, kp_ori * ori_err])
        dq = _dls_solve(J, err6, damping=damping)
        dq = np.clip(dq, -max_dq, max_dq)

        q_cur = self.data.qpos[:7]
        q_tgt = q_cur + dq
        q_tgt = np.clip(q_tgt, self.ctrl_min, self.ctrl_max)

        self.data.ctrl[:7] = q_tgt
        self.data.ctrl[7] = np.clip(finger_open, 0.0, 255.0)

    def ik_solve(
        self,
        target_pos: np.ndarray,
        q_seed: np.ndarray | None = None,
        kp_pos: float = 10.0,
        kp_ori: float = 0.6,
        damping: float = 0.04,
        max_dq: float = 0.12,
        iters: int = 80,
    ) -> np.ndarray:
        tmp = mujoco.MjData(self.model)
        tmp.qpos[:] = self.data.qpos
        tmp.qvel[:] = 0.0
        if q_seed is not None:
            tmp.qpos[:7] = np.array(q_seed, dtype=float).reshape(7)
        mujoco.mj_forward(self.model, tmp)

        target_pos = np.array(target_pos, dtype=float).reshape(3)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        for _ in range(iters):
            cur_pos = np.array(tmp.xpos[self.hand_id])
            pos_err = target_pos - cur_pos
            R_cur = tmp.xmat[self.hand_id].reshape(3, 3)
            ori_err = self._orientation_error(R_cur)

            if np.linalg.norm(pos_err) < 2e-3 and np.linalg.norm(ori_err) < 5e-3:
                break

            jacp.fill(0.0)
            jacr.fill(0.0)
            mujoco.mj_jacBody(self.model, tmp, jacp, jacr, self.hand_id)
            J = np.vstack([jacp[:, :7], jacr[:, :7]])
            err6 = np.concatenate([kp_pos * pos_err, kp_ori * ori_err])
            dq = _dls_solve(J, err6, damping=damping)
            dq = np.clip(dq, -max_dq, max_dq)

            tmp.qpos[:7] = np.clip(tmp.qpos[:7] + dq, self.ctrl_min, self.ctrl_max)
            mujoco.mj_forward(self.model, tmp)

        return np.array(tmp.qpos[:7], dtype=float)


def run_pick_and_place(
    save_path: str = "/work/final_state.npz",
    max_steps: int = 5200,
    early_save_step: int = 300,
    render_debug_dir: str | None = None,
) -> dict:
    sim = Sim()
    policy = PickPlacePolicy(sim)

    cup = sim.cup_position()
    cup_pre = policy.hand_target_from_pinch(cup + np.array([0.0, 0.0, 0.16]))
    cup_grasp = policy.hand_target_from_pinch(cup + np.array([0.0, 0.0, 0.02]))
    cup_lift = policy.hand_target_from_pinch(cup + np.array([0.0, 0.0, 0.28]))

    bin_center = np.array(BIN_CENTER, dtype=float)
    place_pre = policy.hand_target_from_pinch(bin_center + np.array([0.0, 0.0, 0.34]))
    place_down = policy.hand_target_from_pinch(bin_center + np.array([0.0, 0.0, 0.20]))
    retreat = policy.hand_target_from_pinch(bin_center + np.array([-0.10, 0.0, 0.38]))

    finger_open = 255.0
    finger_closed = 0.0

    # Precompute joint-space waypoints with a lightweight IK solve.
    q_ready = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8], dtype=float)
    q_ready = np.clip(q_ready, policy.ctrl_min, policy.ctrl_max)

    q_cup_pre = policy.ik_solve(cup_pre, q_seed=q_ready)
    q_cup_grasp = policy.ik_solve(cup_grasp, q_seed=q_cup_pre)
    q_cup_lift = policy.ik_solve(cup_lift, q_seed=q_cup_grasp)
    q_place_pre = policy.ik_solve(place_pre, q_seed=q_cup_lift)
    q_place_down = policy.ik_solve(place_down, q_seed=q_place_pre)
    q_retreat = policy.ik_solve(retreat, q_seed=q_place_down)

    phase = 0
    contact_seen = False
    best_cup_z = float(sim.cup_position()[2])
    phase_start = 0
    q_phase_start = sim.data.qpos[:7].copy()

    if render_debug_dir:
        os.makedirs(render_debug_dir, exist_ok=True)

    for step in range(max_steps):
        cup_pos = sim.cup_position()
        best_cup_z = max(best_cup_z, float(cup_pos[2]))
        contact = sim.has_gripper_cup_contact()
        contact_seen = contact_seen or contact

        if phase == 0:
            # Move to a reasonable "ready" pose first.
            sim.data.ctrl[:7] = q_ready
            sim.data.ctrl[7] = finger_open
            if step > 500:
                phase = 1
                phase_start = step
                q_phase_start = sim.data.qpos[:7].copy()
        elif phase == 1:
            # Open and move above cup.
            sim.data.ctrl[:7] = q_cup_pre
            sim.data.ctrl[7] = finger_open
            if step - phase_start > 500:
                phase = 2
                phase_start = step
        elif phase == 2:
            # Descend to grasp pose.
            sim.data.ctrl[:7] = q_cup_grasp
            sim.data.ctrl[7] = finger_open
            if step - phase_start > 450:
                phase = 3
                phase_start = step
        elif phase == 3:
            # Close fingers and "squeeze" for contact.
            sim.data.ctrl[:7] = q_cup_grasp
            sim.data.ctrl[7] = finger_closed
            if contact_seen and step - phase_start > 250:
                phase = 4
                phase_start = step
        elif phase == 4:
            # Lift.
            sim.data.ctrl[:7] = q_cup_lift
            sim.data.ctrl[7] = finger_closed
            if best_cup_z >= 0.53 and step - phase_start > 350:
                phase = 5
                phase_start = step
        elif phase == 5:
            # Move above bin.
            sim.data.ctrl[:7] = q_place_pre
            sim.data.ctrl[7] = finger_closed
            if step - phase_start > 650:
                phase = 6
                phase_start = step
        elif phase == 6:
            # Lower into bin.
            sim.data.ctrl[:7] = q_place_down
            sim.data.ctrl[7] = finger_closed
            if step - phase_start > 450:
                phase = 7
                phase_start = step
        elif phase == 7:
            # Release.
            sim.data.ctrl[:7] = q_place_down
            sim.data.ctrl[7] = finger_open
            if step - phase_start > 350:
                phase = 8
                phase_start = step
        elif phase == 8:
            # Move away to reduce residual contact.
            sim.data.ctrl[:7] = q_retreat
            sim.data.ctrl[7] = finger_open
        else:
            sim.data.ctrl[:7] = q_retreat
            sim.data.ctrl[7] = finger_open

        sim.step(1)

        if step == early_save_step:
            sim.save_final_state(save_path)

        if render_debug_dir and step % 250 == 0:
            frame = sim.render(width=480, height=360)
            # Save in a format that's quick to write without extra deps.
            np.save(os.path.join(render_debug_dir, f"frame_{step:05d}.npy"), frame)

    # Leave a little time for the final released cup to settle while arm is away.
    sim.step(400)
    sim.save_final_state(save_path)

    return {
        "saved_to": os.path.abspath(save_path),
        "contact_seen": bool(contact_seen),
        "best_cup_z": float(best_cup_z),
        "final_cup_pos": sim.cup_position().copy(),
        "final_contact": bool(sim.has_gripper_cup_contact()),
        "final_time": float(sim.data.time),
        "phase": int(phase),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", action="store_true", help="Print model info and exit.")
    parser.add_argument("--save", default="/work/final_state.npz", help="Where to write final_state.npz")
    parser.add_argument("--steps", type=int, default=5200, help="Number of control steps to run")
    parser.add_argument("--early-save-step", type=int, default=300, help="Checkpoint save step")
    parser.add_argument("--render-debug-dir", default=None, help="Optional directory to dump .npy frames")
    args = parser.parse_args()

    if args.info:
        sim = Sim()
        print(f"nq={sim.model.nq} nv={sim.model.nv} nu={sim.model.nu}")
        print("actuators:", sim.actuator_names())
        print("joints:", sim.joint_names())
        print("initial cup pos:", sim.cup_position())
        raise SystemExit(0)

    stats = run_pick_and_place(
        save_path=args.save,
        max_steps=args.steps,
        early_save_step=args.early_save_step,
        render_debug_dir=args.render_debug_dir,
    )
    print("run stats:", stats)
