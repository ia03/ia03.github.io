"""Panda arm + table + cup + obstacle wall MuJoCo simulation."""
import os
import tempfile
import numpy as np
import mujoco
import argparse
import time

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
PANDA_XML = os.path.join(MENAGERIE, "franka_emika_panda", "panda.xml")

TABLE_CENTER = (0.5, 0.0, 0.2)
TABLE_HALF_SIZE = (0.26, 0.26, 0.2)
CUP_RADIUS = 0.022
CUP_HALF_HEIGHT = 0.035
CUP_MASS = 0.05
CUP_FRICTION = (2.0, 0.5, 0.1)
CUP_INIT_POS = (0.55, 0.15, 0.4 + CUP_HALF_HEIGHT)
SUCCESS_CUP_Z = 0.56
WALL_CENTER = (0.49, 0.0, 0.48)
WALL_SIZE = (0.015, 0.11, 0.045)
TRACE_EVERY_STEPS = 10


def build_spec():
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.85, 0.85, 0.85, 1])
    table = world.add_body(name="table", pos=list(TABLE_CENTER))
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TABLE_HALF_SIZE), rgba=[0.55, 0.35, 0.2, 1])
    wall = world.add_body(name="wall")
    wall.add_geom(name="wall_geom", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(WALL_CENTER), size=list(WALL_SIZE), rgba=[0.8, 0.8, 0.85, 1])
    cup = world.add_body(name="cup", pos=list(CUP_INIT_POS))
    cup.add_freejoint()
    cup.add_geom(name="cup_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_RADIUS, CUP_HALF_HEIGHT, 0], mass=CUP_MASS, friction=list(CUP_FRICTION), rgba=[0.2, 0.5, 0.9, 1])
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


def _quat_from_mat(mat3):
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.asarray(mat3, dtype=float).reshape(9))
    return quat


def _quat_conj(quat):
    quat = np.asarray(quat, dtype=float)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def _quat_mul(q1, q2):
    out = np.zeros(4, dtype=float)
    mujoco.mju_mulQuat(out, np.asarray(q1, dtype=float), np.asarray(q2, dtype=float))
    return out


def _orientation_error_vec(R_des, R_cur):
    q_des = _quat_from_mat(R_des)
    q_cur = _quat_from_mat(R_cur)
    q_err = _quat_mul(q_des, _quat_conj(q_cur))
    vel = np.zeros(3, dtype=float)
    mujoco.mju_quat2Vel(vel, q_err, 1.0)
    return vel


def solve_ik_hand(
    model,
    data_seed,
    hand_body_id,
    target_pos,
    target_rot,
    max_iters=80,
    pos_weight=1.0,
    rot_weight=0.25,
    damping=1e-2,
):
    """Damped-least-squares IK on the 7 arm joints to reach a hand body pose."""
    data = mujoco.MjData(model)
    data.qpos[:] = data_seed.qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    joint_min = model.actuator_ctrlrange[:7, 0].copy()
    joint_max = model.actuator_ctrlrange[:7, 1].copy()
    q = data.qpos[:7].copy()

    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        cur_pos = np.array(data.xpos[hand_body_id], dtype=float)
        cur_rot = np.array(data.xmat[hand_body_id], dtype=float).reshape(3, 3)

        pos_err = np.asarray(target_pos, dtype=float) - cur_pos
        rot_err = _orientation_error_vec(target_rot, cur_rot)
        err6 = np.concatenate([pos_weight * pos_err, rot_weight * rot_err], dtype=float)

        if np.linalg.norm(pos_err) < 2e-3 and np.linalg.norm(rot_err) < 3e-2:
            break

        mujoco.mj_jacBody(model, data, jacp, jacr, hand_body_id)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        A = J @ J.T + (damping**2) * np.eye(6)
        dq = (J.T @ np.linalg.solve(A, err6)).reshape(-1)
        dq = np.clip(dq, -0.15, 0.15)

        q = np.clip(q + dq, joint_min, joint_max)
        data.qpos[:7] = q

    return q


def solve_ik_hand_pos(model, data_seed, hand_body_id, target_pos, max_iters=120, damping=1e-2):
    """Damped-least-squares IK on hand position only (7 arm joints)."""
    data = mujoco.MjData(model)
    data.qpos[:] = data_seed.qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    joint_min = model.actuator_ctrlrange[:7, 0].copy()
    joint_max = model.actuator_ctrlrange[:7, 1].copy()
    q = data.qpos[:7].copy()

    jacp = np.zeros((3, model.nv), dtype=float)
    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        cur_pos = np.array(data.xpos[hand_body_id], dtype=float)
        pos_err = np.asarray(target_pos, dtype=float) - cur_pos
        if np.linalg.norm(pos_err) < 2e-3:
            break

        mujoco.mj_jacBody(model, data, jacp, None, hand_body_id)
        J = jacp[:, :7]
        A = J @ J.T + (damping**2) * np.eye(3)
        dq = (J.T @ np.linalg.solve(A, pos_err)).reshape(-1)
        dq = np.clip(dq, -0.2, 0.2)
        q = np.clip(q + dq, joint_min, joint_max)
        data.qpos[:7] = q

    return q


def solve_ik_hand_pos_best(
    model,
    data_seed,
    hand_body_id,
    target_pos,
    num_random_seeds=10,
    max_iters=160,
    damping=1e-2,
):
    """Try multiple initial joint seeds and return the best position IK solution."""
    joint_min = model.actuator_ctrlrange[:7, 0].copy()
    joint_max = model.actuator_ctrlrange[:7, 1].copy()

    rng = np.random.default_rng(0)
    seeds = [np.array(data_seed.qpos[:7], dtype=float), np.zeros(7, dtype=float)]
    for _ in range(int(num_random_seeds)):
        seeds.append(rng.uniform(joint_min, joint_max))

    best = None
    best_err = float("inf")
    target = np.asarray(target_pos, dtype=float)

    for q0 in seeds:
        data = mujoco.MjData(model)
        data.qpos[:] = data_seed.qpos
        data.qvel[:] = 0
        data.qpos[:7] = np.clip(q0, joint_min, joint_max)
        mujoco.mj_forward(model, data)

        q = data.qpos[:7].copy()
        jacp = np.zeros((3, model.nv), dtype=float)
        for _ in range(max_iters):
            mujoco.mj_forward(model, data)
            cur = np.array(data.xpos[hand_body_id], dtype=float)
            err = target - cur
            if np.linalg.norm(err) < 1.5e-3:
                break
            mujoco.mj_jacBody(model, data, jacp, None, hand_body_id)
            J = jacp[:, :7]
            A = J @ J.T + (damping**2) * np.eye(3)
            dq = (J.T @ np.linalg.solve(A, err)).reshape(-1)
            dq = np.clip(dq, -0.25, 0.25)
            q = np.clip(q + dq, joint_min, joint_max)
            data.qpos[:7] = q

        mujoco.mj_forward(model, data)
        final_err = float(np.linalg.norm(np.array(data.xpos[hand_body_id], dtype=float) - target))
        if final_err < best_err:
            best_err = final_err
            best = q.copy()

    return best, best_err


def evaluate_trace(sim: Sim):
    trace = sim._trace
    if not trace:
        return {}
    best_x = float(np.min([t["cup_pos"][0] for t in trace]))
    best_z = float(np.max([t["cup_pos"][2] for t in trace]))
    ever_contact = float(np.max([t["cup_contact"] for t in trace])) > 0.5
    return {
        "final_cup": sim.cup_position(),
        "best_x": best_x,
        "best_z": best_z,
        "ever_contact": ever_contact,
    }


def run_scripted_policy(sim: Sim, save_path="/work/final_state.npz", render_dir=None):
    m, d = sim.model, sim.data
    hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_finger_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    # Desired hand orientation: z down, fingers open/close along world +y/-y.
    R_des = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)

    def _stage(target_hand_pos, gripper_ctrl, steps, snap=False, orient=True):
        if orient:
            q_target = solve_ik_hand(m, d, hand_id, target_hand_pos, R_des)
        else:
            q_target = solve_ik_hand_pos(m, d, hand_id, target_hand_pos)
        q_start = d.ctrl[:7].copy()
        for i in range(int(steps)):
            frac = (i + 1) / float(steps)
            d.ctrl[:7] = (1.0 - frac) * q_start + frac * q_target
            d.ctrl[7] = gripper_ctrl
            sim.step(1)
        if snap and render_dir:
            frame = sim.render()
            os.makedirs(render_dir, exist_ok=True)
            import imageio.v2 as imageio

            imageio.imwrite(os.path.join(render_dir, f"frame_{int(sim.data.time*1000):07d}.png"), frame)

    def _stage_q(q_target, gripper_ctrl, steps, snap=False):
        q_start = d.ctrl[:7].copy()
        for i in range(int(steps)):
            frac = (i + 1) / float(steps)
            d.ctrl[:7] = (1.0 - frac) * q_start + frac * np.asarray(q_target, dtype=float)
            d.ctrl[7] = gripper_ctrl
            sim.step(1)
        if snap and render_dir:
            frame = sim.render()
            os.makedirs(render_dir, exist_ok=True)
            import imageio.v2 as imageio

            imageio.imwrite(os.path.join(render_dir, f"frame_{int(sim.data.time*1000):07d}.png"), frame)

    # Open gripper.
    d.ctrl[:7] = d.qpos[:7]
    d.ctrl[7] = 255
    sim.step(50)

    cup0 = sim.cup_position()

    # High approach from the side (cup is at y~0.15, outside wall y-span).
    _stage([0.45, 0.30, 0.80], 255, 600, snap=True)
    _stage([cup0[0], 0.25, 0.80], 255, 550)
    _stage([cup0[0], cup0[1], 0.67], 255, 450, snap=True)

    # Descend above the cup, then solve for a low reach with a global seed search.
    cup = sim.cup_position()
    _stage([cup[0], cup[1], 0.62], 255, 650, snap=True, orient=False)
    q_low, err_low = solve_ik_hand_pos_best(m, d, hand_id, [cup[0], cup[1], 0.50], num_random_seeds=14)
    _stage_q(q_low, 255, 900, snap=True)

    # Close and squeeze a bit.
    cup = sim.cup_position()
    finger_center = 0.5 * (np.array(d.xpos[left_finger_id]) + np.array(d.xpos[right_finger_id]))
    print(
        "preclose cup:",
        cup.tolist(),
        "finger_center:",
        finger_center.tolist(),
        "dist:",
        float(np.linalg.norm(cup - finger_center)),
        "ik_low_err:",
        float(err_low),
    )
    for g in [220, 170, 120, 60, 20, 0]:
        d.ctrl[7] = g
        sim.step(140)

    # Lift above threshold, then retrieve back to robot side.
    _stage([cup[0], cup[1], 0.82], 0, 800, snap=True)
    _stage([0.36, cup[1], 0.80], 0, 900, snap=True)

    # Hold steady for a bit so final state is stable and in contact.
    d.ctrl[:7] = d.ctrl[:7]
    d.ctrl[7] = 0
    sim.step(900)

    sim.save_final_state(save_path)
    return evaluate_trace(sim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_dir", default=None, help="Optional directory to write debug PNG frames.")
    parser.add_argument("--save_path", default="/work/final_state.npz")
    args = parser.parse_args()

    sim = Sim()
    print(f"nq={sim.model.nq} nv={sim.model.nv} nu={sim.model.nu}")
    print("initial cup pos:", sim.cup_position())
    print(f"target: cup_z >= {SUCCESS_CUP_Z}")
    t0 = time.time()
    metrics = run_scripted_policy(sim, save_path=args.save_path, render_dir=args.render_dir)
    dt = time.time() - t0
    print("metrics:", {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()})
    print(f"saved: {os.path.abspath(args.save_path)} (runtime {dt:.2f}s)")
