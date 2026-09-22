"""Panda arm + table + cup + bin MuJoCo simulation."""
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


if __name__ == "__main__":
    sim = Sim()
    print(f"nq={sim.model.nq} nv={sim.model.nv} nu={sim.model.nu}")
    print("actuators:", sim.actuator_names())
    print("joints:", sim.joint_names())
    print("initial cup pos:", sim.cup_position())
