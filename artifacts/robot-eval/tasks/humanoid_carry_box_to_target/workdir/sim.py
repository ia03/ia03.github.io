"""Unitree H1 humanoid carrying a large box to a floor target."""
import os
import tempfile
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
H1_SCENE_XML = os.path.join(MENAGERIE, "unitree_h1", "scene.xml")

BOX_SIZE = (0.08, 0.05, 0.07)
BOX_MASS = 0.6
BOX_INIT_POS = (0.14, 0.0, 0.88)
TARGET_XY = np.array([0.25, 0.0])
TARGET_RADIUS = 0.20
MIN_PELVIS_Z = 0.74
MIN_TORSO_UP = 0.50
MIN_BOX_Z = 0.72
MAX_BOX_TO_PELVIS = 0.42
TRACE_EVERY_STEPS = 10


def build_spec():
    spec = mujoco.MjSpec.from_file(H1_SCENE_XML)
    world = spec.worldbody
    pedestal = world.add_body(name="pedestal")
    pedestal.add_geom(
        name="pedestal_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[BOX_INIT_POS[0], BOX_INIT_POS[1], 0.40],
        size=[0.12, 0.10, 0.40],
        rgba=[0.45, 0.45, 0.5, 1],
    )
    box = world.add_body(name="carry_box", pos=list(BOX_INIT_POS))
    box.add_freejoint()
    box.add_geom(
        name="carry_box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(BOX_SIZE),
        mass=BOX_MASS,
        friction=[1.3, 0.4, 0.1],
        rgba=[0.85, 0.65, 0.15, 1],
    )
    world.add_geom(
        name="target_marker",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[float(TARGET_XY[0]), float(TARGET_XY[1]), 0.005],
        size=[TARGET_RADIUS, 0.005, 0],
        contype=0,
        conaffinity=0,
        rgba=[0.15, 0.7, 0.25, 0.30],
    )
    return spec


class Sim:
    def __init__(self):
        self.spec = build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "carry_box")
        self.box_joint_id = int(self.model.body_jntadr[self.box_body_id])
        self._home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.reset()
        self._initial_qpos = self.data.qpos.copy()
        self._trace = []
        self._step_counter = 0
        self._ctrl_trace = []

    def reset(self):
        if self._home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        box_qpos_adr = self.model.jnt_qposadr[self.box_joint_id]
        self.data.qpos[box_qpos_adr:box_qpos_adr + 3] = BOX_INIT_POS
        self.data.qpos[box_qpos_adr + 3:box_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        box_qvel_adr = self.model.jnt_dofadr[self.box_joint_id]
        self.data.qvel[box_qvel_adr:box_qvel_adr + 6] = 0.0
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

    def pelvis_position(self):
        return np.array(self.data.xpos[self.pelvis_body_id])

    def torso_up(self):
        return float(self.data.xmat[self.torso_body_id].reshape(3, 3)[2, 2])

    def box_position(self):
        return np.array(self.data.xpos[self.box_body_id])

    def distance_to_target(self):
        return float(np.linalg.norm(self.pelvis_position()[:2] - TARGET_XY))

    def box_distance_to_target(self):
        return float(np.linalg.norm(self.box_position()[:2] - TARGET_XY))

    def box_distance_to_pelvis(self):
        return float(np.linalg.norm(self.box_position() - self.pelvis_position()))

    def _record_trace(self):
        self._trace.append(
            {
                "time": float(self.data.time),
                "pelvis_pos": self.pelvis_position(),
                "torso_up": self.torso_up(),
                "box_pos": self.box_position(),
                "box_distance_to_pelvis": self.box_distance_to_pelvis(),
            }
        )

    def home_ctrl(self):
        return self._initial_qpos[7:7 + self.model.nu].copy()

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
        trace_pelvis = np.array([entry["pelvis_pos"] for entry in self._trace], dtype=float)
        trace_torso_up = np.array([entry["torso_up"] for entry in self._trace], dtype=float)
        trace_box = np.array([entry["box_pos"] for entry in self._trace], dtype=float)
        trace_box_distance_to_pelvis = np.array([entry["box_distance_to_pelvis"] for entry in self._trace], dtype=float)
        payload = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "ctrl_trace": np.array(self._ctrl_trace, dtype=float).reshape(-1, self.model.nu),
            "initial_qpos": self._initial_qpos,
            "pelvis_pos": self.pelvis_position(),
            "torso_up": self.torso_up(),
            "box_pos": self.box_position(),
            "target_xy": TARGET_XY.copy(),
            "trace_time": trace_time,
            "trace_pelvis": trace_pelvis,
            "trace_torso_up": trace_torso_up,
            "trace_box": trace_box,
            "trace_box_distance_to_pelvis": trace_box_distance_to_pelvis,
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
    print("pelvis:", sim.pelvis_position())
    print("box:", sim.box_position())
    print("distance_to_target:", sim.distance_to_target())
    print("box_distance_to_target:", sim.box_distance_to_target())
    print("box_distance_to_pelvis:", sim.box_distance_to_pelvis())
    print("home_ctrl:", sim.home_ctrl())
