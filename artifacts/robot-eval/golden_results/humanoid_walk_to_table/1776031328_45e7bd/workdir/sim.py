"""Unitree H1 humanoid + table waypoint locomotion task."""
import os
import tempfile
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
H1_SCENE_XML = os.path.join(MENAGERIE, "unitree_h1", "scene.xml")

TABLE_CENTER = (0.88, 0.0, 0.38)
TABLE_HALF_SIZE = (0.22, 0.32, 0.38)
TARGET_XY = np.array([0.58, 0.0])
TARGET_RADIUS = 0.13
MIN_PELVIS_Z = 0.50
MIN_TORSO_UP = 0.55
TRACE_EVERY_STEPS = 10


def build_spec():
    spec = mujoco.MjSpec.from_file(H1_SCENE_XML)
    spec.option.gravity = [0.0, 0.0, -7.0]
    world = spec.worldbody
    table = world.add_body(name="table")
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(TABLE_CENTER),
        size=list(TABLE_HALF_SIZE),
        rgba=[0.55, 0.35, 0.2, 1],
    )
    world.add_geom(
        name="target_marker",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[float(TARGET_XY[0]), float(TARGET_XY[1]), 0.005],
        size=[TARGET_RADIUS, 0.005, 0],
        contype=0,
        conaffinity=0,
        rgba=[0.15, 0.7, 0.25, 0.35],
    )
    return spec


class Sim:
    def __init__(self):
        self.spec = build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
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

    def distance_to_target(self):
        pelvis_xy = self.pelvis_position()[:2]
        return float(np.linalg.norm(pelvis_xy - TARGET_XY))

    def _record_trace(self):
        self._trace.append(
            {
                "time": float(self.data.time),
                "pelvis_pos": self.pelvis_position(),
                "torso_up": self.torso_up(),
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
        payload = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "ctrl_trace": np.array(self._ctrl_trace, dtype=float).reshape(-1, self.model.nu),
            "initial_qpos": self._initial_qpos,
            "pelvis_pos": self.pelvis_position(),
            "torso_up": self.torso_up(),
            "target_xy": TARGET_XY.copy(),
            "trace_time": trace_time,
            "trace_pelvis": trace_pelvis,
            "trace_torso_up": trace_torso_up,
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
    print("torso_up:", sim.torso_up())
    print("distance_to_target:", sim.distance_to_target())
    print("home_ctrl:", sim.home_ctrl())
