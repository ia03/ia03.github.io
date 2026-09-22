"""Panda arm + table + cup MuJoCo simulation.

Quick start:
    from sim import Sim
    sim = Sim()
    print("actuators:", sim.actuator_names())
    print("initial cup pos:", sim.cup_position())
    sim.data.ctrl[:] = ...        # set control targets
    sim.step(n=1000)
    sim.save_final_state("/work/final_state.npz")

Or just run: `python sim.py`
"""
import os
import tempfile
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
PANDA_XML = os.path.join(MENAGERIE, "franka_emika_panda", "panda.xml")

CUP_RADIUS = 0.022
CUP_HALF_HEIGHT = 0.035
CUP_MASS = 0.05
CUP_FRICTION = (2.0, 0.5, 0.1)
CUP_INIT_POS = (0.5, 0.0, 0.4 + CUP_HALF_HEIGHT)  # resting on table top
TABLE_CENTER = (0.5, 0.0, 0.2)
TABLE_HALF_SIZE = (0.2, 0.2, 0.2)  # top surface at z=0.4
SUCCESS_CUP_Z = 0.52


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
    return spec


class Sim:
    def __init__(self):
        self.spec = build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.cup_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cup"
        )
        self.reset()
        self._initial_qpos = self.data.qpos.copy()
        self._ctrl_trace = []

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._ctrl_trace = []

    def step(self, n=1):
        for _ in range(n):
            self._ctrl_trace.append(self.data.ctrl.copy())
            mujoco.mj_step(self.model, self.data)

    def cup_position(self):
        return np.array(self.data.xpos[self.cup_body_id])

    def actuator_names(self):
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

    def joint_names(self):
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]

    def render(self, width=640, height=480, camera=-1):
        r = mujoco.Renderer(self.model, height=height, width=width)
        r.update_scene(self.data, camera=camera)
        return r.render()

    def save_final_state(self, path="/work/final_state.npz"):
        payload = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "ctrl_trace": np.array(self._ctrl_trace, dtype=float).reshape(-1, self.model.nu),
            "initial_qpos": self._initial_qpos,
            "cup_pos": self.cup_position(),
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
    print(f"target: cup_z >= {SUCCESS_CUP_Z}")
