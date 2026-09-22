"""Panda arm + table + two blocks MuJoCo simulation."""
import os
import tempfile
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
PANDA_XML = os.path.join(MENAGERIE, "franka_emika_panda", "panda.xml")

TABLE_CENTER = (0.5, 0.0, 0.2)
TABLE_HALF_SIZE = (0.26, 0.26, 0.2)
BLOCK_HALF = 0.025
RED_INIT_POS = (0.48, -0.09, 0.4 + BLOCK_HALF)
GREEN_INIT_POS = (0.63, 0.08, 0.4 + BLOCK_HALF)


def build_spec():
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.85, 0.85, 0.85, 1])
    table = world.add_body(name="table", pos=list(TABLE_CENTER))
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TABLE_HALF_SIZE), rgba=[0.55, 0.35, 0.2, 1])
    red = world.add_body(name="red_block", pos=list(RED_INIT_POS))
    red.add_freejoint()
    red.add_geom(name="red_block_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BLOCK_HALF, BLOCK_HALF, BLOCK_HALF], mass=0.08, friction=[1.6, 0.4, 0.1], rgba=[0.85, 0.2, 0.2, 1])
    green = world.add_body(name="green_block", pos=list(GREEN_INIT_POS))
    green.add_freejoint()
    green.add_geom(name="green_block_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BLOCK_HALF, BLOCK_HALF, BLOCK_HALF], mass=0.12, friction=[1.8, 0.5, 0.1], rgba=[0.2, 0.8, 0.3, 1])
    return spec


class Sim:
    def __init__(self):
        self.spec = build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.red_block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
        self.green_block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "green_block")
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

    def block_positions(self):
        return {
            "red": np.array(self.data.xpos[self.red_block_id]),
            "green": np.array(self.data.xpos[self.green_block_id]),
        }

    def actuator_names(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]

    def joint_names(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]

    def render(self, width=640, height=480, camera=-1):
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def save_final_state(self, path="/work/final_state.npz"):
        positions = self.block_positions()
        payload = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "ctrl_trace": np.array(self._ctrl_trace, dtype=float).reshape(-1, self.model.nu),
            "initial_qpos": self._initial_qpos,
            "red_pos": positions["red"],
            "green_pos": positions["green"],
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
    print("block positions:", sim.block_positions())
