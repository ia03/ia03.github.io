"""Panda arm + peg + side insertion fixture MuJoCo simulation."""
import os
import tempfile
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
PANDA_XML = os.path.join(MENAGERIE, "franka_emika_panda", "panda.xml")

TABLE_CENTER = (0.5, 0.0, 0.2)
TABLE_HALF_SIZE = (0.30, 0.26, 0.2)
PEG_HALF = (0.03, 0.012, 0.012)
PEG_INIT_POS = (0.47, -0.102, 0.508 + PEG_HALF[2])
BOARD_CENTER = (0.52, -0.10, 0.52)
BOARD_SIZE = (0.02, 0.10, 0.10)
SLOT_HALF_HEIGHT = 0.025
SLOT_HALF_WIDTH = 0.040
GUIDE_CENTER = (0.49, -0.102, 0.50)
GUIDE_HALF = (0.08, 0.03, 0.008)


def build_spec():
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.85, 0.85, 0.85, 1])
    table = world.add_body(name="table", pos=list(TABLE_CENTER))
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TABLE_HALF_SIZE), rgba=[0.55, 0.35, 0.2, 1])
    guide = world.add_body(name="guide")
    guide.add_geom(name="guide_shelf", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(GUIDE_CENTER), size=list(GUIDE_HALF), rgba=[0.45, 0.45, 0.5, 1])

    peg = world.add_body(name="peg", pos=list(PEG_INIT_POS))
    peg.add_freejoint()
    peg.add_geom(name="peg_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(PEG_HALF), mass=0.07, friction=[2.0, 0.5, 0.1], rgba=[0.85, 0.55, 0.2, 1])
    peg.add_geom(
        name="peg_tab",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[-0.050, 0.0, 0.0],
        size=[0.018, 0.030, 0.030],
        mass=0.02,
        friction=[2.0, 0.5, 0.1],
        rgba=[0.95, 0.75, 0.25, 1],
    )

    board = world.add_body(name="board")
    top_inner_z = BOARD_CENTER[2] + SLOT_HALF_HEIGHT
    top_outer_z = BOARD_CENTER[2] + BOARD_SIZE[2]
    bottom_inner_z = BOARD_CENTER[2] - SLOT_HALF_HEIGHT
    bottom_outer_z = BOARD_CENTER[2] - BOARD_SIZE[2]
    side_inner_y = BOARD_CENTER[1] + SLOT_HALF_WIDTH
    side_outer_y = BOARD_CENTER[1] + BOARD_SIZE[1]
    board.add_geom(
        name="board_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[BOARD_CENTER[0], BOARD_CENTER[1], 0.5 * (top_inner_z + top_outer_z)],
        size=[BOARD_SIZE[0], BOARD_SIZE[1], 0.5 * (top_outer_z - top_inner_z)],
        rgba=[0.25, 0.25, 0.25, 1],
    )
    board.add_geom(
        name="board_bottom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[BOARD_CENTER[0], BOARD_CENTER[1], 0.5 * (bottom_inner_z + bottom_outer_z)],
        size=[BOARD_SIZE[0], BOARD_SIZE[1], 0.5 * (bottom_inner_z - bottom_outer_z)],
        rgba=[0.25, 0.25, 0.25, 1],
    )
    board.add_geom(
        name="board_side_pos",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[BOARD_CENTER[0], 0.5 * (side_inner_y + side_outer_y), BOARD_CENTER[2]],
        size=[BOARD_SIZE[0], 0.5 * (side_outer_y - side_inner_y), SLOT_HALF_HEIGHT],
        rgba=[0.25, 0.25, 0.25, 1],
    )
    board.add_geom(
        name="board_side_neg",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[BOARD_CENTER[0], BOARD_CENTER[1] - 0.5 * (BOARD_SIZE[1] + SLOT_HALF_WIDTH), BOARD_CENTER[2]],
        size=[BOARD_SIZE[0], 0.5 * (side_outer_y - side_inner_y), SLOT_HALF_HEIGHT],
        rgba=[0.25, 0.25, 0.25, 1],
    )
    return spec


class Sim:
    def __init__(self):
        self.spec = build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.peg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
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

    def peg_position(self):
        return np.array(self.data.xpos[self.peg_body_id])

    def actuator_names(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]

    def joint_names(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]

    def render(self, width=640, height=480, camera=-1):
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def save_final_state(self, path="/work/final_state.npz"):
        payload = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "ctrl_trace": np.array(self._ctrl_trace, dtype=float).reshape(-1, self.model.nu),
            "initial_qpos": self._initial_qpos,
            "peg_pos": self.peg_position(),
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
    print("initial peg pos:", sim.peg_position())
