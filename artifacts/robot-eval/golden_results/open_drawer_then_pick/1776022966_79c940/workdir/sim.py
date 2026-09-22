"""Panda arm + sliding drawer + block MuJoCo simulation."""
import os
import tempfile
import numpy as np
import mujoco

MENAGERIE = os.environ.get("MENAGERIE_ROOT", "/opt/menagerie")
PANDA_XML = os.path.join(MENAGERIE, "franka_emika_panda", "panda.xml")

TABLE_CENTER = (0.5, 0.0, 0.2)
TABLE_HALF_SIZE = (0.30, 0.28, 0.2)
CABINET_POS = (0.66, -0.02, 0.44)
CABINET_HALF = (0.09, 0.12, 0.04)
DRAWER_HALF = (0.075, 0.105, 0.03)
DRAWER_RANGE = (0.0, 0.16)
BLOCK_RADIUS = 0.018
BLOCK_HALF_Z = 0.04
BLOCK_INIT_POS = (0.62, 0.04, 0.4 + BLOCK_HALF_Z)
TRACE_EVERY_STEPS = 10
DRAWER_HANDLE_HALF = (0.012, 0.05, 0.012)
DRAWER_HANDLE_X = DRAWER_HALF[0] + 0.025


def build_spec():
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    world = spec.worldbody
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.85, 0.85, 0.85, 1])
    table = world.add_body(name="table", pos=list(TABLE_CENTER))
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TABLE_HALF_SIZE), rgba=[0.55, 0.35, 0.2, 1])

    cabinet = world.add_body(name="cabinet", pos=list(CABINET_POS))
    cabinet.add_geom(name="cabinet_left", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -CABINET_HALF[1], 0], size=[CABINET_HALF[0], 0.01, CABINET_HALF[2]], rgba=[0.45, 0.45, 0.5, 1])
    cabinet.add_geom(name="cabinet_right", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, CABINET_HALF[1], 0], size=[CABINET_HALF[0], 0.01, CABINET_HALF[2]], rgba=[0.45, 0.45, 0.5, 1])
    cabinet.add_geom(name="cabinet_back", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-CABINET_HALF[0], 0, 0], size=[0.01, CABINET_HALF[1], CABINET_HALF[2]], rgba=[0.45, 0.45, 0.5, 1])
    cabinet.add_geom(name="cabinet_bottom", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -CABINET_HALF[2]], size=[CABINET_HALF[0], CABINET_HALF[1], 0.01], rgba=[0.45, 0.45, 0.5, 1])
    cabinet.add_geom(name="cabinet_shelf", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.01, 0, -0.005], size=[0.065, 0.095, 0.008], rgba=[0.6, 0.6, 0.65, 1])

    drawer = cabinet.add_body(name="drawer")
    drawer.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=list(DRAWER_RANGE), damping=2.0)
    drawer.add_geom(name="drawer_front", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[DRAWER_HALF[0], 0, 0], size=[0.01, DRAWER_HALF[1], DRAWER_HALF[2]], rgba=[0.75, 0.55, 0.3, 1])
    drawer.add_geom(name="drawer_handle", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[DRAWER_HANDLE_X, 0, 0.0], size=list(DRAWER_HANDLE_HALF), rgba=[0.15, 0.15, 0.15, 1])
    spec.add_actuator(
        name="drawer_motor",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target="drawer_slide",
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=[40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        biastype=mujoco.mjtBias.mjBIAS_NONE,
        ctrllimited=True,
        ctrlrange=[-1.0, 1.0],
    )

    block = world.add_body(name="block", pos=list(BLOCK_INIT_POS))
    block.add_freejoint()
    block.add_geom(
        name="block_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[BLOCK_RADIUS, BLOCK_HALF_Z, 0],
        mass=0.03,
        friction=[1.8, 0.5, 0.1],
        rgba=[0.85, 0.2, 0.2, 1],
    )
    return spec


class Sim:
    def __init__(self):
        self.spec = build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.drawer_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        self.block_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.block_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom")
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

    def drawer_open_amount(self):
        qpos_adr = self.model.jnt_qposadr[self.drawer_joint_id]
        return float(self.data.qpos[qpos_adr])

    def block_position(self):
        return np.array(self.data.xpos[self.block_body_id])

    def _gripper_geom_ids(self):
        ids = set()
        for body_name in ("left_finger", "right_finger"):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for geom_id in range(self.model.ngeom):
                if int(self.model.geom_bodyid[geom_id]) == body_id:
                    ids.add(geom_id)
        return ids

    def has_gripper_block_contact(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if self.block_geom_id in (g1, g2) and (g1 in self.finger_geom_ids or g2 in self.finger_geom_ids):
                return True
        return False

    def _record_trace(self):
        self._trace.append(
            {
                "time": float(self.data.time),
                "drawer_open": self.drawer_open_amount(),
                "block_pos": self.block_position(),
                "block_contact": float(self.has_gripper_block_contact()),
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
        trace_drawer_open = np.array([entry["drawer_open"] for entry in self._trace], dtype=float)
        trace_block = np.array([entry["block_pos"] for entry in self._trace], dtype=float)
        trace_contact = np.array([entry["block_contact"] for entry in self._trace], dtype=float)
        payload = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "ctrl_trace": np.array(self._ctrl_trace, dtype=float).reshape(-1, self.model.nu),
            "initial_qpos": self._initial_qpos,
            "drawer_open": self.drawer_open_amount(),
            "block_pos": self.block_position(),
            "trace_time": trace_time,
            "trace_drawer_open": trace_drawer_open,
            "trace_block": trace_block,
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
    print("drawer open:", sim.drawer_open_amount())
    print("block pos:", sim.block_position())
