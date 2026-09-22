import numpy as np
import mujoco

from sim import Sim


HAND_BODY = "hand"


def clip_arm_ctrl(sim, q):
    lo = sim.model.actuator_ctrlrange[:7, 0]
    hi = sim.model.actuator_ctrlrange[:7, 1]
    return np.clip(q, lo, hi)


def hand_pos(sim):
    body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    return np.array(sim.data.xpos[body_id])


def ik_step(sim, target, nominal=None, pos_gain=4.0, damping=1e-3, reg=0.02):
    body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, body_id)
    j = jacp[:, :7]
    q = sim.data.qpos[:7].copy()
    err = target - hand_pos(sim)
    dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(3), pos_gain * err)
    if nominal is not None:
        dq += reg * (nominal - q)
    return clip_arm_ctrl(sim, q + dq)


def drive_to(sim, target, steps, grip, drawer, nominal, pos_gain=4.0):
    for _ in range(steps):
        sim.data.ctrl[:7] = ik_step(sim, target, nominal=nominal, pos_gain=pos_gain)
        sim.data.ctrl[7] = grip
        sim.data.ctrl[8] = drawer
        sim.step()


def current_block_top(sim):
    pos = sim.block_position().copy()
    pos[2] += 0.03
    return pos


def run_attempt(save_path="/work/final_state.npz"):
    sim = Sim()
    nominal = sim.data.qpos[:7].copy()

    # Early replayable artifact: open the drawer and move toward the workspace.
    drive_to(sim, np.array([0.45, 0.00, 0.72]), steps=120, grip=255, drawer=1.0, nominal=nominal)
    sim.save_final_state(save_path)

    drive_to(sim, np.array([0.62, 0.00, 0.62]), steps=140, grip=255, drawer=1.0, nominal=nominal)
    block = current_block_top(sim)
    drive_to(sim, block + np.array([0.0, 0.0, 0.10]), steps=140, grip=255, drawer=1.0, nominal=nominal)
    block = current_block_top(sim)
    drive_to(sim, block + np.array([0.0, 0.0, 0.045]), steps=140, grip=255, drawer=1.0, nominal=nominal, pos_gain=3.0)

    for grip in np.linspace(255, 0, 120):
        block = current_block_top(sim)
        sim.data.ctrl[:7] = ik_step(sim, block + np.array([0.0, 0.0, 0.045]), nominal=nominal, pos_gain=3.0)
        sim.data.ctrl[7] = float(grip)
        sim.data.ctrl[8] = 1.0
        sim.step()

    block = sim.block_position().copy()
    lift_target = np.array([block[0], block[1], 0.72])
    drive_to(sim, lift_target, steps=220, grip=0, drawer=1.0, nominal=nominal, pos_gain=3.5)
    drive_to(sim, lift_target + np.array([-0.02, 0.0, 0.0]), steps=180, grip=0, drawer=1.0, nominal=nominal, pos_gain=3.0)

    sim.save_final_state(save_path)
    return sim


def evaluate_ctrl_trace(path="/work/final_state.npz"):
    payload = np.load(path)
    ctrl_trace = payload["ctrl_trace"]
    sim = Sim()
    max_drawer = 0.0
    max_block_z = -1.0
    first_high = None
    first_open = None
    for i, ctrl in enumerate(ctrl_trace):
        sim.data.ctrl[:] = ctrl
        sim.step()
        drawer = sim.drawer_open_amount()
        block_z = sim.block_position()[2]
        max_drawer = max(max_drawer, drawer)
        max_block_z = max(max_block_z, block_z)
        if first_open is None and drawer >= 0.05:
            first_open = i
        if first_high is None and block_z >= 0.50:
            first_high = i
    contact_steps = 0
    for _ in range(500):
        sim.step()
        if sim.has_gripper_block_contact():
            contact_steps += 1
    return {
        "steps": int(len(ctrl_trace)),
        "drawer_final": float(sim.drawer_open_amount()),
        "block_final_z": float(sim.block_position()[2]),
        "max_drawer": float(max_drawer),
        "max_block_z": float(max_block_z),
        "first_open": first_open,
        "first_high": first_high,
        "contact_fraction": contact_steps / 500.0,
    }


if __name__ == "__main__":
    sim = run_attempt()
    print("final drawer", sim.drawer_open_amount())
    print("final block", sim.block_position())
    print("eval", evaluate_ctrl_trace())
