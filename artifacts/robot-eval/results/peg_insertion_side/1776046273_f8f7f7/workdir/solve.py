import numpy as np
import mujoco
from PIL import Image

import sim


HAND_BODY = "hand"
ARM_DOF = np.arange(7)


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def body_pose(data, body_id):
    pos = data.xpos[body_id].copy()
    rot = data.xmat[body_id].reshape(3, 3).copy()
    return pos, rot


def step_toward(sim_inst, target_pos, gain=0.35, damping=1e-4):
    model = sim_inst.model
    data = sim_inst.data
    hand_id = sim_inst.hand_id
    pos, _ = body_pose(data, hand_id)
    err = target_pos - pos
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
    J = jacp[:, ARM_DOF]
    A = J @ J.T + damping * np.eye(3)
    dq = J.T @ np.linalg.solve(A, err)
    q = data.qpos[:7].copy()
    q_des = q + gain * dq
    q_des = clamp(q_des, model.jnt_range[:7, 0], model.jnt_range[:7, 1])
    data.ctrl[:7] = q_des
    data.ctrl[7] = 255.0


def run():
    s = sim.Sim()
    s.hand_id = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)

    # Make sure the gripper starts opening immediately.
    s.data.ctrl[:] = 0.0
    s.data.ctrl[7] = 255.0

    stages = [
        (250, np.array([0.24, 0.00, 0.82])),
        (250, np.array([0.34, -0.03, 0.68])),
        (250, np.array([0.40, -0.08, 0.60])),
        (450, np.array([0.54, -0.102, 0.58])),
        (250, np.array([0.54, -0.102, 0.58])),
    ]

    for steps, target in stages:
        for _ in range(steps):
            step_toward(s, target)
            s.step(1)

    # Save a frame for quick inspection.
    frame = s.render(640, 480)
    Image.fromarray(frame).save("/work/final_frame.png")

    # Save the first replayable candidate immediately.
    s.save_final_state("/work/final_state.npz")

    print("final hand pos:", np.round(s.data.xpos[s.hand_id], 4))
    print("final peg pos:", np.round(s.peg_position(), 4))
    print("saved /work/final_state.npz and /work/final_frame.png")


if __name__ == "__main__":
    run()
