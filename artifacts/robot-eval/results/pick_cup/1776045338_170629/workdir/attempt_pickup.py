import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

from sim import Sim


HAND_BODY = "hand"
LEFT_FINGER_BODY = "left_finger"
RIGHT_FINGER_BODY = "right_finger"
CUP_BODY = "cup"
CUP_GEOM = "cup_geom"
LEFT_PAD_GEOM = 69
RIGHT_PAD_GEOM = 77
ARM_DOF = 7


def mat_to_quat(mat):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.reshape(9))
    return quat


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    return np.array(
        [
            a[0] * b[0] - np.dot(a[1:], b[1:]),
            a[0] * b[1] + b[0] * a[1] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] + b[0] * a[2] + a[3] * b[1] - a[1] * b[3],
            a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1],
        ]
    )


def quat_err(target, current):
    q = quat_mul(target, quat_conj(current))
    if q[0] < 0:
        q = -q
    return 2.0 * q[1:]


def body_pose(sim, body_name):
    body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = sim.data.xpos[body_id].copy()
    quat = mat_to_quat(sim.data.xmat[body_id].reshape(3, 3))
    return pos, quat


def pad_midpoint(sim):
    return 0.5 * (sim.data.geom_xpos[LEFT_PAD_GEOM] + sim.data.geom_xpos[RIGHT_PAD_GEOM])


def cup_contacts(sim):
    cup_geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, CUP_GEOM)
    finger_body_ids = {
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, LEFT_FINGER_BODY),
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_FINGER_BODY),
    }
    count = 0
    for i in range(sim.data.ncon):
        con = sim.data.contact[i]
        if con.geom1 == cup_geom_id:
            other = con.geom2
        elif con.geom2 == cup_geom_id:
            other = con.geom1
        else:
            continue
        if sim.model.geom_bodyid[other] in finger_body_ids:
            count += 1
    return count


def eval_saved_state(path):
    payload = np.load(path)
    ctrl_trace = payload["ctrl_trace"]
    final_ctrl = payload["ctrl"]
    sim = Sim()
    for ctrl in ctrl_trace:
        sim.data.ctrl[:] = ctrl
        sim.step()
    finger_contact_steps = 0
    for _ in range(500):
        sim.data.ctrl[:] = final_ctrl
        sim.step()
        if cup_contacts(sim) > 0:
            finger_contact_steps += 1
    cup_z = sim.cup_position()[2]
    contact_fraction = finger_contact_steps / 500.0
    height_progress = np.clip((cup_z - 0.435) / (0.52 - 0.435), 0.0, 1.0)
    contact_progress = np.clip(contact_fraction / 0.5, 0.0, 1.0)
    score = 0.5 * height_progress + 0.5 * contact_progress
    return dict(cup_z=cup_z, contact_fraction=contact_fraction, progress_score=score)


@dataclass
class Phase:
    name: str
    steps: int
    pad_target: np.ndarray
    grip: float


def ik_step(sim, target_pad_pos, target_quat, q_nominal):
    hand_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    jacp_l = np.zeros((3, sim.model.nv))
    jacp_r = np.zeros((3, sim.model.nv))
    mujoco.mj_jacGeom(sim.model, sim.data, jacp_l, None, LEFT_PAD_GEOM)
    mujoco.mj_jacGeom(sim.model, sim.data, jacp_r, None, RIGHT_PAD_GEOM)
    mujoco.mj_jacBody(sim.model, sim.data, jacp, jacr, hand_id)
    hand_pos, hand_quat = body_pose(sim, HAND_BODY)
    current_pad_pos = pad_midpoint(sim)

    pos_err = target_pad_pos - current_pad_pos
    rot_err = quat_err(target_quat, hand_quat)
    jacp_avg = 0.5 * (jacp_l[:, :ARM_DOF] + jacp_r[:, :ARM_DOF])
    task = np.concatenate([12.0 * pos_err, 2.0 * rot_err])
    J = np.vstack([12.0 * jacp_avg, 2.0 * jacr[:, :ARM_DOF]])

    reg = 0.02 * np.eye(ARM_DOF)
    dq = np.linalg.solve(J.T @ J + reg, J.T @ task)
    dq += 0.03 * (q_nominal - sim.data.qpos[:ARM_DOF])

    q_next = sim.data.qpos[:ARM_DOF] + np.clip(dq, -0.05, 0.05)
    q_min = sim.model.jnt_range[:ARM_DOF, 0]
    q_max = sim.model.jnt_range[:ARM_DOF, 1]
    return np.clip(q_next, q_min, q_max)


def run_attempt():
    sim = Sim()
    nominal_q = np.array([0.0, 0.25, 0.0, -1.9, 0.0, 2.2, 0.78])
    sim.data.ctrl[:ARM_DOF] = nominal_q
    sim.data.ctrl[7] = 255.0
    sim.step(300)

    _, hand_quat0 = body_pose(sim, HAND_BODY)

    cup0 = sim.cup_position().copy()
    phases = [
        Phase("pregrasp_high", 220, cup0 + np.array([0.0, 0.0, 0.11]), 255.0),
        Phase("pregrasp_mid", 160, cup0 + np.array([0.0, 0.0, 0.06]), 150.0),
        Phase("descend", 220, cup0 + np.array([0.0, 0.0, 0.012]), 120.0),
        Phase("close", 260, cup0 + np.array([0.0, 0.0, 0.018]), 0.0),
        Phase("lift", 320, cup0 + np.array([0.0, 0.0, 0.18]), 0.0),
        Phase("hold", 320, cup0 + np.array([0.0, 0.0, 0.18]), 0.0),
    ]

    saved_once = False
    best = None
    for phase in phases:
        for _ in range(phase.steps):
            q_target = ik_step(sim, phase.pad_target, hand_quat0, nominal_q)
            sim.data.ctrl[:ARM_DOF] = q_target
            sim.data.ctrl[7] = phase.grip
            sim.step()

        if not saved_once and phase.name in {"close", "lift", "hold"}:
            sim.save_final_state("/work/final_state.npz")
            saved_once = True
            best = eval_saved_state("/work/final_state.npz")
            print("early_save", phase.name, best)

        if saved_once:
            metrics = eval_saved_state("/work/final_state.npz")
            if best is None or metrics["progress_score"] > best["progress_score"]:
                best = metrics

    sim.save_final_state("/work/final_state.npz")
    metrics = eval_saved_state("/work/final_state.npz")
    print("final", metrics)
    frame = sim.render()
    try:
        from PIL import Image

        Image.fromarray(frame).save("/work/attempt_final.png")
    except Exception:
        pass
    return metrics


if __name__ == "__main__":
    metrics = run_attempt()
    print(metrics)
