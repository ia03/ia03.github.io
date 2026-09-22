import numpy as np

from sim import Sim, MIN_PELVIS_Z, MIN_TORSO_UP, MIN_BOX_Z, MAX_BOX_TO_PELVIS


JOINTS = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "torso",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]


def build_joint_map(sim: Sim):
    qpos_adr = {}
    qvel_adr = {}
    for jid, name in enumerate(sim.joint_names()):
        if name is None:
            continue
        qpos_adr[name] = int(sim.model.jnt_qposadr[jid])
        qvel_adr[name] = int(sim.model.jnt_dofadr[jid])
    return qpos_adr, qvel_adr


def get_joint_qpos(sim: Sim, qpos_adr, name):
    adr = qpos_adr[name]
    if name == "torso":
        return float(sim.data.qpos[adr])
    return float(sim.data.qpos[adr])


def set_pd_ctrl(sim: Sim, qpos_adr, qvel_adr, q_des, kp, kd, bias=None):
    ctrl = np.zeros(sim.model.nu)
    for i, name in enumerate(JOINTS):
        q = get_joint_qpos(sim, qpos_adr, name)
        v = float(sim.data.qvel[qvel_adr[name]])
        u = kp[i] * (q_des[name] - q) - kd[i] * v
        if bias is not None:
            u += bias[i]
        lo, hi = sim.model.actuator_ctrlrange[i]
        ctrl[i] = np.clip(u, lo, hi)
    sim.data.ctrl[:] = ctrl


def evaluate(sim: Sim, steps=1600):
    pelvis_x = []
    box_x = []
    for _ in range(steps):
        sim.step(1)
        pelvis_x.append(sim.pelvis_position()[0])
        box_x.append(sim.box_position()[0])
    pelvis_x = np.array(pelvis_x)
    box_x = np.array(box_x)
    return {
        "pelvis_x_max": float(pelvis_x.max()),
        "box_x_max": float(box_x.max()),
        "final_pelvis": sim.pelvis_position().copy(),
        "final_box": sim.box_position().copy(),
        "pelvis_z_min": float(np.min([t["pelvis_pos"][2] for t in sim._trace])),
        "torso_up_min": float(np.min([t["torso_up"] for t in sim._trace])),
        "box_z_min": float(np.min([t["box_pos"][2] for t in sim._trace])),
        "box_to_pelvis_max": float(np.max([t["box_distance_to_pelvis"] for t in sim._trace])),
    }


def main():
    sim = Sim()
    qpos_adr, qvel_adr = build_joint_map(sim)

    home = {
        "left_hip_yaw": 0.0,
        "left_hip_roll": 0.0,
        "left_hip_pitch": -0.42,
        "left_knee": 0.82,
        "left_ankle": -0.42,
        "right_hip_yaw": 0.0,
        "right_hip_roll": 0.0,
        "right_hip_pitch": -0.42,
        "right_knee": 0.82,
        "right_ankle": -0.42,
        "torso": 0.0,
        "left_shoulder_pitch": -0.25,
        "left_shoulder_roll": 0.18,
        "left_shoulder_yaw": 0.0,
        "left_elbow": 0.55,
        "right_shoulder_pitch": -0.25,
        "right_shoulder_roll": -0.18,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.55,
    }

    crouch = home.copy()
    crouch.update(
        {
            "torso": 0.22,
            "left_hip_pitch": -0.58,
            "right_hip_pitch": -0.58,
            "left_knee": 1.05,
            "right_knee": 1.05,
            "left_ankle": -0.47,
            "right_ankle": -0.47,
            "left_shoulder_pitch": 0.05,
            "right_shoulder_pitch": 0.05,
            "left_shoulder_roll": 0.35,
            "right_shoulder_roll": -0.35,
            "left_elbow": 0.95,
            "right_elbow": 0.95,
        }
    )

    walk = crouch.copy()
    walk.update(
        {
            "torso": 0.30,
            "left_hip_pitch": -0.70,
            "right_hip_pitch": -0.56,
            "left_knee": 1.18,
            "right_knee": 0.96,
            "left_ankle": -0.50,
            "right_ankle": -0.40,
            "left_shoulder_pitch": 0.18,
            "right_shoulder_pitch": 0.18,
            "left_shoulder_roll": 0.48,
            "right_shoulder_roll": -0.48,
            "left_elbow": 1.10,
            "right_elbow": 1.10,
        }
    )

    kp = np.array([120, 120, 180, 220, 40, 120, 120, 180, 220, 40, 120, 50, 40, 25, 25, 50, 40, 25, 25], dtype=float)
    kd = np.array([10, 10, 14, 16, 4, 10, 10, 14, 16, 4, 10, 6, 5, 3, 3, 6, 5, 3, 3], dtype=float)
    bias = np.zeros(sim.model.nu)

    # First half: settle into a cradle pose and press the box against the torso.
    for _ in range(700):
        set_pd_ctrl(sim, qpos_adr, qvel_adr, crouch, kp, kd, bias)
        sim.step(1)

    # Second half: bias forward and alternate hip/knee load slightly to encourage progression.
    phase = 0.0
    for _ in range(900):
        phase += 0.035
        target = walk.copy()
        s = np.sin(phase)
        target["left_hip_pitch"] += 0.03 * s
        target["right_hip_pitch"] -= 0.03 * s
        target["left_knee"] -= 0.03 * s
        target["right_knee"] += 0.03 * s
        target["left_ankle"] += 0.01 * s
        target["right_ankle"] -= 0.01 * s
        bias[:] = 0.0
        bias[2] = 8.0
        bias[7] = 8.0
        bias[10] = 5.0
        set_pd_ctrl(sim, qpos_adr, qvel_adr, target, kp, kd, bias)
        sim.step(1)

    stats = evaluate(sim, 100)
    print(stats)
    print("final pelvis", sim.pelvis_position())
    print("final box", sim.box_position())
    print("distance_to_target", sim.distance_to_target())
    print("box_distance_to_target", sim.box_distance_to_target())
    print("box_distance_to_pelvis", sim.box_distance_to_pelvis())
    print("min constraints", stats["pelvis_z_min"], stats["torso_up_min"], stats["box_z_min"], stats["box_to_pelvis_max"])
    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
