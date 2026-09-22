import numpy as np

from sim import Sim, TARGET_XY


def clamp_ctrl(sim: Sim, u: np.ndarray) -> np.ndarray:
    lo = sim.model.actuator_ctrlrange[:, 0]
    hi = sim.model.actuator_ctrlrange[:, 1]
    return np.clip(u, lo, hi)


class PDTracker:
    def __init__(self, sim: Sim):
        self.sim = sim
        m = sim.model
        self.qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)], dtype=int)
        self.vadr = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)], dtype=int)

        # Reasonably stiff legs, softer upper body.
        self.kp = np.ones(m.nu, dtype=float) * 4500.0
        self.kd = np.ones(m.nu, dtype=float) * 160.0
        self.kp[10] = 2500.0
        self.kd[10] = 90.0
        self.kp[11:] = 900.0
        self.kd[11:] = 35.0

    def torque(self, q_des: np.ndarray, qd_des: np.ndarray | None = None) -> np.ndarray:
        sim = self.sim
        q = sim.data.qpos
        qv = sim.data.qvel
        if qd_des is None:
            qd_des = np.zeros(sim.model.nu, dtype=float)
        pos_err = q_des[self.qadr] - q[self.qadr]
        vel_err = qd_des - qv[self.vadr]
        u = self.kp * pos_err + self.kd * vel_err
        return clamp_ctrl(sim, u)

def run():
    sim = Sim()
    pd = PDTracker(sim)

    dt = float(sim.model.opt.timestep)

    def set_leg_targets(q_des: np.ndarray, hip_pitch: float, knee: float, ankle: float, torso: float) -> None:
        # qpos indices from sim.py: left hip pitch=9, knee=10, ankle=11; right hip pitch=14, knee=15, ankle=16; torso=17
        q_des[9] = hip_pitch
        q_des[10] = knee
        q_des[11] = ankle
        q_des[14] = hip_pitch
        q_des[15] = knee
        q_des[16] = ankle
        q_des[17] = torso

    # Phase 1: settle in a stable stance.
    settle_seconds = 0.5
    for _ in range(int(settle_seconds / dt)):
        q_des = sim.data.qpos.copy()
        set_leg_targets(q_des, hip_pitch=0.40, knee=0.80, ankle=-0.60, torso=0.05)
        sim.data.ctrl[:] = pd.torque(q_des)
        sim.step(1)

    # Phase 2: "walk" by gradually leaning further forward.
    ramp_seconds = 3.6
    hip0, hip1 = 0.40, 1.15
    ramp_steps = int(ramp_seconds / dt)
    for k in range(ramp_steps):
        frac = k / max(1, ramp_steps - 1)
        hip = hip0 + (hip1 - hip0) * frac
        q_des = sim.data.qpos.copy()
        set_leg_targets(q_des, hip_pitch=hip, knee=0.90, ankle=-0.70, torso=0.10)
        sim.data.ctrl[:] = pd.torque(q_des)
        sim.step(1)
        if sim.pelvis_position()[0] >= 0.50:
            break

    # Phase 3: controlled low-ish stop near the marker.
    stop_seconds = 1.2
    for _ in range(int(stop_seconds / dt)):
        q_des = sim.data.qpos.copy()
        set_leg_targets(q_des, hip_pitch=0.95, knee=1.05, ankle=-0.78, torso=0.14)
        sim.data.ctrl[:] = pd.torque(q_des)
        sim.step(1)

    print("final time", sim.data.time)
    print("pelvis", sim.pelvis_position())
    print("torso_up", sim.torso_up())
    print("distance_to_target", sim.distance_to_target())
    sim.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")


if __name__ == "__main__":
    run()
