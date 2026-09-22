import numpy as np

from sim import Sim


def pd_torque(sim: Sim, q_des: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> np.ndarray:
    # Actuated joints are exactly qpos[7:26] and qvel[6:25] for this model.
    q = sim.data.qpos[7:26]
    qd = sim.data.qvel[6:25]
    return kp * (q_des - q) - kd * qd


def main():
    sim = Sim()

    # Home joint angles (not torques). We'll track this with a PD controller.
    q_home = sim.home_ctrl().copy()

    kp = np.array(
        [
            250,
            250,
            250,
            320,
            90,  # left leg
            250,
            250,
            250,
            320,
            90,  # right leg
            200,  # torso
            60,
            60,
            30,
            30,  # left arm
            60,
            60,
            30,
            30,  # right arm
        ],
        dtype=float,
    )
    kd = kp * 0.06

    # First executable end-to-end attempt: just stabilize at home for a bit and save.
    sim.reset()
    for _ in range(1200):
        sim.data.ctrl[:] = pd_torque(sim, q_home, kp, kd)
        sim.step(1)

    sim.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")
    print("pelvis", sim.pelvis_position(), "torso_up", sim.torso_up())
    print("box", sim.box_position(), "box_to_pelvis", sim.box_distance_to_pelvis())


if __name__ == "__main__":
    main()

