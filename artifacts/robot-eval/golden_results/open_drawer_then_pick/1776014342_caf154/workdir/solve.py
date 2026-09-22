import numpy as np

from sim import Sim


HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
Q_HANDLE = np.array(
    [0.2887739083550356, 0.6178838727508454, -0.2969239199163194, -1.7053069639778142, 0.06592295071522526, 3.7525, 0.9064924581847854],
    dtype=float,
)
Q_PULL = np.array(
    [0.13213800116989255, 1.214627915621127, -0.37697910858577277, -0.6215977616246623, -0.01917876978463884, 3.361660629985802, 1.164405291264782],
    dtype=float,
)
Q_RETRACT = np.array(
    [0.260114734942896, 0.5978935416914477, -0.2548739609310991, -1.7011568239308756, 0.0731417910040021, 3.7525, 0.8723261334629767],
    dtype=float,
)


def move_q(sim, q_target, grip, steps):
    q_start = sim.data.qpos[:7].copy()
    g_start = float(sim.data.ctrl[7])
    for i in range(steps):
        a = (i + 1) / steps
        sim.data.ctrl[:7] = (1 - a) * q_start + a * q_target
        sim.data.ctrl[7] = (1 - a) * g_start + a * grip
        sim.step(1)


def hold(sim, q_target, grip, steps):
    for _ in range(steps):
        sim.data.ctrl[:7] = q_target
        sim.data.ctrl[7] = grip
        sim.step(1)


def main():
    sim = Sim()
    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.step(200)

    move_q(sim, Q_HANDLE, 255, 260)
    move_q(sim, Q_HANDLE, 40, 160)
    move_q(sim, Q_PULL, 40, 320)
    hold(sim, Q_PULL, 40, 120)
    move_q(sim, Q_RETRACT, 255, 240)
    hold(sim, Q_RETRACT, 255, 120)

    sim.save_final_state("/work/final_state.npz")


if __name__ == "__main__":
    main()
