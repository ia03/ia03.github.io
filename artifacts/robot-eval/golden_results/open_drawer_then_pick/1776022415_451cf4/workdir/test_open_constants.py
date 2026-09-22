import numpy as np

from sim import Sim


HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
Q_HANDLE = np.array([0.289, 0.618, -0.297, -1.705, 0.066, 3.752, 0.906], dtype=float)
Q_PULL = np.array([0.132, 1.215, -0.377, -0.622, -0.019, 3.362, 1.164], dtype=float)


def move_q(sim, q_target, grip, steps=260):
    q_start = sim.data.qpos[:7].copy()
    g_start = float(sim.data.ctrl[7])
    for i in range(steps):
        a = (i + 1) / steps
        sim.data.ctrl[:7] = (1 - a) * q_start + a * q_target
        sim.data.ctrl[7] = (1 - a) * g_start + a * grip
        sim.step(1)


def main():
    sim = Sim()
    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.step(200)
    move_q(sim, Q_HANDLE, 255, 260)
    print("after handle", sim.drawer_open_amount(), sim.block_position(), flush=True)
    move_q(sim, Q_HANDLE, 40, 160)
    print("after close", sim.drawer_open_amount(), sim.block_position(), flush=True)
    move_q(sim, Q_PULL, 40, 320)
    print("after pull", sim.drawer_open_amount(), sim.block_position(), flush=True)


if __name__ == "__main__":
    main()
