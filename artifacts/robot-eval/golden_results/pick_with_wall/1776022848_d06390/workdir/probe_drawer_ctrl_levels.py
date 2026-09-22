import numpy as np

from sim import Sim


HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
Q_HANDLE = np.array(
    [0.260114734942896, 0.5978935416914477, -0.2548739609310991, -1.7011568239308756, 0.0731417910040021, 3.7525, 0.8723261334629767],
    dtype=float,
)
Q_PULL = np.array(
    [0.12598874856351472, 1.1973462499176968, -0.34850701301211456, -0.625005200213342, -0.022487162744560703, 3.3513450345879296, 1.1390361931507589],
    dtype=float,
)


def move_q(sim, q_target, grip, steps, drawer_ctrl=1.0):
    q_start = sim.data.qpos[:7].copy()
    g_start = float(sim.data.ctrl[7])
    d_start = float(sim.data.ctrl[8])
    for i in range(steps):
        a = (i + 1) / steps
        sim.data.ctrl[:7] = (1 - a) * q_start + a * q_target
        sim.data.ctrl[7] = (1 - a) * g_start + a * grip
        sim.data.ctrl[8] = (1 - a) * d_start + a * drawer_ctrl
        sim.step(1)


def hold(sim, q_target, grip, steps, drawer_ctrl=1.0):
    for _ in range(steps):
        sim.data.ctrl[:7] = q_target
        sim.data.ctrl[7] = grip
        sim.data.ctrl[8] = drawer_ctrl
        sim.step(1)


def run(target_ctrl):
    sim = Sim()
    sim.data.ctrl[:7] = HOME
    sim.data.ctrl[7] = 255
    sim.data.ctrl[8] = 1.0
    sim.step(200)
    move_q(sim, Q_HANDLE, 255, 260, drawer_ctrl=target_ctrl)
    move_q(sim, Q_HANDLE, 40, 160, drawer_ctrl=target_ctrl)
    move_q(sim, Q_PULL, 40, 320, drawer_ctrl=target_ctrl)
    hold(sim, Q_PULL, 40, 120, drawer_ctrl=target_ctrl)
    print({
        "target_ctrl": target_ctrl,
        "drawer": sim.drawer_open_amount(),
        "block": sim.block_position().tolist(),
    })


if __name__ == "__main__":
    for ctrl in [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]:
        run(ctrl)
