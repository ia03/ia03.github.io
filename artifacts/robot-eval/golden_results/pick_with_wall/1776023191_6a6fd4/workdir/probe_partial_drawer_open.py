import numpy as np
from sim import Sim


HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)


def main():
    for target in [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.16]:
        sim = Sim()
        sim.data.ctrl[:7] = HOME
        sim.data.ctrl[7] = 255
        sim.data.ctrl[8] = 1.0
        sim.step(200)
        for _ in range(280):
            sim.data.ctrl[:7] = HOME
            sim.data.ctrl[7] = 255
            sim.data.ctrl[8] = target
            sim.step(1)
        for _ in range(120):
            sim.data.ctrl[:7] = HOME
            sim.data.ctrl[7] = 255
            sim.data.ctrl[8] = target
            sim.step(1)
        print({"target": target, "drawer": sim.drawer_open_amount(), "block": sim.block_position().tolist()})


if __name__ == "__main__":
    main()
