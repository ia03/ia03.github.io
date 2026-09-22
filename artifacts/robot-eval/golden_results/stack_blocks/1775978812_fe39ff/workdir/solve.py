import numpy as np
import mujoco

from sim import Sim


ARM_DOF = np.arange(7)
Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
LEFT_PAD_GEOM = 69
RIGHT_PAD_GEOM = 77


class StackController:
    def __init__(self):
        self.sim = Sim()
        self.model = self.sim.model
        self.data = self.sim.data
        self.red_id = self.sim.red_block_id
        self.green_id = self.sim.green_block_id
        self.qmin = self.model.actuator_ctrlrange[:7, 0]
        self.qmax = self.model.actuator_ctrlrange[:7, 1]

    def pinch_center(self):
        return 0.5 * (
            self.data.geom_xpos[LEFT_PAD_GEOM] + self.data.geom_xpos[RIGHT_PAD_GEOM]
        )

    def pinch_jacobian(self):
        jacp_left = np.zeros((3, self.model.nv))
        jacp_right = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model, self.data, jacp_left, jacr, LEFT_PAD_GEOM)
        mujoco.mj_jacGeom(self.model, self.data, jacp_right, jacr, RIGHT_PAD_GEOM)
        return 0.5 * (jacp_left[:, ARM_DOF] + jacp_right[:, ARM_DOF])

    def move_pinch(self, target, steps, gripper, tol=0.003, gain=0.45, home_weight=0.01):
        for i in range(steps):
            current = self.pinch_center()
            err = target - current
            jac = self.pinch_jacobian()
            reg = 3e-4 * np.eye(3)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + reg, err)
            q = self.data.qpos[:7].copy()
            dq += home_weight * (Q_HOME - q)
            self.data.ctrl[:7] = np.clip(q + gain * dq, self.qmin, self.qmax)
            self.data.ctrl[7] = gripper
            self.sim.step()
            if np.linalg.norm(err) < tol and i > 30:
                return True
        return False

    def run(self):
        self.data.ctrl[:7] = Q_HOME
        self.data.ctrl[7] = 255
        self.sim.step(1000)

        red = self.data.xpos[self.red_id].copy()
        green = self.data.xpos[self.green_id].copy()
        sequence = [
            (np.array([red[0], red[1], 0.54]), 255, 800),
            (np.array([red[0], red[1], red[2] + 0.002]), 255, 900),
            (np.array([red[0], red[1], red[2] + 0.002]), 0, 500),
            (np.array([red[0], red[1], 0.56]), 0, 900),
            (np.array([green[0], green[1], 0.56]), 0, 1200),
            (np.array([green[0], green[1], green[2] + 0.052]), 0, 1000),
            (np.array([green[0], green[1], green[2] + 0.052]), 255, 500),
            (np.array([green[0] - 0.05, green[1], 0.58]), 255, 900),
        ]
        for target, gripper, steps in sequence:
            self.move_pinch(target, steps=steps, gripper=gripper)

        self.sim.step(800)
        self.sim.save_final_state("/work/final_state.npz")
        return self.sim.block_positions()


if __name__ == "__main__":
    controller = StackController()
    positions = controller.run()
    delta = positions["red"] - positions["green"]
    print("red", positions["red"])
    print("green", positions["green"])
    print("delta", delta)
