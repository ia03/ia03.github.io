import numpy as np
import mujoco
import sim


def build_phase_controller(sim_obj):
    model = sim_obj.model
    data = sim_obj.data
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    # Control bounds for 7 arm joints
    ctrl_min = model.actuator_ctrlrange[:7, 0].copy()
    ctrl_max = model.actuator_ctrlrange[:7, 1].copy()

    def gripper_midpoint_world():
        lf = data.xpos[left_id]
        rf = data.xpos[right_id]
        return 0.5 * (lf + rf)

    def move_to(target, steps=200, gain=0.8, damping=1e-3, hold=0):
        """Move hand fingertip midpoint toward target xyz."""
        for _ in range(steps):
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
            J = jacp[:, :7]
            pos = gripper_midpoint_world()
            err = target - pos
            # damped least squares step in joint space
            A = J @ J.T + damping * np.eye(3)
            dq = J.T @ np.linalg.solve(A, err)
            dq = np.clip(dq, -0.35, 0.35)
            q = data.qpos[:7]
            qtarget = np.clip(q + gain * dq, ctrl_min, ctrl_max)
            data.ctrl[:7] = qtarget
            sim_obj.step()

        # allow small settling at fixed target
        for _ in range(hold):
            sim_obj.step()

    def set_fingers(value, steps=80):
        v = np.clip(value, 0, 255)
        start = float(data.ctrl[7])
        for i in range(steps):
            if steps > 1:
                frac = (i + 1) / steps
                data.ctrl[7] = start + (v - start) * frac
            else:
                data.ctrl[7] = v
            sim_obj.step()

    return move_to, set_fingers


def run():
    sim_obj = sim.Sim()
    move_to, set_fingers = build_phase_controller(sim_obj)

    # Start from home pose as a stable base.
    home_ctrl = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 255.0])
    sim_obj.data.ctrl[:] = home_ctrl
    sim_obj.step(250)

    # Open hand slightly and approach red block from above.
    set_fingers(255, steps=30)
    move_to(np.array([0.48, -0.10, 0.55]), steps=260, gain=1.2, hold=40)
    move_to(np.array([0.48, -0.10, 0.44]), steps=220, gain=1.1, hold=30)

    # Grasp red.
    set_fingers(30, steps=120)
    sim_obj.step(80)

    # Lift.
    move_to(np.array([0.50, -0.10, 0.59]), steps=220, gain=1.1, hold=50)

    # Transport to green stack target.
    move_to(np.array([0.63, 0.08, 0.59]), steps=260, gain=1.0, hold=40)
    move_to(np.array([0.63, 0.08, 0.50]), steps=150, gain=1.0, hold=30)

    # Place + release, then settle above stack.
    set_fingers(255, steps=140)
    move_to(np.array([0.63, 0.08, 0.58]), steps=150, gain=0.9, hold=80)

    # final settle before recording
    sim_obj.step(600)

    sim_obj.save_final_state("/work/final_state.npz")
    red = sim_obj.block_positions()["red"]
    green = sim_obj.block_positions()["green"]
    print("final red", red)
    print("final green", green)
    d = red - green
    print("dx,dy,dz", d)


if __name__ == "__main__":
    run()
