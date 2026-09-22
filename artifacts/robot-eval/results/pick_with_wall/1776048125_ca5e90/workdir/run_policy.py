import numpy as np
import mujoco
from sim import Sim

np.set_printoptions(precision=4, suppress=True)

ARM_DOF = 7
GRIP_OPEN = 255.0
GRIP_CLOSE = 0.0


def set_arm_state(sim, q):
    sim.data.qpos[:ARM_DOF] = q
    sim.data.qvel[:ARM_DOF] = 0
    # keep fingers open for IK calculations
    sim.data.qpos[7] = 0.04
    sim.data.qpos[8] = 0.04
    mujoco.mj_forward(sim.model, sim.data)


def hand_pos(sim):
    hid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    return sim.data.xpos[hid].copy()


def hand_axes(sim):
    hid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    return sim.data.xmat[hid].reshape(3, 3).copy()


def ik_to_pos(sim, target, q_init, max_iters=250, tol=0.004):
    model = sim.model
    data = sim.data
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    jmin = model.jnt_range[:ARM_DOF, 0]
    jmax = model.jnt_range[:ARM_DOF, 1]

    q = q_init.copy()
    damp = 2e-3
    alpha = 0.7

    for _ in range(max_iters):
        set_arm_state(sim, q)
        p = data.xpos[hid].copy()
        err = target - p
        if np.linalg.norm(err) < tol:
            return q, True, np.linalg.norm(err)

        jacp = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacBodyCom(model, data, jacp, None, hid)
        J = jacp[:, :ARM_DOF]

        A = J @ J.T + damp * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)
        q = np.clip(q + alpha * dq, jmin, jmax)

    set_arm_state(sim, q)
    final_err = np.linalg.norm(target - data.xpos[hid])
    return q, final_err < tol * 1.5, final_err


def move_to(sim, q_target, grip, steps=220):
    q_start = sim.data.qpos[:ARM_DOF].copy()
    for i in range(1, steps + 1):
        a = i / steps
        a = 3 * a * a - 2 * a * a * a
        q = (1 - a) * q_start + a * q_target
        sim.data.ctrl[:ARM_DOF] = q
        sim.data.ctrl[7] = grip
        sim.step(1)


def hold(sim, steps, grip=None):
    sim.data.ctrl[:ARM_DOF] = sim.data.qpos[:ARM_DOF]
    if grip is not None:
        sim.data.ctrl[7] = grip
    sim.step(steps)


def eval_progress(sim):
    trace = sim._trace
    cup = np.array([t["cup_pos"] for t in trace])
    contact = np.array([t["cup_contact"] for t in trace])
    best_x = float(np.min(cup[:, 0]))
    best_z = float(np.max(cup[:, 2]))
    final = sim.cup_position()
    replay_steps = len(sim._ctrl_trace)

    grasp = 1.0 if np.any(contact > 0.5) else 0.0
    return_prog = np.clip((0.55 - best_x) / (0.55 - 0.48), 0, 1)
    lift_prog = np.clip((best_z - 0.435) / (0.56 - 0.435), 0, 1)
    final_ret = np.clip((0.55 - final[0]) / (0.55 - 0.42), 0, 1)
    final_h = np.clip((final[2] - 0.435) / (0.56 - 0.435), 0, 1)
    eff = np.clip(2600 / max(replay_steps, 1), 0, 1)
    score = 0.05 * grasp + 0.15 * return_prog + 0.10 * lift_prog + 0.10 * final_ret + 0.25 * final_h + 0.10 * eff

    return {
        "replay_steps": replay_steps,
        "best_x": best_x,
        "best_z": best_z,
        "final_x": float(final[0]),
        "final_z": float(final[2]),
        "any_contact": bool(np.any(contact > 0.5)),
        "score_like": float(score),
    }


def run_attempt():
    sim = Sim()

    # Start open
    sim.data.ctrl[:ARM_DOF] = sim.data.qpos[:ARM_DOF]
    sim.data.ctrl[7] = GRIP_OPEN
    sim.step(250)

    cup0 = sim.cup_position()
    print("cup0", cup0)

    q0 = sim.data.qpos[:ARM_DOF].copy()

    targets = [
        np.array([0.50, 0.15, 0.64]),  # over wall / pregrasp high
        np.array([0.55, 0.15, 0.53]),  # just above cup
        np.array([0.555, 0.15, 0.475]),  # descend for grasp
    ]

    qs = []
    qcur = q0.copy()
    for t in targets:
        q, ok, err = ik_to_pos(sim, t, qcur)
        print("ik", t, "ok", ok, "err", err, "hand", hand_pos(sim), "zaxis", hand_axes(sim)[:, 2])
        qs.append(q)
        qcur = q

    # Execute approach
    for q in qs[:-1]:
        move_to(sim, q, GRIP_OPEN, steps=260)
    move_to(sim, qs[-1], GRIP_OPEN, steps=200)
    hold(sim, 120, GRIP_OPEN)

    # close and lift
    hold(sim, 280, GRIP_CLOSE)

    post_targets = [
        np.array([0.53, 0.14, 0.62]),
        np.array([0.43, 0.10, 0.64]),
        np.array([0.38, 0.05, 0.64]),
    ]
    qcur = sim.data.qpos[:ARM_DOF].copy()
    post_qs = []
    for t in post_targets:
        q, ok, err = ik_to_pos(sim, t, qcur)
        print("ik-post", t, "ok", ok, "err", err, "hand", hand_pos(sim), "zaxis", hand_axes(sim)[:, 2])
        post_qs.append(q)
        qcur = q

    for q in post_qs:
        move_to(sim, q, GRIP_CLOSE, steps=260)

    # settle while still grasping
    hold(sim, 500, GRIP_CLOSE)

    metrics = eval_progress(sim)
    print("metrics", metrics)

    sim.save_final_state('/work/final_state.npz')
    print("saved /work/final_state.npz")


if __name__ == '__main__':
    run_attempt()
