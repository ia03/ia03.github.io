import numpy as np
import mujoco

from sim import Sim


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"missing body: {name}")
    return bid


def ik_solve_hand_xyz(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    target_pos: np.ndarray,
    q_init: np.ndarray,
    *,
    iters: int = 80,
    tol: float = 1e-4,
    damping: float = 5e-2,
    max_dq: float = 0.08,
    q_nom: np.ndarray | None = None,
    null_gain: float = 0.2,
) -> np.ndarray:
    q = q_init.astype(float).copy()
    qmin = model.actuator_ctrlrange[:7, 0]
    qmax = model.actuator_ctrlrange[:7, 1]

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(iters):
        data.qpos[:7] = np.clip(q, qmin, qmax)
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        cur = data.xpos[body_id].copy()
        err = target_pos - cur
        if float(np.linalg.norm(err)) < tol:
            break

        jacp[:] = 0
        jacr[:] = 0
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        j = jacp[:, :7]

        jj_t = j @ j.T
        inv = np.linalg.solve(jj_t + damping * np.eye(3), np.eye(3))
        dq_task = j.T @ (inv @ err)

        dq = dq_task
        if q_nom is not None:
            n = np.eye(7) - j.T @ (inv @ j)
            dq = dq + n @ (null_gain * (q_nom - q))

        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > max_dq:
            dq = dq * (max_dq / dq_norm)
        q = np.clip(q + dq, qmin, qmax)

    return q


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)


def peg_metrics(sim: Sim) -> dict:
    m = sim.model
    d = sim.data
    peg_id = _body_id(m, "peg")
    p = sim.peg_position()
    mat = d.xmat[peg_id].reshape(3, 3)
    x_axis_world = mat[:, 0]
    alignment = float(np.dot(x_axis_world, np.array([1.0, 0.0, 0.0])))
    return {
        "peg_pos": p,
        "x": float(p[0]),
        "y": float(p[1]),
        "z": float(p[2]),
        "x_axis_alignment": alignment,
    }


def run_episode(*, render_debug: bool = False) -> Sim:
    # Plan IK waypoints in a scratch sim (do not accumulate ctrl_trace here).
    scratch = Sim()
    m = scratch.model
    d = scratch.data
    ee_id = _body_id(m, "left_finger")

    q0 = d.qpos[:7].copy()
    q_nom = np.array([0.0, -0.8, 0.0, -2.2, 0.0, 1.6, 0.8])

    # Targets: approach behind tab, descend to shelf height, then push forward.
    y_lane = -0.102
    z_lane = 0.52
    x_behind_tab = 0.395
    x_push_end = 0.545

    p_above = np.array([x_behind_tab, y_lane, 0.70])
    p_contact = np.array([x_behind_tab, y_lane, z_lane])
    p_push_end = np.array([x_push_end, y_lane, z_lane])
    p_lift = np.array([x_push_end, y_lane, 0.65])

    q_above = ik_solve_hand_xyz(m, d, ee_id, p_above, q0, iters=220, q_nom=q_nom)
    q_contact = ik_solve_hand_xyz(m, d, ee_id, p_contact, q_above, iters=260, q_nom=q_nom)
    q_push = ik_solve_hand_xyz(m, d, ee_id, p_push_end, q_contact, iters=260, q_nom=q_nom)
    q_lift = ik_solve_hand_xyz(m, d, ee_id, p_lift, q_push, iters=220, q_nom=q_nom)

    # Execute from canonical start with logged control trace.
    sim = Sim()
    sim.data.ctrl[:] = 0

    def segment(q_start: np.ndarray, q_end: np.ndarray, steps: int):
        for i in range(steps):
            t = (i + 1) / steps
            sim.data.ctrl[:7] = lerp(q_start, q_end, t)
            sim.data.ctrl[7] = 0.0
            sim.step(1)

    # Move to above, descend, push, dwell, then lift away.
    segment(sim.data.qpos[:7].copy(), q_above, 500)
    segment(q_above, q_contact, 500)
    segment(q_contact, q_push, 900)
    segment(q_push, q_push, 300)
    segment(q_push, q_lift, 600)
    segment(q_lift, q_lift, 200)

    if render_debug:
        import imageio.v2 as imageio

        frames = []
        for _ in range(60):
            frames.append(sim.render(640, 480))
            sim.step(1)
        imageio.mimsave("debug.gif", frames, fps=30)

    return sim


def main():
    sim = run_episode(render_debug=False)
    metrics = peg_metrics(sim)
    print("final metrics:", metrics)
    sim.save_final_state("/work/final_state.npz")
    print("saved /work/final_state.npz")


if __name__ == "__main__":
    main()
