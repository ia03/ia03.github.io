import numpy as np
import mujoco
from sim import Sim


def bid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


class Runner:
    def __init__(self):
        self.sim = Sim()
        self.m = self.sim.model
        self.d = self.sim.data
        self.lf = bid(self.m, "left_finger")
        self.rf = bid(self.m, "right_finger")
        self.hand = bid(self.m, "hand")
        self.cup = bid(self.m, "cup")
        self.ctrl_min = self.m.actuator_ctrlrange[:7, 0]
        self.ctrl_max = self.m.actuator_ctrlrange[:7, 1]
        self.qhome = np.array([0.0, 0.3, 0.0, -1.2, 0.0, 1.3, 0.0])

    def reset(self):
        self.sim.reset()

    def pinch(self):
        return 0.5 * (self.d.xpos[self.lf] + self.d.xpos[self.rf])

    def cup_pos(self):
        return np.array(self.d.xpos[self.cup])

    def cup_up_z(self):
        return float(self.d.xmat[self.cup].reshape(3, 3)[2, 2])

    def step_to(self, target, grip, steps, k=4.0, damp=1e-3, reg=0.04):
        for _ in range(steps):
            mujoco.mj_forward(self.m, self.d)
            e = target - self.pinch()
            Jl = np.zeros((3, self.m.nv))
            Jr = np.zeros((3, self.m.nv))
            z = np.zeros((3, self.m.nv))
            mujoco.mj_jacBodyCom(self.m, self.d, Jl, z, self.lf)
            mujoco.mj_jacBodyCom(self.m, self.d, Jr, z, self.rf)
            J = 0.5 * (Jl[:, :7] + Jr[:, :7])
            A = J.T @ J + damp * np.eye(7)
            b = J.T @ (k * e) - reg * (self.d.qpos[:7] - self.qhome)
            dq = np.linalg.solve(A, b)
            self.d.ctrl[:7] = np.clip(self.d.qpos[:7] + dq, self.ctrl_min, self.ctrl_max)
            self.d.ctrl[7] = grip
            self.sim.step(1)

    def run(self, p, save_path=None):
        self.reset()
        c0 = self.cup_pos().copy()
        replay_contacts = []
        best_z = c0[2]

        def stage(target, grip, steps):
            nonlocal best_z
            self.step_to(target, grip, steps)
            best_z = max(best_z, float(self.cup_pos()[2]))

        stage(np.array([c0[0] + p[0], c0[1] + p[1], p[2]]), p[8], 700)   # high pregrasp
        stage(np.array([c0[0] + p[3], c0[1] + p[4], p[5]]), p[8], 700)   # descend

        for _ in range(int(p[9])):
            self.d.ctrl[:7] = self.d.qpos[:7]
            self.d.ctrl[7] = 0.0
            self.sim.step(1)
            replay_contacts.append(self.sim.has_gripper_cup_contact())
            best_z = max(best_z, float(self.cup_pos()[2]))

        stage(np.array([c0[0] + p[6], c0[1] + p[7], p[10]]), 0.0, 900)   # lift+small move
        stage(np.array([0.70 + p[11], 0.14 + p[12], p[13]]), 0.0, 900)   # to bin high
        stage(np.array([0.70 + p[11], 0.14 + p[12], p[14]]), 0.0, 700)   # lower
        stage(np.array([0.70 + p[11], 0.14 + p[12], p[15]]), 255.0, 900) # release+raise
        stage(np.array([0.58, 0.16, 0.62]), 255.0, 700)                   # retreat

        # record contact for replay criteria based on whole trajectory end tail
        trace_contact = np.array([t['cup_contact'] for t in self.sim._trace], dtype=float)

        self.sim.step(500)
        settle_contacts = []
        for _ in range(100):
            settle_contacts.append(float(self.sim.has_gripper_cup_contact()))
            self.sim.step(1)

        cup = self.cup_pos()
        upz = abs(self.cup_up_z())
        settle_frac = float(np.mean(settle_contacts))
        inside = abs(cup[0] - 0.70) <= 0.046 and abs(cup[1] - 0.14) <= 0.046
        c0dist = np.linalg.norm(c0[:2] - np.array([0.70, 0.14]))
        cfdist = np.linalg.norm(cup[:2] - np.array([0.70, 0.14]))

        grasp_progress = 1.0 if np.any(trace_contact > 0.5) else 0.0
        lift_progress = np.clip((best_z - 0.435) / 0.065, 0, 1)
        inside_bin_progress = 1.0 if inside else 0.0
        placement_progress = 1.0 if inside else np.clip(1 - cfdist / c0dist, 0, 1)
        release_progress = np.clip(1 - settle_frac / 0.1, 0, 1)
        upright_progress = np.clip(upz / 0.85, 0, 1)
        score = (0.05 * grasp_progress + 0.10 * lift_progress + 0.40 * inside_bin_progress +
                 0.10 * placement_progress + 0.10 * release_progress + 0.25 * upright_progress)

        last10 = float(np.mean(trace_contact[-10:])) if trace_contact.size >= 10 else float(np.mean(trace_contact))
        passed = (
            len(self.sim._ctrl_trace) >= 20 and
            np.any(trace_contact > 0.5) and
            best_z >= 0.50 and
            last10 < 0.5 and
            settle_frac < 0.1 and
            inside and
            cup[2] >= 0.42 and
            upz >= 0.85
        )

        if save_path is not None:
            self.sim.save_final_state(save_path)

        return {
            'score': float(score),
            'pass': bool(passed),
            'best_z': float(best_z),
            'cup': cup.copy(),
            'upz': float(upz),
            'settle_frac': float(settle_frac),
            'inside': bool(inside),
            'contact_any': bool(np.any(trace_contact > 0.5)),
            'last10': float(last10),
        }


def sample(rng):
    p = np.zeros(16)
    p[0] = rng.uniform(-0.01, 0.01)
    p[1] = rng.uniform(-0.01, 0.01)
    p[2] = rng.uniform(0.62, 0.70)
    p[3] = rng.uniform(-0.02, 0.02)
    p[4] = rng.uniform(-0.02, 0.02)
    p[5] = rng.uniform(0.445, 0.475)
    p[6] = rng.uniform(-0.02, 0.04)
    p[7] = rng.uniform(-0.02, 0.04)
    p[8] = rng.uniform(80, 220)
    p[9] = rng.integers(400, 1100)
    p[10] = rng.uniform(0.54, 0.62)
    p[11] = rng.uniform(-0.01, 0.01)
    p[12] = rng.uniform(-0.01, 0.01)
    p[13] = rng.uniform(0.56, 0.64)
    p[14] = rng.uniform(0.46, 0.52)
    p[15] = rng.uniform(0.52, 0.60)
    return p


def main():
    rng = np.random.default_rng(0)
    runner = Runner()

    best = None
    best_p = None
    n_trials = 24
    for i in range(n_trials):
        p = sample(rng)
        out = runner.run(p)
        if best is None or out['score'] > best['score']:
            best, best_p = out, p.copy()
        print(f"trial {i:02d} score={out['score']:.3f} pass={out['pass']} contact={out['contact_any']} best_z={out['best_z']:.3f} inside={out['inside']} cup={out['cup']}")
        if out['pass']:
            best, best_p = out, p.copy()
            break

    final = runner.run(best_p, save_path='/work/final_state.npz')
    print('best params', best_p)
    print('final', final)


if __name__ == '__main__':
    main()
