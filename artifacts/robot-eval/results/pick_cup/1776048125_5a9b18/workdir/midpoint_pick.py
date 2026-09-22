import numpy as np
import mujoco
from sim import Sim


def get_ids(m):
    return {
        'cup': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'cup'),
        'left': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'left_finger'),
        'right': mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_finger'),
    }


def midpoint_and_jac(sim, I):
    m, d = sim.model, sim.data
    lp = d.xpos[I['left']].copy()
    rp = d.xpos[I['right']].copy()
    mid = 0.5 * (lp + rp)

    jl_p = np.zeros((3, m.nv)); jl_r = np.zeros((3, m.nv))
    jr_p = np.zeros((3, m.nv)); jr_r = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jl_p, jl_r, I['left'])
    mujoco.mj_jacBody(m, d, jr_p, jr_r, I['right'])
    J = 0.5 * (jl_p[:, :7] + jr_p[:, :7])
    return mid, J


def track_midpoint_step(sim, q_target, target, gain=2.5, damp=1e-3):
    m, d = sim.model, sim.data
    I = get_ids(m)
    mid, J = midpoint_and_jac(sim, I)
    e = target - mid
    A = J @ J.T + damp * np.eye(3)
    dq = J.T @ np.linalg.solve(A, gain * e)
    dq = np.clip(dq, -0.04, 0.04)
    q_target = q_target + dq
    for j in range(7):
        lo, hi = m.jnt_range[j]
        q_target[j] = np.clip(q_target[j], lo, hi)
    d.ctrl[:7] = q_target
    return q_target, np.linalg.norm(e)


def run_once(params):
    sim = Sim(); m,d=sim.model,sim.data
    I = get_ids(m)
    q_target = np.array(params['q_seed'], dtype=float)

    # initialize to bent pose to avoid singular straight-arm local minima
    d.ctrl[:7] = q_target
    d.ctrl[7] = 255
    sim.step(600)

    def phase(target, steps, grip):
        nonlocal q_target
        target = np.array(target, dtype=float)
        for _ in range(steps):
            q_target, _ = track_midpoint_step(sim, q_target, target, gain=params['gain'])
            d.ctrl[7] = grip
            sim.step(1)

    cup = sim.cup_position().copy()
    phase([cup[0], cup[1], cup[2] + params['z_above']], params['n_above'], 255)
    phase([cup[0], cup[1], cup[2] + params['z_pre']], params['n_pre'], 255)
    phase([cup[0], cup[1], cup[2] + params['z_grasp']], params['n_grasp'], 230)

    # close while holding pose
    for t in range(params['n_close']):
        q_target, _ = track_midpoint_step(sim, q_target, np.array([cup[0], cup[1], cup[2] + params['z_grasp']]), gain=params['gain'])
        a = (t + 1) / params['n_close']
        d.ctrl[7] = (1 - a) * 230 + a * params['g_close']
        sim.step(1)

    phase([cup[0], cup[1], cup[2] + params['z_grasp']], params['n_hold'], params['g_close'])
    phase([cup[0], cup[1], cup[2] + params['z_lift']], params['n_lift'], params['g_close'])
    phase([cup[0], cup[1], cup[2] + params['z_lift']], params['n_final'], params['g_close'])
    return sim, np.array(sim._ctrl_trace, dtype=float)


def replay_eval(ctrl_trace):
    sim = Sim(); m,d=sim.model,sim.data
    I=get_ids(m); cup_bid=I['cup']; finger_bodies={I['left'],I['right']}
    for u in ctrl_trace:
        d.ctrl[:] = u
        sim.step(1)
    hit=0
    for _ in range(500):
        sim.step(1)
        touched=False
        for i in range(d.ncon):
            c=d.contact[i]
            b1=m.geom_bodyid[c.geom1]; b2=m.geom_bodyid[c.geom2]
            if (cup_bid in (b1,b2)) and ((b1 in finger_bodies) or (b2 in finger_bodies)):
                touched=True; break
        if touched: hit+=1
    cup_z=sim.cup_position()[2]
    cf=hit/500
    hp=np.clip((cup_z-0.435)/(0.52-0.435),0,1)
    cp=np.clip(cf/0.5,0,1)
    return {'cup_z':float(cup_z),'contact_frac':float(cf),'score':float(0.5*hp+0.5*cp),'success': bool(cup_z>=0.52 and cf>=0.5)}


def main():
    seeds = [
        [-1.139,-0.354,1.048,-2.286,0.32,2.535,2.825],
        [-2.384,-1.687,1.485,-1.88,-2.082,0.196,-2.159],
        [-0.141,-0.031,0.111,-1.921,-0.109,1.639,0.904],
    ]
    best=None
    tid=0
    for q_seed in seeds:
        for z_grasp in [0.01, 0.0, -0.005, -0.01]:
            for g_close in [0, 5, 10, 15]:
                tid += 1
                p = dict(q_seed=q_seed,gain=2.5,z_above=0.10,z_pre=0.04,z_grasp=z_grasp,z_lift=0.18,
                         n_above=280,n_pre=220,n_grasp=220,n_close=220,n_hold=120,n_lift=360,n_final=180,g_close=g_close)
                sim,trace=run_once(p)
                ev=replay_eval(trace)
                if best is None or ev['score']>best[0]['score']:
                    sim.save_final_state('/work/final_state.npz')
                    best=(ev,p)
                    flag='BEST'
                else:
                    flag='----'
                print(f"trial {tid:02d} {flag} score={ev['score']:.3f} z={ev['cup_z']:.3f} cf={ev['contact_frac']:.3f} success={ev['success']} g={g_close} zg={z_grasp} seed0={q_seed[0]:.2f}")
                if ev['success']:
                    sim.save_final_state('/work/final_state.npz')
                    print('SUCCESS')
                    print(best)
                    return
    print('best',best)

if __name__=='__main__':
    main()
