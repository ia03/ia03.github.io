import numpy as np
import mujoco
from sim import Sim

ARM=7
OPEN=255.0
CLOSE=0.0
L_TIP=69
R_TIP=77

Q_GRASP_BASE=np.array([-1.21088501,-0.65125911,1.08584398,-2.23071577,1.53703729,1.87211929,2.2199704])


def tip_mid(sim):
    d=sim.data
    return 0.5*(d.geom_xpos[L_TIP]+d.geom_xpos[R_TIP])


def tip_sep(sim):
    d=sim.data
    return d.geom_xpos[R_TIP]-d.geom_xpos[L_TIP]


def set_state_open(sim,q):
    d=sim.data
    d.qpos[:ARM]=q; d.qvel[:ARM]=0
    d.qpos[7]=0.04; d.qpos[8]=0.04
    mujoco.mj_forward(sim.model,d)


def ik_tip_mid(sim,target,q0,iters=260):
    m=sim.model; d=sim.data
    jmin=m.jnt_range[:ARM,0]; jmax=m.jnt_range[:ARM,1]
    q=q0.copy()
    for _ in range(iters):
        set_state_open(sim,q)
        err=target-tip_mid(sim)
        if np.linalg.norm(err)<0.003:
            return q,True
        jl=np.zeros((3,m.nv)); jr=np.zeros((3,m.nv))
        mujoco.mj_jacGeom(m,d,jl,None,L_TIP)
        mujoco.mj_jacGeom(m,d,jr,None,R_TIP)
        J=0.5*(jl[:,:ARM]+jr[:,:ARM])
        dq=J.T@np.linalg.solve(J@J.T+2e-3*np.eye(3),err)
        dq += 0.02*(Q_GRASP_BASE-q)
        q=np.clip(q+0.65*dq,jmin,jmax)
    return q,False


def move_to(sim,q_target,grip,steps=320):
    d=sim.data
    q_start=d.qpos[:ARM].copy()
    for i in range(1,steps+1):
        a=i/steps; a=3*a*a-2*a*a*a
        d.ctrl[:ARM]=(1-a)*q_start+a*q_target
        d.ctrl[7]=grip
        sim.step(1)


def hold(sim,steps,grip):
    d=sim.data
    d.ctrl[:ARM]=d.qpos[:ARM]
    d.ctrl[7]=grip
    sim.step(steps)


def metrics(sim):
    tr=sim._trace
    cup=np.array([t['cup_pos'] for t in tr])
    con=np.array([t['cup_contact'] for t in tr])
    final=sim.cup_position()
    return {
        'any_contact': bool(np.any(con>0.5)),
        'best_x': float(np.min(cup[:,0])),
        'best_z': float(np.max(cup[:,2])),
        'final_x': float(final[0]),
        'final_z': float(final[2]),
        'steps': len(sim._ctrl_trace),
    }


def score(m):
    s=0.0
    s += 2.0 if m['any_contact'] else 0.0
    s += max(0.0, min(1.0,(0.55-m['best_x'])/0.07))
    s += max(0.0, min(1.0,(m['best_z']-0.435)/0.125))
    s += max(0.0, min(1.0,(0.55-m['final_x'])/0.13))
    s += max(0.0, min(1.0,(m['final_z']-0.435)/0.125))
    return s


def run_attempt(z_touch=0.435, ret_x=0.38):
    sim=Sim(); d=sim.data
    hold(sim,180,OPEN)

    q=Q_GRASP_BASE.copy()
    q_high,_=ik_tip_mid(sim,np.array([0.49,0.08,0.66]),q)
    q_pre,_=ik_tip_mid(sim,np.array([0.55,0.15,0.52]),q_high)
    q_touch,_=ik_tip_mid(sim,np.array([0.548,0.152,z_touch]),q_pre)
    q_lift,_=ik_tip_mid(sim,np.array([0.52,0.13,0.66]),q_touch)
    q_cross,_=ik_tip_mid(sim,np.array([0.44,0.08,0.66]),q_lift)
    q_ret,_=ik_tip_mid(sim,np.array([ret_x,0.03,0.64]),q_cross)

    move_to(sim,q_high,OPEN,320)
    move_to(sim,q_pre,OPEN,340)
    move_to(sim,q_touch,OPEN,260)
    hold(sim,80,OPEN)
    hold(sim,320,CLOSE)

    # mandatory early saved attempt
    sim.save_final_state('/work/final_state.npz')

    move_to(sim,q_lift,CLOSE,360)
    move_to(sim,q_cross,CLOSE,340)
    move_to(sim,q_ret,CLOSE,340)
    hold(sim,700,CLOSE)

    m=metrics(sim)
    return sim,m


if __name__=='__main__':
    candidates=[(0.438,0.38),(0.433,0.36),(0.428,0.34),(0.442,0.40)]
    best_s=-1e9
    best_m=None
    for i,(zt,rx) in enumerate(candidates,1):
        sim,m=run_attempt(zt,rx)
        s=score(m)
        print('attempt',i,'z_touch',zt,'ret_x',rx,'metrics',m,'score',s)
        if s>best_s:
            best_s=s
            best_m=m
            sim.save_final_state('/work/final_state.npz')
            print('saved improved state')
    print('best',best_m,'score',best_s)
