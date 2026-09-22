import numpy as np
from sim import Sim, TOUCH_POINT, TARGET_XY

rng=np.random.default_rng(0)
CHECK=[0.25,0.55,0.80]


def eval_trace(ctrl):
    sim=Sim(); sim.reset()
    n=len(ctrl)
    hit=[False]*3;stable=0;best=1e9
    for u in ctrl:
        sim.data.ctrl[:] = u
        sim.step(1)
        p=sim.pelvis_position();tu=sim.torso_up()
        d=min(np.linalg.norm(sim.elbow_positions()['left']-TOUCH_POINT),np.linalg.norm(sim.elbow_positions()['right']-TOUCH_POINT))
        if d<best: best=d
        for j,c in enumerate(CHECK):
            if p[0]>=c: hit[j]=True
        if p[2]>=0.23 and tu>=0.55: stable += 1
    stable_frac=stable/max(n,1)
    final_target=np.linalg.norm(sim.pelvis_position()[:2]-TARGET_XY)
    minz=9; minup=9; settle_best=9
    for _ in range(500):
        sim.step(1)
        p=sim.pelvis_position();tu=sim.torso_up()
        if p[2]<minz:minz=p[2]
        if tu<minup:minup=tu
        d=min(np.linalg.norm(sim.elbow_positions()['left']-TOUCH_POINT),np.linalg.norm(sim.elbow_positions()['right']-TOUCH_POINT))
        if d<settle_best:settle_best=d
    checkpoint_progress=sum(hit)/3
    touch_progress=1.0 if best<=0.12 else max(0.0,1-(best-0.12)/1.0)
    stability_progress=min(max(stable_frac/0.35,0),1)
    distance_progress=1.0 if final_target<=0.45 else max(0.0,1-(final_target-0.45)/1.0)
    settle_posture_progress=0.5*min(max(minz/0.23,0),1)+0.5*min(max(minup/0.55,0),1)
    efficiency_progress=min(max(2200/max(n,1),0),1)
    score=0.15*checkpoint_progress+0.30*touch_progress+0.20*stability_progress+0.15*distance_progress+0.10*settle_posture_progress+0.10*efficiency_progress
    hard=(n>=30 and all(hit) and stable_frac>=0.35 and best<=0.12 and final_target<=0.45 and minz>=0.23 and minup>=0.55 and settle_best<=0.12)
    return score,hard,(hit,stable_frac,best,final_target,minz,minup,settle_best,sim.pelvis_position()[0])


def make_trace(p):
    n1,n2=int(p[0]),int(p[1])
    u1=np.zeros(19);u2=np.zeros(19)
    # phase1 locomotion
    u1[2]=p[2];u1[7]=p[3];u1[3]=p[4];u1[8]=p[5];u1[4]=p[6];u1[9]=p[7];u1[10]=p[8]
    # slight arm prep
    u1[15]=p[14]*0.4;u1[16]=p[15]*0.4;u1[17]=p[16]*0.4
    # phase2 reach/stabilize
    u2[2]=p[9];u2[7]=p[10];u2[3]=p[11];u2[8]=p[12];u2[10]=p[13]
    u2[15]=p[14];u2[16]=p[15];u2[17]=p[16]
    return np.vstack([np.repeat(u1[None,:],n1,axis=0), np.repeat(u2[None,:],n2,axis=0)])

# seed from current saved
best_score=-1
best_ctrl=None
for i in range(60):
    p=np.zeros(17)
    p[0]=rng.integers(120,420)
    p[1]=rng.integers(80,320)
    p[2]=rng.uniform(-120,120); p[3]=rng.uniform(-120,120)
    p[4]=rng.uniform(-140,160); p[5]=rng.uniform(-140,160)
    p[6]=rng.uniform(-40,40); p[7]=rng.uniform(-40,40)
    p[8]=rng.uniform(-120,120)
    p[9]=rng.uniform(-100,100); p[10]=rng.uniform(-100,100)
    p[11]=rng.uniform(-130,130); p[12]=rng.uniform(-130,130)
    p[13]=rng.uniform(-100,100)
    p[14]=rng.uniform(-35,35); p[15]=rng.uniform(-35,35); p[16]=rng.uniform(-15,15)

    ctrl=make_trace(p)
    s,hard,m=eval_trace(ctrl)
    if s>best_score:
        best_score=s; best_ctrl=ctrl.copy(); best_m=m
        print('iter',i,'score',s,'hard',hard,'m',m,'len',len(ctrl))

# compare existing saved file if exists
if best_ctrl is not None:
    sim=Sim(); sim.reset()
    for u in best_ctrl:
        sim.data.ctrl[:] = u
        sim.step(1)
    sim.save_final_state('/work/final_state.npz')
    print('saved best len',len(best_ctrl),'score',best_score,'m',best_m)
