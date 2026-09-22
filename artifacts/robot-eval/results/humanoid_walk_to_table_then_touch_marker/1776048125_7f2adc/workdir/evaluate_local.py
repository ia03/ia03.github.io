import numpy as np
from sim import Sim, TOUCH_POINT, TARGET_XY

CHECKPOINTS=[0.25,0.55,0.80]


def run_ctrl_trace(ctrl_trace):
    sim=Sim()
    sim.reset()
    n=ctrl_trace.shape[0]
    pelvis0=sim.pelvis_position().copy()
    hit=[False]*len(CHECKPOINTS)
    stable=0
    best_touch=1e9
    for i in range(n):
        sim.data.ctrl[:] = ctrl_trace[i]
        sim.step(1)
        p=sim.pelvis_position();tu=sim.torso_up()
        d=min(np.linalg.norm(sim.elbow_positions()['left']-TOUCH_POINT),np.linalg.norm(sim.elbow_positions()['right']-TOUCH_POINT))
        best_touch=min(best_touch,float(d))
        for j,c in enumerate(CHECKPOINTS):
            if p[0]>=c: hit[j]=True
        if p[2]>=0.23 and tu>=0.55: stable += 1
    stable_frac = stable/max(n,1)
    final_target=float(np.linalg.norm(sim.pelvis_position()[:2]-TARGET_XY))
    min_settle_z=1e9;min_settle_up=1e9;settle_best_touch=1e9
    for i in range(500):
        sim.step(1)
        p=sim.pelvis_position();tu=sim.torso_up()
        min_settle_z=min(min_settle_z,float(p[2]))
        min_settle_up=min(min_settle_up,float(tu))
        d=min(np.linalg.norm(sim.elbow_positions()['left']-TOUCH_POINT),np.linalg.norm(sim.elbow_positions()['right']-TOUCH_POINT))
        settle_best_touch=min(settle_best_touch,float(d))

    pass_cond = (
        n>=30 and np.linalg.norm(pelvis0[:2])<=0.15 and all(hit) and stable_frac>=0.35 and best_touch<=0.12 and
        final_target<=0.45 and min_settle_z>=0.23 and min_settle_up>=0.55 and settle_best_touch<=0.12
    )
    checkpoint_progress=sum(hit)/len(hit)
    touch_progress=1.0 if best_touch<=0.12 else max(0.0,1-(best_touch-0.12)/1.0)
    stability_progress=min(max(stable_frac/0.35,0),1)
    distance_progress=1.0 if final_target<=0.45 else max(0.0,1-(final_target-0.45)/1.0)
    settle_posture_progress=0.5*min(max(min_settle_z/0.23,0),1)+0.5*min(max(min_settle_up/0.55,0),1)
    efficiency_progress=min(max(2200/max(n,1),0),1)
    progress_score=0.15*checkpoint_progress+0.30*touch_progress+0.20*stability_progress+0.15*distance_progress+0.10*settle_posture_progress+0.10*efficiency_progress
    return dict(pass_cond=pass_cond,n=n,pelvis0=pelvis0,hit=hit,stable_frac=stable_frac,best_touch=best_touch,final_target=final_target,min_settle_z=min_settle_z,min_settle_up=min_settle_up,settle_best_touch=settle_best_touch,score=progress_score)

if __name__=='__main__':
    import sys
    data=np.load(sys.argv[1])
    res=run_ctrl_trace(data['ctrl_trace'])
    for k,v in res.items(): print(k,v)
