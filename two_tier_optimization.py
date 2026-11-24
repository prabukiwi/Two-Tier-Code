"""T2OF reference implementation module; not original experimental code."""
import numpy as np
from config import N, TOL
from channel_metrics import effective, sinr
from active_beamforming import active
from passive_beamforming import passive

def optimize(C,rng):
    theta=np.exp(2j*np.pi*rng.random(N)); best=-1; result=None
    for _ in range(7):
        w=active(effective(C,theta),rng); trial=passive(C,w,theta); score=np.min(sinr(effective(C,trial),w))
        if score<=best+TOL: break
        best=score; theta=trial; result=(w,theta)
    if result is None: raise RuntimeError('No feasible iterate')
    return result


