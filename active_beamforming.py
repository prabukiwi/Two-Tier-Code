
import numpy as np
import cvxpy as cp
from config import M, K, POWER, NOISE, TOL
from channel_metrics import sinr

def active(h,rng):
    # Scale channels by noise to improve SDP conditioning.
    h=h/np.sqrt(NOISE); Q=[np.outer(v.conj(),v) for v in h]
    W=[cp.Variable((M,M),hermitian=True) for _ in range(K)]
    gamma=cp.Parameter(nonneg=True)
    constraints=[w >> 0 for w in W]+[sum(cp.real(cp.trace(w)) for w in W)<=POWER]
    for k in range(K):
        s=cp.real(cp.trace(Q[k]@W[k])); interference=sum(cp.real(cp.trace(Q[k]@W[j])) for j in range(K) if j!=k)
        constraints.append(s>=gamma*(1+interference))
    prob=cp.Problem(cp.Minimize(0),constraints); lo=0.; hi=float(POWER*np.min(np.sum(abs(h)**2,axis=1))); best=None
    for _ in range(24):
        gamma.value=(lo+hi)/2
        try: prob.solve(solver='CLARABEL')
        except cp.SolverError: prob.solve(solver='SCS',eps=1e-5)
        if prob.status in ('optimal','optimal_inaccurate') and all(w.value is not None for w in W):
            lo=gamma.value; best=[w.value.copy() for w in W]
        else: hi=gamma.value
        if hi-lo<TOL: break
    if best is None: raise RuntimeError('SDP failed')
    factors=[]; principal=[]
    for w in best:
        val,vec=np.linalg.eigh(w); val=np.maximum(val,0); factors.append(vec*np.sqrt(val)); principal.append(vec[:,-1]*np.sqrt(val[-1]))
    candidates=[np.column_stack(principal)]
    for _ in range(32):
        candidates.append(np.column_stack([f@(rng.normal(size=M)+1j*rng.normal(size=M))/np.sqrt(2) for f in factors]))
    candidates=[w*np.sqrt(POWER/max(np.sum(abs(w)**2),1e-30)) for w in candidates]
    return max(candidates,key=lambda w:np.min(sinr(h*np.sqrt(NOISE),w)))


