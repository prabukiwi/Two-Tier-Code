
import numpy as np
import torch
from config import NOISE
from channel_metrics import effective, sinr

def passive(C,w,theta):
    # Smooth minimum of log(1+SINR); optimize real phase coordinates on torus.
    # Backtracking accepts only improvements of actual minimum SINR.
    c=torch.tensor(C,dtype=torch.complex128); wt=torch.tensor(w,dtype=torch.complex128)
    for _ in range(7):
        phi=torch.tensor(np.angle(theta),requires_grad=True,dtype=torch.float64)
        h=c[:,0,:]+torch.einsum('n,knm->km',torch.exp(1j*phi),c[:,1:,:]); p=abs(h@wt)**2; s=p.diag(); r=torch.log1p(s/(torch.clamp(p.sum(1)-s,min=0)+NOISE))
        objective=-torch.logsumexp(-10*r,dim=0)/10; objective.backward(); g=phi.grad.numpy(); g/=max(np.linalg.norm(g),1e-30)
        old=np.min(sinr(effective(C,theta),w)); accepted=False
        for alpha in 0.5**np.arange(12):
            trial=np.exp(1j*(np.angle(theta)+alpha*g))
            if np.min(sinr(effective(C,trial),w))>old:
                theta=trial; accepted=True; break
        if not accepted: break
    return theta


