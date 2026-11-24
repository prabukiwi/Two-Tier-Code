
import numpy as np
from config import K, M, N

def load(path):
    d=np.load(path); hd,hr,G=(d[k] for k in ('hd','hr','G'))
    assert hd.shape[1:]==(K,M) and hr.shape==(len(hd),K,N) and G.shape==(len(hd),N,M)
    assert all(np.isfinite(x).all() for x in (hd,hr,G))
    # Preserve IRS-element contributions so phases can be optimized after prediction.
    return np.concatenate([hd[:,:,None,:],hr[:,:,:,None]*G[:,None,:,:]],axis=2)


