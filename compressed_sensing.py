
import numpy as np
from config import K, M, N, P, NOISE

def acquire(H,rng):
    # Illustrative pilot observation at the AP; IRS contributions require
    # separate identifiable sounding states in a physical implementation.
    # Here each component is observed separately: optimistic sounding assumption.
    pilots=np.eye(P)[np.arange(K)%P]
    Y=np.einsum('kp,tkam->tpam',pilots,H)
    Y+=np.sqrt(NOISE/2)*(rng.normal(size=Y.shape)+1j*rng.normal(size=Y.shape))
    # Sparse angular representation across AP antennas. Shared pilots alone
    # cannot distinguish users with identical support; minimum-norm sparse proxy.
    A=pilots.T
    Z=np.fft.fft(Y,axis=-1,norm='ortho')
    x=np.zeros((len(H),K,N+1,M),complex)
    step=1/np.linalg.norm(A,2)**2
    lam=2*np.sqrt(NOISE)
    for _ in range(40):
        residual=np.einsum('pk,tkam->tpam',A,x)-Z
        u=x-step*np.einsum('pk,tpam->tkam',A,residual)
        x=u*np.maximum(0,1-step*lam/np.maximum(abs(u),1e-30))
    return x


