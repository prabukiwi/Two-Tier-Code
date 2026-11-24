
import numpy as np
from config import NOISE

def effective(C,theta):
    return C[:,0,:]+np.einsum('n,knm->km',theta,C[:,1:,:])


def sinr(h,w):
    p=abs(h@w)**2; s=np.diag(p); return s/(np.maximum(p.sum(1)-s,0)+NOISE)


