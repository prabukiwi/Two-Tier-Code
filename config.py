"""T2OF reference implementation module; not original experimental code."""
import numpy as np
import torch

M,N,K,P=8,64,8,4
FC=3.5e9; BW=10e6; POWER=1.; NOISE=10**((-97-30)/10)
SEED=42; HISTORY=10; TOL=1e-4

def seed():
    np.random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


