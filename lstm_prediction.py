
import numpy as np
import torch
from torch import nn
from config import HISTORY

class Predictor(nn.Module):
    def __init__(self,d):
        super().__init__(); self.a=nn.LSTM(d,64,batch_first=True); self.b=nn.LSTM(64,32,batch_first=True); self.out=nn.Linear(32,d)
    def forward(self,x):
        x,_=self.a(x); x,_=self.b(x); return self.out(x[:,-1])


def predict(sparse):
    shape=sparse.shape[1:]; z=sparse.reshape(len(sparse),-1)
    z=np.concatenate([z.real,z.imag],axis=1); n=len(z); a=int(.7*n); b=int(.85*n)
    assert min(a,b-a,n-b)>HISTORY
    scale=max(np.sqrt(np.mean(z[:a]**2)),1e-20); z=torch.tensor(z/scale,dtype=torch.float32)
    def windows(lo,hi):
        return torch.stack([z[i-HISTORY:i] for i in range(lo+HISTORY,hi)]),z[lo+HISTORY:hi]
    x,y=windows(0,a); vx,vy=windows(a,b); tx,ty=windows(b,n)
    model=Predictor(z.shape[1]); opt=torch.optim.Adam(model.parameters(),lr=.001); loss=nn.MSELoss()
    for epoch in range(100):
        model.train()
        for idx in torch.randperm(len(x)).split(32):
            opt.zero_grad(); l=loss(model(x[idx]),y[idx]); l.backward(); opt.step()
        if epoch%10==0:
            model.eval()
            with torch.no_grad(): print('epoch',epoch,'validation MSE',float(loss(model(vx),vy)))
    model.eval()
    with torch.no_grad():
        out=model(tx).numpy()*scale; print('test normalized MSE',float(loss(model(tx),ty)))
    r,i=np.split(out,2,axis=1)
    return np.fft.ifft((r+1j*i).reshape((-1,)+shape),axis=-1,norm='ortho'),b+HISTORY


