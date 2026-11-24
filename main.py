
import argparse, time, json
import numpy as np
from scipy.special import ndtr
from config import seed, SEED, BW
from channel_data import load
from compressed_sensing import acquire
from lstm_prediction import predict
from channel_metrics import effective, sinr
from two_tier_optimization import optimize

def main():
    p=argparse.ArgumentParser(); p.add_argument('channels'); p.add_argument('--limit',type=int,default=10); p.add_argument('--blocks',type=int,default=1); p.add_argument('--blocklength',type=int,default=256); p.add_argument('--rate',type=float,default=1.0); p.add_argument('--circuit-watts',type=float,default=None); a=p.parse_args(); seed(); rng=np.random.default_rng(SEED)
    H=load(a.channels); S=acquire(H,rng); predicted,start=predict(S); rows=[]
    for j,C in enumerate(predicted[:a.limit]):
        t=time.perf_counter(); w,theta=optimize(C,rng); runtime=(time.perf_counter()-t)*1000
        g=sinr(effective(H[start+j],theta),w); se=np.log2(1+g); dispersion=(1-(1+g)**-2)*np.log2(np.e)**2
        bler=ndtr((a.rate-se)*np.sqrt(a.blocklength/np.maximum(dispersion,1e-15)))
        row={'slot':start+j,'min_sinr_db':float(10*np.log10(max(g.min(),1e-30))),'sum_se_bps_hz':float(se.sum()),'optimization_ms':runtime,'modeled_bler_percent':float(100*bler.mean()),'modeled_pdr_percent':float(100*np.mean((1-bler)**a.blocks))}
        if a.circuit_watts is not None: row['modeled_ee_bits_j']=float(BW*se.sum()/(np.sum(abs(w)**2)+a.circuit_watts))
        rows.append(row); print(row)
    with open('verification_results.json','w') as f: json.dump(rows,f,indent=2)


if __name__ == '__main__':
    main()
