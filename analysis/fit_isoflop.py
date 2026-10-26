import numpy as np
F = {1:1.81,2:1.95,4:2.2427,8:2.796,16:3.904,32:6.118}  # GFLOP/token
data = {
 2e18:{2:3.1766,4:3.1477,8:3.2292,16:3.5412},
 8e18:{2:2.8271,4:2.7428,8:2.7421,16:2.8100},
 2e19:{4:2.5726,8:2.5707,16:2.6016,32:2.6903},
 5e19:{4:2.4602,8:2.4385,16:2.4565,32:2.5197},
 1e20:{4:2.3898,8:2.3619,16:2.3750},
}
rows=[]
for C,d in data.items():
    ks=np.array(sorted(d)); x=np.log2(ks); y=np.array([d[k] for k in ks])
    a,b,c=np.polyfit(x,y,2); xs=-b/(2*a); ks_=2**xs; L=c-b*b/(4*a)
    D=C/((1.6886+0.13842*ks_)*1e9)
    rows.append((C,ks_,D,L)); print(f"C={C:.0e}  k*={ks_:5.2f}  D*={D/1e9:6.2f}B  L*={L:.4f}  (k {list(ks)})")
r=np.array(rows)
for name,i in (("k*",1),("D*",2)):
    for sel,lab in ((slice(None),"C0-C4"),(slice(1,None),"C1-C4"),(slice(2,None),"C2-C4")):
        p=np.polyfit(np.log(r[sel,0]),np.log(r[sel,i]),1)
        print(f"{name} ~ C^{p[0]:.3f} [{lab}] -> at 1e20: {np.exp(np.polyval(p,np.log(1e20))):.3g}, 4.5e20: {np.exp(np.polyval(p,np.log(4.5e20))):.3g}")
