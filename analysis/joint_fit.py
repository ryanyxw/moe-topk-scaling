"""Joint fit L(k, D) = E + A * F(k)^-alpha + B * D^-beta over all cooldown points (top-k >= 2).
F(k) = FLOPs/token (proxy for active compute). Then k*(C) = argmin_k L(k, C/F(k))."""
import numpy as np, sys
from scipy.optimize import minimize
F = {1:1.81,2:1.95,4:2.2427,8:2.796,12:3.35,16:3.904,32:6.118}
lk=np.log2(list(F)); lf=np.log(list(F.values()))
fpt=lambda k: np.exp(np.interp(np.log2(k),lk,lf))*1e9
pts = {  # budget: {k: val loss}
 2e18:{2:3.1766,4:3.1477,8:3.2292,16:3.5412},
 8e18:{2:2.8271,4:2.7428,8:2.7421,16:2.8100},
 2e19:{4:2.5726,8:2.5707,16:2.6016,32:2.6903},
 5e19:{4:2.4602,8:2.4385,16:2.4565,32:2.5197},
 1e20:{8:2.3619},
}
for a in sys.argv[1:]:  # extra points "C:k:loss"
    C,k,l=a.split(":"); pts.setdefault(float(C),{})[int(k)]=float(l)
min_c = float(__import__("os").environ.get("MIN_C", 0))
X=np.array([(C,k,C/fpt(k),l) for C,d in pts.items() if C>=min_c for k,l in d.items()])
Fk=np.array([fpt(k) for k in X[:,1]])/1e9; D=X[:,2]/1e9; L=X[:,3]
def pred(p,Fk,D):
    e,la,al,lb,be=p
    return np.exp(e)+np.exp(la)*Fk**-al+np.exp(lb)*D**-be
def huber(r,d=1e-3): a=np.abs(r); return np.where(a<d,.5*r*r,d*(a-.5*d)).sum()
best=None
for e0 in (0.5,1.0,1.5):
  for al0 in (0.3,1.0,2.0):
    for be0 in (0.2,0.4,0.6):
      r=minimize(lambda p: huber(np.log(pred(p,Fk,D))-np.log(L)),[e0,0.,al0,0.5,be0],method="L-BFGS-B",
                 bounds=[(-3,2),(-10,10),(0,4),(-10,10),(0,2)])
      if best is None or r.fun<best.fun: best=r
p=best.x; e,la,al,lb,be=p
print(f"E={np.exp(e):.3f} A={np.exp(la):.3f} alpha={al:.3f} B={np.exp(lb):.3f} beta={be:.3f}  (n={len(L)}, F in GFLOP, D in B tokens)")
res=L-pred(p,Fk,D); print(f"residual rms={np.sqrt((res**2).mean()):.4f} max={np.abs(res).max():.4f}")
for (C,k,_,l),r in zip(X,res): print(f"  C={C:.0e} k={int(k):2d} loss={l:.4f} resid={r:+.4f}")
ks=2**np.linspace(0.5,6,400)
for C in (2e18,8e18,2e19,5e19,1e20,2e20,4.5e20,1e21):
    Ls=pred(p,fpt(ks)/1e9,C/fpt(ks)/1e9); i=Ls.argmin()
    print(f"C={C:.1e}: k*={ks[i]:5.1f} D*={C/fpt(ks[i])/1e9:6.1f}B L*={Ls[i]:.4f}  L(k=8)={pred(p,fpt(8)/1e9,C/fpt(8)/1e9):.4f} L(k=12)={pred(p,fpt(12)/1e9,C/fpt(12)/1e9):.4f} L(k=16)={pred(p,fpt(16)/1e9,C/fpt(16)/1e9):.4f}")
