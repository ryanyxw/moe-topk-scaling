import numpy as np, matplotlib, io, os, re
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "figures")
matplotlib.use("svg")
import matplotlib.pyplot as plt
FG,MU,LN="#010101","#020202","#030303"
plt.rcParams.update({"svg.fonttype":"none","font.family":"sans-serif","font.size":10.5,
 "text.color":FG,"axes.labelcolor":FG,"xtick.color":MU,"ytick.color":MU,"axes.edgecolor":LN,
 "axes.spines.top":False,"axes.spines.right":False,"grid.color":LN,"grid.linewidth":0.6,
 "figure.facecolor":"none","axes.facecolor":"none","legend.frameon":False})
F={1:1.81,2:1.95,4:2.2427,8:2.796,16:3.904,32:6.118}
lk=np.log2(list(F)); lf=np.log(list(F.values()))
fpt=lambda k: (1.6886+0.13842*np.asarray(k,dtype=float))*1e9  # GFLOP/token, linear in k (exact for k>=2)
data={2e18:{2:3.1766,4:3.1477,8:3.2292,16:3.5412},8e18:{2:2.8271,4:2.7428,8:2.7421,16:2.8100},
 2e19:{4:2.5726,8:2.5707,16:2.6016,32:2.6903},5e19:{4:2.4602,8:2.4385,16:2.4565,32:2.5197},1e20:{4:2.3898,8:2.3619,16:2.3750}}
k1={2e18:3.1430,8e18:2.7574}
col={2e18:"#8a96a8",8e18:"#3f88c5",2e19:"#e09f3e",5e19:"#d1495b",1e20:"#2a9d6f"}
lab={2e18:"C0 2e18",8e18:"C1 8e18",2e19:"C2 2e19",5e19:"C3 5e19",1e20:"C4 1e20"}
def svg(fig):
    b=io.StringIO(); fig.savefig(b,format="svg",bbox_inches="tight"); s=b.getvalue()
    s=s[s.index("<svg"):]
    s=re.sub(r'<svg([^>]*?) width="[^"]*" height="[^"]*"',r'<svg\1 style="width:100%;height:auto"',s,count=1)
    for c,v in ((FG,"var(--fg)"),(MU,"var(--muted)"),(LN,"var(--line)")): s=s.replace(c,v)
    return s
# --- isoFLOP plot
fig,ax=plt.subplots(figsize=(7.2,4.4)); opt=[]
for C,d in data.items():
    ks=np.array(sorted(d)); x=np.log2(ks); y=np.array([d[k] for k in ks])
    a,b,c=np.polyfit(x,y,2); xs=-b/(2*a); Ls=c-b*b/(4*a); opt.append((C,2**xs,C/fpt(2**xs),Ls))
    kk=2**np.linspace(x.min()-.3,x.max()+.3,100)
    ax.plot(C/fpt(kk),np.polyval([a,b,c],np.log2(kk)),color=col[C],lw=1.6,alpha=.85)
    ax.plot(C/fpt(ks),y,"o",color=col[C],ms=6,label=lab[C])
    ax.plot(C/fpt(2**xs),Ls,"*",color=col[C],ms=13,mec=FG,mew=.5)
    for k in ks: ax.annotate(f"k{k}",(C/fpt(k),d[k]),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8,color=MU)
    if C in k1: ax.plot(C/fpt(1),k1[C],"o",mfc="none",mec=col[C],ms=6)
o=np.array(opt); ax.plot(o[:,2],o[:,3],":",color=MU,lw=1.2)
Df=4.5e20/fpt(12)  # final run, k=12
ax.plot(Df,2.2532,"D",mfc="none",mec=MU,ms=7,mew=1.2,label="final, predicted 2.253")
ax.plot(Df,2.2405,"*",color="#7b5cc4",ms=15,mec=FG,mew=.5,label="final 4.5e20, k12: 2.2405")
ax.annotate("k12",(Df,2.2405),xytext=(0,-14),textcoords="offset points",ha="center",fontsize=8,color=MU)
ax.set_xscale("log"); ax.set_xlim(3e8,2.5e11); ax.set_xlabel("Training tokens D (log scale)"); ax.set_ylabel("Validation loss (after cooldown)")
ax.grid(True,which="major"); ax.grid(True,which="minor",alpha=.35); ax.legend(loc="upper right",fontsize=9.5)
fig.tight_layout(); open(f"{RES}/isoflop.svg","w").write(svg(fig))
# --- extrapolation plot
fig,axs=plt.subplots(1,2,figsize=(7.6,3.5)); Cx=np.logspace(np.log10(1.5e18),np.log10(6e20),50)
for ax,i,name,scale in ((axs[0],1,"Optimal top-k  k*",1),(axs[1],2,"Optimal tokens  D* (B)",1e9)):
    preds=[]
    for sel,ls,lb in ((slice(None),"-","fit C0–C4"),(slice(1,None),"--","fit C1–C4")):
        p=np.polyfit(np.log(o[sel,0]),np.log(o[sel,i]),1); preds.append(np.exp(np.polyval(p,np.log(Cx))))
        ax.plot(Cx,preds[-1]/scale,ls,color="#3f88c5",lw=1.4,label=f"{lb}: ∝C^{p[0]:.2f}")
    ax.fill_between(Cx,np.minimum(*preds)/scale,np.maximum(*preds)/scale,color="#3f88c5",alpha=.12,lw=0)
    for C,*r in o: ax.plot(C,r[i-1]/scale,"o",color=col[C],ms=7,zorder=5)
    for C,t in ((4.5e20,"final run"),):
        ax.axvline(C,color=MU,lw=.8,ls=":"); ax.annotate(t,(C,1),xycoords=("data","axes fraction"),xytext=(-3,-22),textcoords="offset points",ha="right",va="top",fontsize=8.5,color=MU,rotation=90)
    ax.plot(4.5e20,(12 if i==1 else 134e9)/scale,"*",color="#2a9d6f",ms=14,mec=FG,mew=.5,zorder=6,label="final run (k12, 134B)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Compute C (FLOPs)"); ax.set_title(name,fontsize=10.5,loc="left")
    ax.grid(True,which="major"); ax.legend(loc="lower right",fontsize=8.5)
axs[0].set_yticks([2,4,8,16]); axs[0].set_yticklabels(["2","4","8","16"]); axs[0].set_ylim(2.5,20); axs[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axs[1].set_yticks([1,10,100]); axs[1].set_yticklabels(["1","10","100"]); axs[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.tight_layout(); open(f"{RES}/extrap.svg","w").write(svg(fig))
print(o)
