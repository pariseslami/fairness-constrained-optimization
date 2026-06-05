# ============================================================
# Method : AFRIAL
# Dataset: MEPS Panel 19
# Label  : UTILIZATION — 1 if total visits >= 10, else 0
# Sens.  : race (White=1, non-White=0), sex (Male=1, Female=0)
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED=11; DELTA=0.01; ANCHOR_SIZE=4096
BASELINE_EPOCHS=20; FAIRNESS_EPOCHS=30
BATCH_SIZE=1024; VAL_BATCH=2048; LAMBDA_MAX=1000.0
BARRIER_EPS=1e-6
T_MIN=0.01
T_MAX=1000.0
CHECKPOINT_EVERY=50
SENSITIVE_COLS=["race","sex"]
DATA_DIR="./data_meps19"
RESULTS_DIR="./results_AFRIAL_meps19"
PLOTS_DIR="./plots_AFRIAL_meps19"
os.makedirs(DATA_DIR,exist_ok=True)
os.makedirs(RESULTS_DIR,exist_ok=True)
os.makedirs(PLOTS_DIR,exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | Method: AFRIAL | Dataset: MEPS Panel 19")

# ── Data ──────────────────────────────────────────────────────────────────────
try:
    from aif360.datasets import MEPSDataset19
except ImportError:
    import subprocess; subprocess.run(["pip","install","-q","aif360"],check=True)
    from aif360.datasets import MEPSDataset19
print("Loading MEPS Panel 19...")
meps=MEPSDataset19()
feature_names=meps.feature_names
X_all=meps.features.astype(np.float32)
y_all=meps.labels.ravel().astype(np.float32)
race_idx=meps.protected_attribute_names.index("RACE")
race_all=meps.protected_attributes[:,race_idx].astype(np.float32)
sex_cols=[i for i,n in enumerate(feature_names) if n in ("SEX_1.0","SEX_1","SEX=1")]
if sex_cols:
    sex_all=X_all[:,sex_cols[0]].astype(np.float32)
elif "SEX" in meps.protected_attribute_names:
    si=meps.protected_attribute_names.index("SEX")
    sex_all=(meps.protected_attributes[:,si]==1.0).astype(np.float32)
else:
    print("WARNING: SEX not found"); sex_all=np.zeros(len(y_all),dtype=np.float32)
print(f"Shape:{X_all.shape} | Label:{y_all.mean()*100:.1f}% | Race(W):{race_all.mean()*100:.1f}% | Sex(M):{sex_all.mean()*100:.1f}%")
df=pd.DataFrame(X_all,columns=feature_names)
df["race"]=race_all; df["sex"]=sex_all; df["label"]=y_all
feat_cols=[c for c in feature_names
           if c not in ("RACE","race","SEX","sex","RACE_1.0","RACE_0.0","SEX_1.0","SEX_2.0")]
df_train_full,df_test_full=train_test_split(df,test_size=0.2,random_state=SEED,stratify=df["label"])
df_train_full=df_train_full.reset_index(drop=True)
df_test_full=df_test_full.reset_index(drop=True)
print(f"Train:{len(df_train_full)} | Test:{len(df_test_full)} | Features:{len(feat_cols)}")

# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self,in_dim,hidden=(256,128),dropout=0.2):
        super().__init__()
        layers,d=[],in_dim
        for h in hidden:
            layers+=[nn.Linear(d,h),nn.ReLU(),nn.Dropout(dropout)]; d=h
        layers+=[nn.Linear(d,1)]; self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(1)
criterion=nn.BCEWithLogitsLoss()
def make_optimizer(model,lr=5e-4,wd=1e-4):
    return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)

# ── Metrics ───────────────────────────────────────────────────────────────────
def p_rule_meanprob_torch(probs,groups,eps=1e-8):
    g0,g1=(groups==0),(groups==1)
    if not g0.any() or not g1.any(): return probs.new_tensor(0.5)
    return torch.minimum(probs[g0].mean(),probs[g1].mean())/torch.maximum(probs[g0].mean(),probs[g1].mean()).clamp_min(eps)
@torch.no_grad()
def p_rule_selection_torch(probs,groups,threshold=0.5,eps=1e-8):
    preds=(probs>=threshold).float(); g0,g1=(groups==0),(groups==1)
    if not g0.any() or not g1.any(): return probs.new_tensor(0.5)
    return torch.minimum(preds[g0].mean(),preds[g1].mean())/torch.maximum(preds[g0].mean(),preds[g1].mean()).clamp_min(eps)
@torch.no_grad()
def eval_metrics(model,loader):
    model.eval(); ys,ps,losses,rs_mp,rs_sel=[],[],[],[],[]
    for xb,yb,gb in loader:
        xb,yb,gb=xb.to(DEVICE),yb.to(DEVICE),gb.to(DEVICE)
        logits=model(xb); probs=torch.sigmoid(logits)
        losses.append(criterion(logits,yb).item()*xb.size(0))
        ys.append(yb.cpu().numpy()); ps.append(probs.cpu().numpy())
        rs_mp.append(p_rule_meanprob_torch(probs,gb).item())
        rs_sel.append(p_rule_selection_torch(probs,gb).item())
    y_true=np.concatenate(ys).ravel(); y_prob=np.concatenate(ps).ravel()
    acc=accuracy_score(y_true,(y_prob>=0.5).astype(int))
    try:    auc=roc_auc_score(y_true,y_prob)
    except: auc=float("nan")
    return {"acc":float(acc),"auc":float(auc),"loss":float(np.sum(losses)/len(y_true)),
            "p_rule_mp":float(np.mean(rs_mp)),"p_rule_sel":float(np.mean(rs_sel))}
def build_anchor(X_t,g_t,n=ANCHOR_SIZE,seed=SEED):
    rng=np.random.RandomState(seed); idx=rng.choice(len(X_t),size=min(n,len(X_t)),replace=False)
    return X_t[idx].to(DEVICE),g_t[idx].to(DEVICE)

# ── Score ─────────────────────────────────────────────────────────────────────
def selection_score(metrics,B_val,delta):
    c=metrics["loss"]-(B_val+delta)
    if c<=0: return 2.0*metrics["p_rule_sel"]+0.5*(-c)+0.25*(metrics["acc"]+(metrics["auc"] if np.isfinite(metrics["auc"]) else 0.0))
    return -10.0*c+1.0*metrics["p_rule_sel"]+0.25*metrics["acc"]
def make_row(method,sensitive_col,params,te_m,va_m,B_val,t_history=None):
    row={"method":method,"dataset":"meps19","sensitive_col":sensitive_col,"params":str(params),
          "test_acc":te_m["acc"],"test_auc":te_m["auc"],"test_loss":te_m["loss"],
          "test_p_rule_sel":te_m["p_rule_sel"],"test_p_rule_mp":te_m["p_rule_mp"],
          "val_acc":va_m["acc"],"val_loss":va_m["loss"],"val_p_rule_sel":va_m["p_rule_sel"],
          "val_constraint":va_m["loss"]-(B_val+DELTA),"score":selection_score(va_m,B_val,DELTA),
          "B_val":B_val,"delta":DELTA}
    if t_history:
        row["t_final"]=t_history[-1]; row["t_min_reached"]=min(t_history); row["t_max_reached"]=max(t_history)
    return row

# ── Grid ──────────────────────────────────────────────────────────────────────
AFRIAL_GRID=[{"rho":rho,"lambda_init":li,"t0":t0,"mu_up":mu,"mu_down":mu,"eps_f":eps,"lr":lr}
    for rho in [0.2,0.5,1.0] for li in [0.0,0.1] for t0 in [0.1,1.0,5.0]
    for mu in [1.5,2.0] for eps in [0.003,0.01] for lr in [1e-4,5e-4]]
print(f"\nAFRIAL grid: {len(AFRIAL_GRID)} configs x {FAIRNESS_EPOCHS} epochs")
print(f"\nAFRIAL grid: {len(AFRIAL_GRID)} configs x {FAIRNESS_EPOCHS} epochs\nCheckpointing every {CHECKPOINT_EVERY} configs")

# ── Main loop ─────────────────────────────────────────────────────────────────
all_sensitive_results={}
for SENSITIVE_COL in SENSITIVE_COLS:
    print(f"\n{'='*70}\nSENSITIVE ATTRIBUTE: {SENSITIVE_COL}\n{'='*70}")
    fc=feat_cols+[SENSITIVE_COL]
    X_tr_df,X_val_df,y_tr_np,y_val_np,g_tr_np,g_val_np=train_test_split(
        df_train_full[fc],df_train_full["label"].values.astype(np.float32),
        df_train_full[SENSITIVE_COL].values.astype(np.float32),
        test_size=0.2,random_state=SEED,stratify=df_train_full["label"].values)
    scaler=StandardScaler()
    X_tr=scaler.fit_transform(X_tr_df.values); X_val=scaler.transform(X_val_df.values); X_te=scaler.transform(df_test_full[fc].values)
    def tt(a): return torch.tensor(a,dtype=torch.float32)
    X_tr_t=tt(X_tr); X_val_t=tt(X_val); X_te_t=tt(X_te)
    y_tr_t=torch.tensor(y_tr_np,dtype=torch.float32)
    y_val_t=torch.tensor(y_val_np,dtype=torch.float32)
    y_te_t=torch.tensor(df_test_full["label"].values.astype(np.float32),dtype=torch.float32)
    g_tr_t=torch.tensor(g_tr_np,dtype=torch.float32)
    g_val_t=torch.tensor(g_val_np,dtype=torch.float32)
    g_te_t=torch.tensor(df_test_full[SENSITIVE_COL].values.astype(np.float32),dtype=torch.float32)
    input_dim=X_tr_t.shape[1]
    print(f"Input dim:{input_dim} | Train:{len(X_tr_t)} Val:{len(X_val_t)} Test:{len(X_te_t)}")
    print(f"Group-1%: train={g_tr_np.mean()*100:.1f}% val={g_val_np.mean()*100:.1f}% test={g_te_t.mean()*100:.1f}%")
    train_loader=DataLoader(TensorDataset(X_tr_t,y_tr_t,g_tr_t),batch_size=BATCH_SIZE,shuffle=True)
    val_loader=DataLoader(TensorDataset(X_val_t,y_val_t,g_val_t),batch_size=VAL_BATCH,shuffle=False)
    test_loader=DataLoader(TensorDataset(X_te_t,y_te_t,g_te_t),batch_size=VAL_BATCH,shuffle=False)

    print("\n--- Training Baseline ---")
    _model=MLP(input_dim).to(DEVICE); _opt=make_optimizer(_model,lr=1e-3,wd=1e-4)
    _best_state,_best_val_loss=None,float("inf")
    for ep in range(1,BASELINE_EPOCHS+1):
        _model.train()
        for xb,yb,_ in train_loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); _opt.zero_grad()
            criterion(_model(xb),yb).backward(); nn.utils.clip_grad_norm_(_model.parameters(),5.0); _opt.step()
        vm=eval_metrics(_model,val_loader)
        if vm["loss"]<_best_val_loss:
            _best_val_loss=vm["loss"]
            _best_state={k:v.detach().cpu().clone() for k,v in _model.state_dict().items()}
        print(f"  [Baseline ep={ep:02d}] val_loss={vm['loss']:.4f} val_acc={vm['acc']:.4f} p%-rule(sel)={vm['p_rule_sel']:.4f}")
    _model.load_state_dict(_best_state)
    B_val=eval_metrics(_model,val_loader)["loss"]
    baseline_te=eval_metrics(_model,test_loader)
    baseline_va=eval_metrics(_model,val_loader)
    print(f"\n  B_val={B_val:.6f}")
    print(f"  Baseline Test → acc={baseline_te['acc']:.4f} auc={baseline_te['auc']:.4f} p%-rule(sel)={baseline_te['p_rule_sel']:.4f} p%-rule(mp)={baseline_te['p_rule_mp']:.4f}")
    pd.DataFrame([{"method":"Baseline","dataset":"meps19","sensitive_col":SENSITIVE_COL,
                   "params":"{}","test_acc":baseline_te["acc"],"test_auc":baseline_te["auc"],
                   "test_loss":baseline_te["loss"],"test_p_rule_sel":baseline_te["p_rule_sel"],
                   "test_p_rule_mp":baseline_te["p_rule_mp"],"val_acc":baseline_va["acc"],
                   "val_loss":baseline_va["loss"],"val_p_rule_sel":baseline_va["p_rule_sel"],
                   "val_constraint":0.0,"score":0.0,"B_val":B_val,"delta":DELTA}
    ]).to_csv(os.path.join(RESULTS_DIR,f"baseline_{SENSITIVE_COL}.csv"),index=False)
    X_anchor,g_anchor=build_anchor(X_tr_t,g_tr_t)
    print(f"  Anchor: {X_anchor.shape[0]} samples")

    print(f"\n--- AFRIAL Grid Search ({len(AFRIAL_GRID)} configs) ---")
    checkpoint_path=os.path.join(RESULTS_DIR,f"results_AFRIAL_{SENSITIVE_COL}_partial.csv")
    if os.path.exists(checkpoint_path):
        existing=pd.read_csv(checkpoint_path); start_from=len(existing); afrial_rows=existing.to_dict("records")
        print(f"  Resuming from config {start_from+1} ({start_from} done)")
    else:
        start_from=0; afrial_rows=[]
    best_constraint_curve=None; best_lambda_curve=None; best_t_curve=None; best_score_seen=-1e18

    for i,params in enumerate(AFRIAL_GRID):
        if i<start_from: continue
        print(f"\n  [AFRIAL {i+1}/{len(AFRIAL_GRID)}] {params}")
        try:
            model=MLP(input_dim).to(DEVICE); model.load_state_dict(_best_state)
            opt=make_optimizer(model,lr=params["lr"])
            lam=torch.tensor(params["lambda_init"],device=DEVICE,dtype=torch.float32)
            rho=params["rho"]; t=params["t0"]; mu_up=params["mu_up"]; mu_dn=params["mu_down"]; eps_f=params["eps_f"]
            best_state,best_score=None,-1e18
            constraint_history=[]; lambda_history=[]; t_history=[]

            for ep in range(1,FAIRNESS_EPOCHS+1):
                model.train()
                for xb,yb,_ in train_loader:
                    xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
                    r=p_rule_meanprob_torch(torch.sigmoid(model(X_anchor)),g_anchor); R=1.0-r
                    loss_mb=criterion(model(xb),yb); g_b=loss_mb-(B_val+DELTA); g_pos=torch.relu(g_b)
                    barrier=-(1.0/t)*torch.log((-g_b).clamp_min(BARRIER_EPS))
                    (R+lam*g_b+0.5*rho*g_pos**2+barrier).backward()
                    nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
                vm=eval_metrics(model,val_loader); c_val=vm["loss"]-(B_val+DELTA)
                constraint_history.append(c_val)
                if c_val<-eps_f: t=min(t*mu_up,T_MAX)
                elif c_val>0:    t=max(t/mu_dn,T_MIN)
                t_history.append(t)
                lam=torch.clamp(lam+rho*torch.tensor(c_val,device=DEVICE,dtype=torch.float32),min=0.0,max=LAMBDA_MAX)
                lambda_history.append(float(lam.item()))
                score=selection_score(vm,B_val,DELTA)
                if score>best_score:
                    best_score=score
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

            model.load_state_dict(best_state)
            te_m=eval_metrics(model,test_loader); va_m=eval_metrics(model,val_loader)
            row=make_row("AFRIAL",SENSITIVE_COL,params,te_m,va_m,B_val,t_history)
            afrial_rows.append(row)
            feasible="✓" if va_m["loss"]-(B_val+DELTA)<=0 else "✗"
            print(f"  → acc={te_m['acc']:.4f} auc={te_m['auc']:.4f} p%-rule(sel)={te_m['p_rule_sel']:.4f} p%-rule(mp)={te_m['p_rule_mp']:.4f} val_c={va_m['loss']-(B_val+DELTA):+.4f} t={t:.2f} λ={float(lam.item()):.3f} {feasible}")
            if row["score"]>best_score_seen:
                best_score_seen=row["score"]
                best_constraint_curve=constraint_history; best_lambda_curve=lambda_history; best_t_curve=t_history
        except Exception as e:
            print(f"  FAILED: {e}")
        if (i+1)%CHECKPOINT_EVERY==0 or (i+1)==len(AFRIAL_GRID):
            pd.DataFrame(afrial_rows).to_csv(checkpoint_path,index=False)
            print(f"  CHECKPOINT saved at config {i+1}")

    csv_path=os.path.join(RESULTS_DIR,f"results_AFRIAL_{SENSITIVE_COL}.csv")
    pd.DataFrame(afrial_rows).to_csv(csv_path,index=False)
    print(f"\nSaved {len(afrial_rows)} rows → {csv_path}")
    all_sensitive_results[SENSITIVE_COL]={"afrial_rows":afrial_rows,"baseline_te":baseline_te,"B_val":B_val,"best_constraint_curve":best_constraint_curve,"best_lambda_curve":best_lambda_curve,"best_t_curve":best_t_curve}

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
for SENSITIVE_COL in SENSITIVE_COLS:
    if SENSITIVE_COL not in all_sensitive_results: continue
    data=all_sensitive_results[SENSITIVE_COL]; rows=data["afrial_rows"]; bl_te=data["baseline_te"]
    if not rows: continue
    feasible=[r for r in rows if r["val_constraint"]<=0]; pool=feasible if feasible else rows
    best=max(pool,key=lambda r:float(r["score"]))
    print(f"\n--- {SENSITIVE_COL} ---")
    print(f"  Baseline  : acc={bl_te['acc']:.4f} auc={bl_te['auc']:.4f} p%-rule(sel)={bl_te['p_rule_sel']:.4f} p%-rule(mp)={bl_te['p_rule_mp']:.4f}")
    print(f"  AFRIAL Best: acc={best['test_acc']:.4f} auc={best['test_auc']:.4f} p%-rule(sel)={best['test_p_rule_sel']:.4f} p%-rule(mp)={best['test_p_rule_mp']:.4f} val_c={best['val_constraint']:+.4f}")
    print(f"  Feasible  : {len(feasible)}/{len(rows)} ({100*len(feasible)/max(len(rows),1):.0f}%)")

# ── Plots ─────────────────────────────────────────────────────────────────────
for SENSITIVE_COL in SENSITIVE_COLS:
    if SENSITIVE_COL not in all_sensitive_results: continue
    data=all_sensitive_results[SENSITIVE_COL]; rows=data["afrial_rows"]; bl_te=data["baseline_te"]
    if not rows: continue
    df_res=pd.DataFrame(rows); c_curve=data["best_constraint_curve"]
    fig,ax=plt.subplots(figsize=(8,6)); fmask=df_res["val_constraint"]<=0
    ax.scatter(df_res.loc[fmask,"test_p_rule_sel"],df_res.loc[fmask,"test_acc"],
               color="#F44336",alpha=0.6,s=60,marker="P",label="AFRIAL feasible")
    ax.scatter(df_res.loc[~fmask,"test_p_rule_sel"],df_res.loc[~fmask,"test_acc"],
               facecolors="none",edgecolors="#F44336",alpha=0.5,s=60,marker="P",label="AFRIAL infeasible")
    ax.scatter(bl_te["p_rule_sel"],bl_te["acc"],color="black",marker="*",s=200,label="Baseline",zorder=10)
    ax.axvline(bl_te["p_rule_sel"],color="gray",linestyle=":",alpha=0.5)
    ax.set_xlabel("p%-rule (selection rate)",fontsize=12); ax.set_ylabel("Test Accuracy",fontsize=12)
    ax.set_title(f"AFRIAL — MEPS Panel 19 — {SENSITIVE_COL}\nAll runs",fontsize=12)
    ax.legend(fontsize=9); ax.grid(True,linestyle="--",alpha=0.4); plt.tight_layout()
    p=os.path.join(PLOTS_DIR,f"plot_frontier_all_{SENSITIVE_COL}.png")
    plt.savefig(p,dpi=150); plt.close(); print(f"Saved: {p}")
    if c_curve:
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot(range(1,len(c_curve)+1),c_curve,color="#F44336",linewidth=2)
        ax.axhline(0,color="black",linestyle="--",linewidth=1.5,label="Feasibility boundary")
        ax.fill_between(range(1,len(c_curve)+1),0,max(max(c_curve),0.01),alpha=0.08,color="red")
        ax.set_xlabel("Epoch",fontsize=12); ax.set_ylabel("g = val_loss − (B+δ)",fontsize=11)
        ax.set_title(f"AFRIAL — MEPS Panel 19 — {SENSITIVE_COL}\nConstraint curve",fontsize=12)
        ax.legend(fontsize=9); ax.grid(True,linestyle="--",alpha=0.4); plt.tight_layout()
        p=os.path.join(PLOTS_DIR,f"plot_constraint_curve_{SENSITIVE_COL}.png")
        plt.savefig(p,dpi=150); plt.close(); print(f"Saved: {p}")
    lam_curve=data.get("best_lambda_curve")
    if lam_curve and c_curve:
        fig,ax1=plt.subplots(figsize=(8,4))
        ax1.plot(range(1,len(lam_curve)+1),lam_curve,color="#F44336",linewidth=2,label="λ")
        ax1.set_xlabel("Epoch",fontsize=12); ax1.set_ylabel("λ",color="#F44336",fontsize=12)
        ax1.tick_params(axis="y",labelcolor="#F44336")
        ax2=ax1.twinx()
        ax2.plot(range(1,len(c_curve)+1),c_curve,color="steelblue",linewidth=2,linestyle="--",label="g")
        ax2.axhline(0,color="black",linewidth=1,linestyle=":")
        ax2.set_ylabel("g",color="steelblue",fontsize=12); ax2.tick_params(axis="y",labelcolor="steelblue")
        l1,lb1=ax1.get_legend_handles_labels(); l2,lb2=ax2.get_legend_handles_labels()
        ax1.legend(l1+l2,lb1+lb2,fontsize=9,loc="upper left")
        ax1.set_title(f"AFRIAL — MEPS 19 — {SENSITIVE_COL}\nλ evolution",fontsize=12)
        ax1.grid(True,linestyle="--",alpha=0.4); plt.tight_layout()
        p=os.path.join(PLOTS_DIR,f"plot_lambda_{SENSITIVE_COL}.png")
        plt.savefig(p,dpi=150); plt.close(); print(f"Saved: {p}")
    t_curve=data.get("best_t_curve")
    if t_curve and c_curve:
        fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,7),sharex=True)
        ax1.plot(range(1,len(t_curve)+1),t_curve,color="#F44336",linewidth=2)
        ax1.set_ylabel("t (log)",fontsize=11); ax1.set_yscale("log")
        ax1.set_title(f"AFRIAL — MEPS 19 — {SENSITIVE_COL}\nAdaptive t",fontsize=12)
        ax1.grid(True,linestyle="--",alpha=0.4)
        for ep in range(1,len(t_curve)):
            if t_curve[ep]>t_curve[ep-1]: ax1.annotate("↑",(ep+1,t_curve[ep]),fontsize=7,color="green",alpha=0.6)
            elif t_curve[ep]<t_curve[ep-1]: ax1.annotate("↓",(ep+1,t_curve[ep]),fontsize=7,color="red",alpha=0.6)
        ax2.plot(range(1,len(c_curve)+1),c_curve,color="steelblue",linewidth=2)
        ax2.axhline(0,color="black",linestyle="--",linewidth=1.5,label="Feasibility boundary")
        ax2.fill_between(range(1,len(c_curve)+1),0,max(max(c_curve),0.001),alpha=0.08,color="red")
        ax2.set_xlabel("Epoch",fontsize=12); ax2.set_ylabel("g = val_loss − (B+δ)",fontsize=11)
        ax2.legend(fontsize=9); ax2.grid(True,linestyle="--",alpha=0.4); plt.tight_layout()
        p=os.path.join(PLOTS_DIR,f"plot_adaptive_t_{SENSITIVE_COL}.png")
        plt.savefig(p,dpi=150); plt.close(); print(f"Saved: {p}")
print(f"\n{'='*70}\nAFRIAL — MEPS Panel 19 — COMPLETE\nResults: {RESULTS_DIR}/\nPlots: {PLOTS_DIR}/\n{'='*70}")
