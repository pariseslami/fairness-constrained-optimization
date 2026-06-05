# ============================================================
# Method : Log-Barrier + Augmented Lagrangian (LBALM)
# Dataset: COMPAS Recidivism
# Sensitive attributes: race, sex
# Objective: R + lambda*g + (rho/2)*g^2 - (1/t)*log(-g)
# ============================================================

import os, urllib.request
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED=11; DELTA=0.05; ANCHOR_SIZE=99999
BASELINE_EPOCHS=20; FAIRNESS_EPOCHS=50
BATCH_SIZE=256; VAL_BATCH=512; LAMBDA_MAX=1000.0; BARRIER_EPS=1e-6
SENSITIVE_COLS=["race","sex"]; SENSITIVE_COL_MAP={"race":"race_binary","sex":"sex_binary"}
DATA_DIR="./data_compas"; RESULTS_DIR="./results_LBALM_compas"; PLOTS_DIR="./plots_LBALM_compas"
os.makedirs(DATA_DIR,exist_ok=True); os.makedirs(RESULTS_DIR,exist_ok=True); os.makedirs(PLOTS_DIR,exist_ok=True)

np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | Method: LBALM | Dataset: COMPAS")

COMPAS_URL="https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
compas_path=os.path.join(DATA_DIR,"compas-scores-two-years.csv")
if not os.path.exists(compas_path):
    try: urllib.request.urlretrieve(COMPAS_URL,compas_path)
    except: pass

df_raw=pd.read_csv(compas_path)
df=df_raw[(df_raw["days_b_screening_arrest"]<=30)&(df_raw["days_b_screening_arrest"]>=-30)&
          (df_raw["is_recid"]!=-1)&(df_raw["c_charge_degree"]!="O")&
          (df_raw["score_text"]!="N/A")].copy().reset_index(drop=True)
df["label"]=df["is_recid"].astype(int)
df["race_binary"]=(df["race"]=="African-American").astype(int)
df["sex_binary"]=(df["sex"]=="Male").astype(int)
feat_raw=["age","juv_fel_count","juv_misd_count","juv_other_count","priors_count","c_charge_degree","age_cat"]
df_feat=df[feat_raw+["label","race_binary","sex_binary"]].dropna().reset_index(drop=True)
df_feat=pd.get_dummies(df_feat,columns=["c_charge_degree","age_cat"],drop_first=True)
FEATURE_COLS=[c for c in df_feat.columns if c not in ["label","race_binary","sex_binary"]]
print(f"COMPAS shape: {df_feat.shape}")

class MLP(nn.Module):
    def __init__(self,in_dim,hidden=(128,64),dropout=0.2):
        super().__init__()
        layers,d=[],in_dim
        for h in hidden: layers+=[nn.Linear(d,h),nn.ReLU(),nn.Dropout(dropout)]; d=h
        layers+=[nn.Linear(d,1)]; self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(1)

criterion=nn.BCEWithLogitsLoss()
def make_optimizer(model,lr=5e-4,wd=1e-4): return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)

def p_rule_meanprob_torch(probs,groups,eps=1e-8):
    g0,g1=(groups==0),(groups==1)
    if not g0.any() or not g1.any(): return probs.new_tensor(0.5)
    p0=probs[g0].mean(); p1=probs[g1].mean()
    return torch.minimum(p0,p1)/torch.maximum(p0,p1).clamp_min(eps)

@torch.no_grad()
def p_rule_selection_torch(probs,groups,threshold=0.5,eps=1e-8):
    preds=(probs>=threshold).float(); g0,g1=(groups==0),(groups==1)
    if not g0.any() or not g1.any(): return probs.new_tensor(0.5)
    sr0=preds[g0].mean(); sr1=preds[g1].mean()
    return torch.minimum(sr0,sr1)/torch.maximum(sr0,sr1).clamp_min(eps)

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
    try: auc=roc_auc_score(y_true,y_prob)
    except: auc=float("nan")
    return {"acc":float(acc),"auc":float(auc),"loss":float(np.sum(losses)/len(y_true)),
            "p_rule_mp":float(np.mean(rs_mp)),"p_rule_sel":float(np.mean(rs_sel))}

def build_anchor(X_t,g_t,n=ANCHOR_SIZE,seed=SEED):
    rng=np.random.RandomState(seed); idx=rng.choice(len(X_t),size=min(n,len(X_t)),replace=False)
    return X_t[idx].to(DEVICE),g_t[idx].to(DEVICE)

def selection_score(metrics,B_val,delta):
    c=metrics["loss"]-(B_val+delta)
    if c<=0: return 2.0*metrics["p_rule_sel"]+0.5*(-c)+0.25*(metrics["acc"]+(metrics["auc"] if np.isfinite(metrics["auc"]) else 0.0))
    else: return -10.0*c+1.0*metrics["p_rule_sel"]+0.25*metrics["acc"]

def make_row(method,sensitive_col,params,te_m,va_m,B_val,dataset):
    return {"method":method,"dataset":dataset,"sensitive_col":sensitive_col,"params":str(params),
            "test_acc":te_m["acc"],"test_auc":te_m["auc"],"test_loss":te_m["loss"],
            "test_p_rule_sel":te_m["p_rule_sel"],"test_p_rule_mp":te_m["p_rule_mp"],
            "val_acc":va_m["acc"],"val_loss":va_m["loss"],"val_p_rule_sel":va_m["p_rule_sel"],
            "val_constraint":va_m["loss"]-(B_val+DELTA),"score":selection_score(va_m,B_val,DELTA),
            "B_val":B_val,"delta":DELTA}

LBALM_GRID=[{"rho":rho,"lambda_init":li,"lr":lr,"t":t}
            for rho in [0.1,0.5,1.0]
            for li  in [0.0,0.1]
            for lr  in [1e-4,5e-4,1e-3]
            for t   in [5.0,10.0,20.0]]
print(f"LBALM grid: {len(LBALM_GRID)} configs x {FAIRNESS_EPOCHS} epochs")

all_sensitive_results={}

for SENSITIVE_COL in SENSITIVE_COLS:
    BINARY_COL=SENSITIVE_COL_MAP[SENSITIVE_COL]
    print(f"\n{'='*70}\nSENSITIVE: {SENSITIVE_COL} -> {BINARY_COL}\n{'='*70}")

    X_all=df_feat[FEATURE_COLS].values.astype(np.float32)
    y_all=df_feat["label"].values.astype(np.float32)
    g_all=df_feat[BINARY_COL].values.astype(np.float32)

    X_train_f,X_te,y_train_f,y_te,g_train_f,g_te=train_test_split(X_all,y_all,g_all,test_size=0.2,random_state=SEED,stratify=y_all)
    X_tr,X_val,y_tr,y_val,g_tr,g_val=train_test_split(X_train_f,y_train_f,g_train_f,test_size=0.2,random_state=SEED,stratify=y_train_f)
    scaler=StandardScaler(with_mean=True); X_tr=scaler.fit_transform(X_tr); X_val=scaler.transform(X_val); X_te=scaler.transform(X_te)

    def tt(a): return torch.tensor(a,dtype=torch.float32)
    X_tr_t=tt(X_tr); X_val_t=tt(X_val); X_te_t=tt(X_te)
    y_tr_t=tt(y_tr); y_val_t=tt(y_val); y_te_t=tt(y_te)
    g_tr_t=tt(g_tr); g_val_t=tt(g_val); g_te_t=tt(g_te)
    input_dim=X_tr_t.shape[1]
    print(f"Input dim: {input_dim} | Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

    train_loader=DataLoader(TensorDataset(X_tr_t,y_tr_t,g_tr_t),batch_size=BATCH_SIZE,shuffle=True)
    val_loader=DataLoader(TensorDataset(X_val_t,y_val_t,g_val_t),batch_size=VAL_BATCH,shuffle=False)
    test_loader=DataLoader(TensorDataset(X_te_t,y_te_t,g_te_t),batch_size=VAL_BATCH,shuffle=False)

    print("\n--- Training Baseline ---")
    _model=MLP(input_dim).to(DEVICE); _opt=make_optimizer(_model,lr=1e-3)
    _best_state,_best_val_loss=None,float("inf")
    for ep in range(1,BASELINE_EPOCHS+1):
        _model.train()
        for xb,yb,_ in train_loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); _opt.zero_grad()
            criterion(_model(xb),yb).backward(); nn.utils.clip_grad_norm_(_model.parameters(),5.0); _opt.step()
        vm=eval_metrics(_model,val_loader)
        if vm["loss"]<_best_val_loss:
            _best_val_loss=vm["loss"]; _best_state={k:v.detach().cpu().clone() for k,v in _model.state_dict().items()}
        print(f"  [Baseline ep={ep:02d}] val_loss={vm['loss']:.4f} val_acc={vm['acc']:.4f} p%-rule(sel)={vm['p_rule_sel']:.3f}")
    _model.load_state_dict(_best_state)
    B_val=eval_metrics(_model,val_loader)["loss"]; baseline_te=eval_metrics(_model,test_loader); baseline_va=eval_metrics(_model,val_loader)
    print(f"\n  B_val={B_val:.6f} | Test: acc={baseline_te['acc']:.4f} p%-rule(sel)={baseline_te['p_rule_sel']:.3f}")
    pd.DataFrame([make_row("Baseline",SENSITIVE_COL,{},baseline_te,baseline_va,B_val,"compas")]).to_csv(os.path.join(RESULTS_DIR,f"baseline_{SENSITIVE_COL}.csv"),index=False)
    X_anchor,g_anchor=build_anchor(X_tr_t,g_tr_t); print(f"  Anchor: {X_anchor.shape[0]} samples")

    print(f"\n--- LBALM Grid Search ({len(LBALM_GRID)} configs) ---")
    lbalm_rows=[]; best_constraint_curve=None; best_score_seen=-1e18

    for i,params in enumerate(LBALM_GRID):
        print(f"\n  [LBALM {i+1}/{len(LBALM_GRID)}] {params}")
        try:
            model=MLP(input_dim).to(DEVICE); model.load_state_dict(_best_state)
            opt=make_optimizer(model,lr=params["lr"])
            lam=torch.tensor(params["lambda_init"],device=DEVICE,dtype=torch.float32)
            rho=params["rho"]; t=params["t"]; best_state,best_score=None,-1e18; constraint_history=[]

            for ep in range(1,FAIRNESS_EPOCHS+1):
                model.train()
                for xb,yb,_ in train_loader:
                    xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
                    logits_a=model(X_anchor); r=p_rule_meanprob_torch(torch.sigmoid(logits_a),g_anchor); R=1.0-r
                    loss_pred=criterion(model(xb),yb); g_batch=loss_pred-(B_val+DELTA)
                    slack=-g_batch; barrier=-(1.0/t)*torch.log(slack.clamp_min(BARRIER_EPS))
                    obj=R+lam*g_batch+0.5*rho*(g_batch**2)+barrier
                    obj.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
                vm=eval_metrics(model,val_loader); c_val=vm["loss"]-(B_val+DELTA); constraint_history.append(c_val)
                lam=torch.clamp(lam+rho*torch.tensor(c_val,device=DEVICE,dtype=torch.float32),min=0.0,max=LAMBDA_MAX)
                score=selection_score(vm,B_val,DELTA)
                if score>best_score: best_score=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

            model.load_state_dict(best_state); te_m=eval_metrics(model,test_loader); va_m=eval_metrics(model,val_loader)
            row=make_row("LBALM",SENSITIVE_COL,params,te_m,va_m,B_val,"compas"); lbalm_rows.append(row)
            print(f"  -> acc={te_m['acc']:.4f} p%-rule(sel)={te_m['p_rule_sel']:.3f} val_c={va_m['loss']-(B_val+DELTA):+.4f} {'OK' if va_m['loss']-(B_val+DELTA)<=0 else 'VIOL'}")
            if row["score"]>best_score_seen: best_score_seen=row["score"]; best_constraint_curve=constraint_history
        except Exception as e: print(f"  FAILED: {e}")

    csv_path=os.path.join(RESULTS_DIR,f"results_LBALM_{SENSITIVE_COL}.csv")
    pd.DataFrame(lbalm_rows).to_csv(csv_path,index=False); print(f"\nSaved {len(lbalm_rows)} rows -> {csv_path}")
    all_sensitive_results[SENSITIVE_COL]={"lbalm_rows":lbalm_rows,"baseline_te":baseline_te,"B_val":B_val,"best_constraint_curve":best_constraint_curve}

for SENSITIVE_COL in SENSITIVE_COLS:
    if SENSITIVE_COL not in all_sensitive_results: continue
    data=all_sensitive_results[SENSITIVE_COL]; rows=data["lbalm_rows"]; bl_te=data["baseline_te"]; B_val=data["B_val"]
    if not rows: continue
    feasible=[r for r in rows if r["val_constraint"]<=0]; pool=feasible if feasible else rows; best=max(pool,key=lambda r:float(r["score"]))
    print(f"\n--- {SENSITIVE_COL} ---")
    print(f"  Baseline: acc={bl_te['acc']:.4f} p%-rule(sel)={bl_te['p_rule_sel']:.3f}")
    print(f"  LBALM Best: acc={best['test_acc']:.4f} p%-rule(sel)={best['test_p_rule_sel']:.3f} val_c={best['val_constraint']:+.4f}")
    print(f"  Feasible: {len(feasible)}/{len(rows)}")
    df_res=pd.DataFrame(rows); fmask=df_res["val_constraint"]<=0
    fig,ax=plt.subplots(figsize=(8,6))
    ax.scatter(df_res.loc[fmask,"test_p_rule_sel"],df_res.loc[fmask,"test_acc"],color="#9C27B0",alpha=0.6,s=60,marker="D",label="LBALM feasible")
    ax.scatter(df_res.loc[~fmask,"test_p_rule_sel"],df_res.loc[~fmask,"test_acc"],facecolors="none",edgecolors="#9C27B0",alpha=0.5,s=60,marker="D",label="LBALM infeasible")
    ax.scatter(bl_te["p_rule_sel"],bl_te["acc"],color="black",marker="*",s=200,label="Baseline",zorder=10)
    ax.set_xlabel("p%-rule (selection rate)",fontsize=12); ax.set_ylabel("Test Accuracy",fontsize=12)
    ax.set_title(f"LBALM - COMPAS - {SENSITIVE_COL}",fontsize=12)
    ax.legend(fontsize=9); ax.grid(True,linestyle="--",alpha=0.4); plt.tight_layout()
    p=os.path.join(PLOTS_DIR,f"plot_frontier_{SENSITIVE_COL}.png"); plt.savefig(p,dpi=150); plt.close(); print(f"Saved: {p}")

print(f"\nLBALM - COMPAS - COMPLETE\nResults: {RESULTS_DIR}/ | Plots: {PLOTS_DIR}/")
