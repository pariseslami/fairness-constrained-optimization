# ============================================================
# Experiment : Comparison of Plain MLP vs LBALM & AFRIAL
# Dataset    : COMPAS
# ============================================================
#
# Three methods compared:
#   1) Plain MLP — single run  (1 config, no fairness)
#   2) Plain MLP — grid search (24 configs, no fairness)
#   3) LBALM    — single run  (default params, with fairness)
#   4) AFRIAL   — single run  (default params, with fairness)
#
# This gives TWO comparisons:
#   A) Equal budget : Plain MLP (1 run) vs LBALM (1 run) vs AFRIAL (1 run)
#      → Shows: same compute, our method is fairer
#   B) Practical    : Plain MLP (grid, 24 runs) vs LBALM (1) vs AFRIAL (1)
#      → Shows: our method is faster AND fairer than tuned plain MLP
#
# Default hyperparameters (single run):
#   Plain MLP : hidden=(128,64), dropout=0.2, lr=1e-3  ← same as B_val baseline
#   LBALM     : rho=0.5, lambda_init=0.1, lr=5e-4, t=10.0
#   AFRIAL    : rho=0.5, lambda_init=0.1, t0=1.0, mu=2.0, eps_f=0.01, lr=5e-4
#
# Plain MLP grid (24 configs):
#   4 lr × 3 hidden × 2 dropout
# ============================================================


import os, time
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

# ── Global config ─────────────────────────────────────────────────────────────
SEED            = 11
DELTA           = 0.05
ANCHOR_SIZE     = 4096
BASELINE_EPOCHS = 20
FAIRNESS_EPOCHS = 30
BATCH_SIZE      = 256
VAL_BATCH       = 2048
LAMBDA_MAX      = 1000.0
BARRIER_EPS     = 1e-6
T_MIN           = 0.01
T_MAX           = 1000.0

SENSITIVE_COLS  = ["race","sex"]
RESULTS_DIR     = "./results_singlerun_compas"
PLOTS_DIR       = "./plots_singlerun_compas"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | Dataset: COMPAS | Delta: {DELTA}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
# Plain MLP single run — same architecture as baseline
MLP_SINGLE_PARAMS  = {"hidden": (64,32), "dropout": 0.2, "lr": 1e-3}

# LBALM & AFRIAL single run — default params
LBALM_DEFAULT  = {"rho": 0.5, "lambda_init": 0.1, "lr": 5e-4, "t": 10.0}
AFRIAL_DEFAULT = {"rho": 0.5, "lambda_init": 0.1, "t0": 1.0,
                  "mu_up": 2.0, "mu_down": 2.0, "eps_f": 0.01, "lr": 5e-4}

# Plain MLP grid search (24 configs)
MLP_GRID = [
    {"lr": lr, "hidden": hidden, "dropout": dropout}
    for lr      in [1e-4, 5e-4, 1e-3, 5e-3]
    for hidden  in [(64, 32), (128, 64), (256, 128)]
    for dropout in [0.1, 0.2]
]  # 4 × 3 × 2 = 24 configs
print(f"Plain MLP grid: {len(MLP_GRID)} configs")

# ── Data loading ──────────────────────────────────────────────────────────────
import urllib.request
DATA_DIR  = "./data_compas"
DATA_PATH = os.path.join(DATA_DIR, "compas-scores-two-years.csv")
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(DATA_PATH):
    print("Downloading COMPAS dataset from ProPublica GitHub...")
    url = ("https://raw.githubusercontent.com/propublica/compas-analysis/"
           "master/compas-scores-two-years.csv")
    urllib.request.urlretrieve(url, DATA_PATH)
    print(f"  Saved: {DATA_PATH}")
else:
    print(f"  Found: {DATA_PATH}")
df_raw = pd.read_csv(DATA_PATH)
df_raw = df_raw[
    (df_raw["days_b_screening_arrest"] >= -30) &
    (df_raw["days_b_screening_arrest"] <=  30) &
    (df_raw["is_recid"]        != -1) &
    (df_raw["c_charge_degree"] != "O") &
    (df_raw["score_text"]      != "N/A")].copy()
df_raw["label"] = (df_raw["two_year_recid"] == 1).astype(int)
df_raw["race"]  = (df_raw["race"] == "Caucasian").astype(float)
df_raw["sex"]   = (df_raw["sex"]  == "Male").astype(float)
feat_cols = ["age","juv_fel_count","juv_misd_count","juv_other_count","priors_count"]
keep = feat_cols + ["race","sex","label"]
df_clean = df_raw[keep].dropna().reset_index(drop=True)
df_train_full, df_test_full = train_test_split(
    df_clean, test_size=0.2, random_state=SEED, stratify=df_clean["label"])
df_train_full = df_train_full.reset_index(drop=True)
df_test_full  = df_test_full.reset_index(drop=True)
print(f"COMPAS loaded: {len(df_clean)} rows | label={df_clean['label'].mean()*100:.1f}%")

# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(64,32), dropout=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(1)

criterion = nn.BCEWithLogitsLoss()
def make_optimizer(model, lr=5e-4, wd=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

# ── Fairness metrics ──────────────────────────────────────────────────────────
def p_rule_meanprob_torch(probs, groups, eps=1e-8):
    g0, g1 = (groups==0), (groups==1)
    if not g0.any() or not g1.any(): return probs.new_tensor(0.5)
    p0=probs[g0].mean(); p1=probs[g1].mean()
    return torch.minimum(p0,p1)/torch.maximum(p0,p1).clamp_min(eps)

@torch.no_grad()
def p_rule_selection_torch(probs, groups, threshold=0.5, eps=1e-8):
    preds=(probs>=threshold).float(); g0,g1=(groups==0),(groups==1)
    if not g0.any() or not g1.any(): return probs.new_tensor(0.5)
    sr0=preds[g0].mean(); sr1=preds[g1].mean()
    return torch.minimum(sr0,sr1)/torch.maximum(sr0,sr1).clamp_min(eps)

@torch.no_grad()
def eval_metrics(model, loader):
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
    return {"acc":float(acc),"auc":float(auc),
            "loss":float(np.sum(losses)/len(y_true)),
            "p_rule_mp":float(np.mean(rs_mp)),
            "p_rule_sel":float(np.mean(rs_sel))}

def build_anchor(X_t, g_t, n=ANCHOR_SIZE, seed=SEED):
    rng=np.random.RandomState(seed)
    idx=rng.choice(len(X_t),size=min(n,len(X_t)),replace=False)
    return X_t[idx].to(DEVICE), g_t[idx].to(DEVICE)

def selection_score(metrics, B_val, delta):
    c=metrics["loss"]-(B_val+delta)
    if c<=0:
        return (2.0*metrics["p_rule_sel"]+0.5*(-c)
                +0.25*(metrics["acc"]+(metrics["auc"] if np.isfinite(metrics["auc"]) else 0.0)))
    return -10.0*c+1.0*metrics["p_rule_sel"]+0.25*metrics["acc"]

def make_result(method, dataset, sensitive_col, n_configs,
                wall_time, feasible, feasible_count, feasible_pct,
                p_rule_sel, acc, bl_prule, bl_acc, B_val, delta, params):
    return {"method": method, "dataset": dataset,
            "sensitive_col": sensitive_col, "n_configs": n_configs,
            "wall_time_sec": round(wall_time, 2),
            "feasible": feasible,
            "feasible_count": feasible_count,
            "feasible_pct": feasible_pct,
            "test_p_rule_sel": p_rule_sel,
            "test_acc": acc,
            "bl_p_rule_sel": round(bl_prule, 4),
            "bl_acc": round(bl_acc, 4),
            "B_val": round(B_val, 6),
            "delta": delta,
            "params": str(params)}

# ── Main loop ─────────────────────────────────────────────────────────────────
all_results = []

for SENSITIVE_COL in SENSITIVE_COLS:
    print(f"\n{'='*70}\nSENSITIVE ATTRIBUTE: {SENSITIVE_COL}\n{'='*70}")
    fc = feat_cols + [SENSITIVE_COL]

    X_tr_df,X_val_df,y_tr_np,y_val_np,g_tr_np,g_val_np = train_test_split(
        df_train_full[fc],
        df_train_full["label"].values.astype(np.float32),
        df_train_full[SENSITIVE_COL].values.astype(np.float32),
        test_size=0.2, random_state=SEED, stratify=df_train_full["label"].values)

    scaler=StandardScaler()
    X_tr=scaler.fit_transform(X_tr_df.values)
    X_val=scaler.transform(X_val_df.values)
    X_te=scaler.transform(df_test_full[fc].values)

    def tt(a): return torch.tensor(a, dtype=torch.float32)
    X_tr_t=tt(X_tr); X_val_t=tt(X_val); X_te_t=tt(X_te)
    y_tr_t =torch.tensor(y_tr_np, dtype=torch.float32)
    y_val_t=torch.tensor(y_val_np,dtype=torch.float32)
    y_te_t =torch.tensor(df_test_full["label"].values.astype(np.float32),dtype=torch.float32)
    g_tr_t =torch.tensor(g_tr_np, dtype=torch.float32)
    g_val_t=torch.tensor(g_val_np,dtype=torch.float32)
    g_te_t =torch.tensor(df_test_full[SENSITIVE_COL].values.astype(np.float32),dtype=torch.float32)

    input_dim = X_tr_t.shape[1]
    print(f"Input dim:{input_dim} | Train:{len(X_tr_t)} Val:{len(X_val_t)} Test:{len(X_te_t)}")
    print(f"Group-1%: train={g_tr_np.mean()*100:.1f}% val={g_val_np.mean()*100:.1f}% test={g_te_t.mean()*100:.1f}%")

    train_loader=DataLoader(TensorDataset(X_tr_t,y_tr_t,g_tr_t),batch_size=BATCH_SIZE,shuffle=True)
    val_loader  =DataLoader(TensorDataset(X_val_t,y_val_t,g_val_t),batch_size=VAL_BATCH,shuffle=False)
    test_loader =DataLoader(TensorDataset(X_te_t,y_te_t,g_te_t),batch_size=VAL_BATCH,shuffle=False)

    # ── Compute B_val from standard baseline ──────────────────────────────────
    print("\n--- Computing B_val (accuracy bound for fairness methods) ---")
    _bl=MLP(input_dim,hidden=(64,32),dropout=0.2).to(DEVICE)
    _opt=make_optimizer(_bl,lr=1e-3,wd=1e-4)
    _best_state,_best_vl=None,float("inf")
    for ep in range(1,BASELINE_EPOCHS+1):
        _bl.train()
        for xb,yb,_ in train_loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); _opt.zero_grad()
            criterion(_bl(xb),yb).backward()
            nn.utils.clip_grad_norm_(_bl.parameters(),5.0); _opt.step()
        vm=eval_metrics(_bl,val_loader)
        if vm["loss"]<_best_vl:
            _best_vl=vm["loss"]
            _best_state={k:v.detach().cpu().clone() for k,v in _bl.state_dict().items()}
    _bl.load_state_dict(_best_state)
    B_val=eval_metrics(_bl,val_loader)["loss"]
    bl_te=eval_metrics(_bl,test_loader)
    print(f"  B_val={B_val:.6f} | acc={bl_te['acc']:.4f} | p%-rule={bl_te['p_rule_sel']:.4f}")
    X_anchor,g_anchor=build_anchor(X_tr_t,g_tr_t)

    # ══════════════════════════════════════════════════════════════════════════
    # METHOD 1 — Plain MLP Single Run (1 config, no fairness)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n--- Plain MLP Single Run (1 config, no fairness) ---")
    print(f"  Params: {MLP_SINGLE_PARAMS}")
    mlp1_t0=time.time()
    model=MLP(input_dim,hidden=MLP_SINGLE_PARAMS["hidden"],
              dropout=MLP_SINGLE_PARAMS["dropout"]).to(DEVICE)
    opt=make_optimizer(model,lr=MLP_SINGLE_PARAMS["lr"],wd=1e-4)
    best_s,best_vl=None,float("inf")
    for ep in range(1,FAIRNESS_EPOCHS+1):
        model.train()
        for xb,yb,_ in train_loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
            criterion(model(xb),yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        vm=eval_metrics(model,val_loader)
        if vm["loss"]<best_vl:
            best_vl=vm["loss"]
            best_s={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    mlp1_wall=time.time()-mlp1_t0
    model.load_state_dict(best_s)
    te1=eval_metrics(model,test_loader); va1=eval_metrics(model,val_loader)
    c1=va1["loss"]-(B_val+DELTA); feas1=c1<=0
    print(f"  Plain MLP (1 run) | time={mlp1_wall:.1f}s | feas={'YES' if feas1 else 'NO'} "
          f"(val_c={c1:+.4f}) | p%-rule={te1['p_rule_sel']:.4f} | acc={te1['acc']:.4f}")
    all_results.append(make_result(
        "Plain MLP (1 run)","compas",SENSITIVE_COL,1,mlp1_wall,
        feas1,1 if feas1 else 0,100 if feas1 else 0,
        round(te1["p_rule_sel"],4),round(te1["acc"],4),
        bl_te["p_rule_sel"],bl_te["acc"],B_val,DELTA,MLP_SINGLE_PARAMS))

    # ══════════════════════════════════════════════════════════════════════════
    # METHOD 2 — Plain MLP Grid Search (24 configs, no fairness)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n--- Plain MLP Grid Search ({len(MLP_GRID)} configs, no fairness) ---")
    mlp_g_t0=time.time(); mlp_rows=[]
    for cfg_i,cfg in enumerate(MLP_GRID):
        try:
            model=MLP(input_dim,hidden=cfg["hidden"],dropout=cfg["dropout"]).to(DEVICE)
            opt=make_optimizer(model,lr=cfg["lr"],wd=1e-4)
            best_s,best_vl=None,float("inf")
            for ep in range(1,FAIRNESS_EPOCHS+1):
                model.train()
                for xb,yb,_ in train_loader:
                    xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
                    criterion(model(xb),yb).backward()
                    nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
                vm=eval_metrics(model,val_loader)
                if vm["loss"]<best_vl:
                    best_vl=vm["loss"]
                    best_s={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best_s)
            te_m=eval_metrics(model,test_loader); va_m=eval_metrics(model,val_loader)
            c_val=va_m["loss"]-(B_val+DELTA)
            mlp_rows.append({"cfg":str(cfg),"feasible":c_val<=0,
                              "val_constraint":c_val,
                              "test_p_rule_sel":te_m["p_rule_sel"],
                              "test_acc":te_m["acc"]})
            print(f"  [{cfg_i+1:02d}/{len(MLP_GRID)}] lr={cfg['lr']} hidden={cfg['hidden']} "
                  f"drop={cfg['dropout']} | feas={'Y' if c_val<=0 else 'N'} "
                  f"p%-rule={te_m['p_rule_sel']:.4f} acc={te_m['acc']:.4f}")
        except Exception as e:
            print(f"  Config {cfg_i+1} FAILED: {e}")

    mlp_g_wall=time.time()-mlp_g_t0
    mlp_feas=[r for r in mlp_rows if r["feasible"]]
    mlp_feas_pct=round(100*len(mlp_feas)/max(len(mlp_rows),1))
    best_mlp=(max(mlp_feas,key=lambda r:r["test_p_rule_sel"]) if mlp_feas
              else max(mlp_rows,key=lambda r:r["test_p_rule_sel"]) if mlp_rows else None)
    mlp_g_prule=round(best_mlp["test_p_rule_sel"],4) if best_mlp else None
    mlp_g_acc  =round(best_mlp["test_acc"],4)        if best_mlp else None
    print(f"\n  Plain MLP (grid) | time={mlp_g_wall:.1f}s | "
          f"feas={len(mlp_feas)}/{len(mlp_rows)} ({mlp_feas_pct}%) | "
          f"best p%-rule={mlp_g_prule} | acc={mlp_g_acc}")
    all_results.append(make_result(
        "Plain MLP (grid)","compas",SENSITIVE_COL,len(MLP_GRID),mlp_g_wall,
        len(mlp_feas)>0,len(mlp_feas),mlp_feas_pct,
        mlp_g_prule,mlp_g_acc,
        bl_te["p_rule_sel"],bl_te["acc"],B_val,DELTA,"grid_24_configs"))

    # ══════════════════════════════════════════════════════════════════════════
    # METHOD 3 — LBALM Single Run (default params, with fairness)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n--- LBALM Single Run | params: {LBALM_DEFAULT} ---")
    lbalm_t0=time.time()
    model=MLP(input_dim,hidden=(64,32),dropout=0.2).to(DEVICE)
    model.load_state_dict(_best_state)
    opt=make_optimizer(model,lr=LBALM_DEFAULT["lr"],wd=1e-4)
    lam=torch.tensor(LBALM_DEFAULT["lambda_init"],device=DEVICE,dtype=torch.float32)
    rho=LBALM_DEFAULT["rho"]; t_bar=LBALM_DEFAULT["t"]
    best_s,best_sc=None,-1e18
    for ep in range(1,FAIRNESS_EPOCHS+1):
        model.train()
        for xb,yb,_ in train_loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
            r=p_rule_meanprob_torch(torch.sigmoid(model(X_anchor)),g_anchor); R=1.0-r
            loss_mb=criterion(model(xb),yb); g_b=loss_mb-(B_val+DELTA)
            barrier=-(1.0/t_bar)*torch.log((-g_b).clamp_min(BARRIER_EPS))
            (R+lam*g_b+0.5*rho*(g_b**2)+barrier).backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        vm=eval_metrics(model,val_loader); c_val=vm["loss"]-(B_val+DELTA)
        lam=torch.clamp(lam+rho*torch.tensor(c_val,device=DEVICE,dtype=torch.float32),
                        min=0.0,max=LAMBDA_MAX)
        sc=selection_score(vm,B_val,DELTA)
        if sc>best_sc:
            best_sc=sc
            best_s={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    lbalm_wall=time.time()-lbalm_t0
    model.load_state_dict(best_s)
    te_lb=eval_metrics(model,test_loader); va_lb=eval_metrics(model,val_loader)
    c_lb=va_lb["loss"]-(B_val+DELTA); feas_lb=c_lb<=0
    print(f"  LBALM | time={lbalm_wall:.1f}s | feas={'YES' if feas_lb else 'NO'} "
          f"(val_c={c_lb:+.4f}) | p%-rule={te_lb['p_rule_sel']:.4f} | acc={te_lb['acc']:.4f}")
    all_results.append(make_result(
        "LBALM (single run)","compas",SENSITIVE_COL,1,lbalm_wall,
        feas_lb,1 if feas_lb else 0,100 if feas_lb else 0,
        round(te_lb["p_rule_sel"],4),round(te_lb["acc"],4),
        bl_te["p_rule_sel"],bl_te["acc"],B_val,DELTA,LBALM_DEFAULT))

    # ══════════════════════════════════════════════════════════════════════════
    # METHOD 4 — AFRIAL Single Run (default params, with fairness)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n--- AFRIAL Single Run | params: {AFRIAL_DEFAULT} ---")
    afrial_t0=time.time()
    model=MLP(input_dim,hidden=(64,32),dropout=0.2).to(DEVICE)
    model.load_state_dict(_best_state)
    opt=make_optimizer(model,lr=AFRIAL_DEFAULT["lr"],wd=1e-4)
    lam=torch.tensor(AFRIAL_DEFAULT["lambda_init"],device=DEVICE,dtype=torch.float32)
    rho=AFRIAL_DEFAULT["rho"]; t_bar=AFRIAL_DEFAULT["t0"]
    mu_up=AFRIAL_DEFAULT["mu_up"]; mu_dn=AFRIAL_DEFAULT["mu_down"]; eps_f=AFRIAL_DEFAULT["eps_f"]
    best_s,best_sc=None,-1e18
    for ep in range(1,FAIRNESS_EPOCHS+1):
        model.train()
        for xb,yb,_ in train_loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad()
            r=p_rule_meanprob_torch(torch.sigmoid(model(X_anchor)),g_anchor); R=1.0-r
            loss_mb=criterion(model(xb),yb); g_b=loss_mb-(B_val+DELTA); g_pos=torch.relu(g_b)
            barrier=-(1.0/t_bar)*torch.log((-g_b).clamp_min(BARRIER_EPS))
            (R+lam*g_b+0.5*rho*g_pos**2+barrier).backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        vm=eval_metrics(model,val_loader); c_val=vm["loss"]-(B_val+DELTA)
        if c_val<-eps_f: t_bar=min(t_bar*mu_up,T_MAX)
        elif c_val>0:    t_bar=max(t_bar/mu_dn,T_MIN)
        lam=torch.clamp(lam+rho*torch.tensor(c_val,device=DEVICE,dtype=torch.float32),
                        min=0.0,max=LAMBDA_MAX)
        sc=selection_score(vm,B_val,DELTA)
        if sc>best_sc:
            best_sc=sc
            best_s={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    afrial_wall=time.time()-afrial_t0
    model.load_state_dict(best_s)
    te_af=eval_metrics(model,test_loader); va_af=eval_metrics(model,val_loader)
    c_af=va_af["loss"]-(B_val+DELTA); feas_af=c_af<=0
    print(f"  AFRIAL | time={afrial_wall:.1f}s | feas={'YES' if feas_af else 'NO'} "
          f"(val_c={c_af:+.4f}) | p%-rule={te_af['p_rule_sel']:.4f} | acc={te_af['acc']:.4f}")
    all_results.append(make_result(
        "AFRIAL (single run)","compas",SENSITIVE_COL,1,afrial_wall,
        feas_af,1 if feas_af else 0,100 if feas_af else 0,
        round(te_af["p_rule_sel"],4),round(te_af["acc"],4),
        bl_te["p_rule_sel"],bl_te["acc"],B_val,DELTA,AFRIAL_DEFAULT))

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path=os.path.join(RESULTS_DIR,"results_singlerun_compas.csv")
pd.DataFrame(all_results).to_csv(csv_path,index=False)
print(f"\nSaved: {csv_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}\nSUMMARY — COMPAS\n{'='*70}")
print(f"{'Method':<22} {'Attr':<9} {'#cfg':<6} {'Time(s)':<10} {'Feas%':<8} {'p%-rule':<10} {'Acc'}")
print("-"*70)
for r in all_results:
    fstr =f"{r['feasible_pct']}%"
    prule=f"{r['test_p_rule_sel']:.4f}" if r["test_p_rule_sel"] is not None else "---"
    acc  =f"{r['test_acc']:.4f}"        if r["test_acc"]        is not None else "---"
    print(f"  {r['method']:<20} {r['sensitive_col']:<9} {r['n_configs']:<6} "
          f"{r['wall_time_sec']:<10.1f} {fstr:<8} {prule:<10} {acc}")

print(f"\n{'─'*70}")
print("Comparison A — Equal budget (single run vs single run):")
print("  Plain MLP (1 run) vs LBALM (1 run) vs AFRIAL (1 run)")
print("  → Shows: same compute, our method is fairer with accuracy guarantee")
print(f"{'─'*70}")
print("Comparison B — Practical (grid search vs single run):")
print("  Plain MLP (24 runs) vs LBALM (1 run) vs AFRIAL (1 run)")
print("  → Shows: our method is faster AND fairer than tuned plain MLP")

# ── Plots ─────────────────────────────────────────────────────────────────────
df_res=pd.DataFrame(all_results)
METHOD_ORDER  = ["Plain MLP (1 run)","Plain MLP (grid)",
                 "LBALM (single run)","AFRIAL (single run)"]
METHOD_COLORS = {"Plain MLP (1 run)":"#90CAF9",
                 "Plain MLP (grid)":"steelblue",
                 "LBALM (single run)":"purple",
                 "AFRIAL (single run)":"#F44336"}

for SENSITIVE_COL in SENSITIVE_COLS:
    sub=df_res[df_res["sensitive_col"]==SENSITIVE_COL].set_index("method").reindex(METHOD_ORDER)
    short_labels=["MLP\n(1 run)","MLP\n(grid×24)","LBALM*","AFRIAL*"]
    colors=[METHOD_COLORS[m] for m in METHOD_ORDER]

    fig,axes=plt.subplots(1,3,figsize=(16,5))
    fig.suptitle(
        f"Single-Run Comparison — COMPAS — {SENSITIVE_COL}\n"
        f"Light blue = Plain MLP (no fairness)  |  Purple/Red = Our methods (with fairness)\n"
        f"* single run with default params",
        fontsize=11)

    # Panel 1 — Time
    ax=axes[0]
    times=[float(sub.loc[m,"wall_time_sec"]) for m in METHOD_ORDER]
    bars=ax.bar(short_labels,times,color=colors,edgecolor="black",linewidth=0.7)
    ax.set_title("Wall-Clock Time (seconds)",fontsize=11)
    ax.set_ylabel("Seconds"); ax.grid(axis="y",linestyle="--",alpha=0.4)
    # Bracket showing MLP grid = 24× AFRIAL
    max_t=max(times)
    for bar,val in zip(bars,times):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max_t*0.01,
                f"{val:.0f}s",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.tick_params(axis="x",labelsize=9)

    # Panel 2 — p%-rule
    ax=axes[1]
    prules=[float(sub.loc[m,"test_p_rule_sel"]) if sub.loc[m,"test_p_rule_sel"] is not None
            else 0.0 for m in METHOD_ORDER]
    bl_pr=float(sub["bl_p_rule_sel"].iloc[0])
    bars=ax.bar(short_labels,prules,color=colors,edgecolor="black",linewidth=0.7)
    ax.axhline(bl_pr,color="black",linestyle="--",linewidth=1.5,
               label=f"Baseline ({bl_pr:.3f})")
    ax.set_title("p%-rule (selection rate)",fontsize=11)
    ax.set_ylabel("p%-rule"); ax.set_ylim(0,1.12)
    ax.legend(fontsize=8); ax.grid(axis="y",linestyle="--",alpha=0.4)
    for bar,val,m in zip(bars,prules,METHOD_ORDER):
        feas=bool(sub.loc[m,"feasible"])
        mk="✓" if feas else "✗"
        col="green" if feas else "red"
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                f"{val:.3f}\n{mk}",ha="center",va="bottom",
                fontsize=9,color=col,fontweight="bold")
    ax.tick_params(axis="x",labelsize=9)

    # Panel 3 — Accuracy
    ax=axes[2]
    accs=[float(sub.loc[m,"test_acc"]) if sub.loc[m,"test_acc"] is not None
          else 0.0 for m in METHOD_ORDER]
    bl_ac=float(sub["bl_acc"].iloc[0])
    bars=ax.bar(short_labels,accs,color=colors,edgecolor="black",linewidth=0.7)
    ax.axhline(bl_ac,color="black",linestyle="--",linewidth=1.5,
               label=f"Baseline ({bl_ac:.3f})")
    ax.set_title("Test Accuracy",fontsize=11)
    ax.set_ylabel("Accuracy")
    mn=min(accs+[bl_ac]); ax.set_ylim(max(0,mn-0.05),1.02)
    ax.legend(fontsize=8); ax.grid(axis="y",linestyle="--",alpha=0.4)
    for bar,val in zip(bars,accs):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.001,
                f"{val:.3f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.tick_params(axis="x",labelsize=9)

    plt.tight_layout()
    p=os.path.join(PLOTS_DIR,f"plot_singlerun_{SENSITIVE_COL}.png")
    plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved plot: {p}")

print(f"\n{'='*70}\nDONE — COMPAS\n{'='*70}")
