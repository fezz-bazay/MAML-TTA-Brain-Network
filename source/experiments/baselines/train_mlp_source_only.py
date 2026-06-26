import torch
import numpy as np
import random
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import torch.nn.functional as F
from omegaconf import OmegaConf
from source.models.mlp_brain import MLPBrain
from source.dataset.preprocess import StandardScaler

SOURCE_SITES = ["NYU", "UM_1", "USM", "UCLA_1"]
RUN_SEEDS    = [43, 44, 45, 46, 47]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS       = 200
LR           = 0.0001
PATIENCE     = 20
MIN_EPOCH    = 30

cfg = OmegaConf.create({
    "dataset": {"node_sz": 200, "node_feature_sz": 200, "timeseries_sz": 200},
})

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def evaluate(model, loader):
    model.eval()
    preds, lbls = [], []
    with torch.no_grad():
        for ts_b, pear_b, lbl_b in loader:
            logits = model(ts_b.to(DEVICE), pear_b.to(DEVICE))
            probs  = F.softmax(logits, dim=1)[:, 1]
            preds += probs.cpu().tolist()
            lbls  += lbl_b.tolist()
    if len(set(lbls)) < 2: return 0.0, 0.0, 0.0, 0.0
    auc      = roc_auc_score(lbls, preds) * 100
    pred_bin = (np.array(preds) > 0.5).astype(int)
    acc      = np.mean(pred_bin == np.array(lbls).astype(int)) * 100
    rep      = classification_report(np.array(lbls).astype(int), pred_bin, output_dict=True, zero_division=0)
    return auc, acc, rep.get("1",{}).get("recall",0)*100, rep.get("0",{}).get("recall",0)*100

def main():
    data    = np.load("data/abide.npy", allow_pickle=True).item()
    ts_raw  = data["timeseires"]; pear = data["corr"]
    labels  = data["label"];      sites = data["site"]
    scaler  = StandardScaler(mean=np.mean(ts_raw), std=np.std(ts_raw))
    ts_t    = torch.from_numpy(scaler.transform(ts_raw)).float()
    pear_t  = torch.from_numpy(pear).float()
    lbl_t   = torch.from_numpy(labels.astype(np.float32))
    source_mask = np.isin(sites, SOURCE_SITES)
    inter_loader = DataLoader(TensorDataset(ts_t[~source_mask], pear_t[~source_mask], lbl_t[~source_mask]), batch_size=16, shuffle=False)

    res = {k: [] for k in ["ia","ic","is","isp","ea","ec","es","esp"]}
    for run_idx, seed in enumerate(RUN_SEEDS):
        set_seed(seed)
        print("\nRun {} | Seed {}".format(run_idx, seed))
        src_idx = np.where(source_mask)[0]
        train_idx, valtest = train_test_split(src_idx, test_size=0.30, stratify=labels[src_idx].astype(int), random_state=seed)
        val_idx, test_idx  = train_test_split(valtest, test_size=0.50, stratify=labels[valtest].astype(int), random_state=seed)
        train_loader = DataLoader(TensorDataset(ts_t[train_idx], pear_t[train_idx], lbl_t[train_idx]), batch_size=16, shuffle=True, drop_last=True)
        val_loader   = DataLoader(TensorDataset(ts_t[val_idx],   pear_t[val_idx],   lbl_t[val_idx]),   batch_size=16)
        test_loader  = DataLoader(TensorDataset(ts_t[test_idx],  pear_t[test_idx],  lbl_t[test_idx]),  batch_size=16)

        model = MLPBrain(cfg).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        best_val, best_epoch, best_state, patience_count = -1.0, 0, None, 0

        for epoch in range(EPOCHS):
            model.train()
            for ts_b, pear_b, lbl_b in train_loader:
                loss = F.cross_entropy(model(ts_b.to(DEVICE), pear_b.to(DEVICE)), lbl_b.to(DEVICE).long())
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            val_auc, _, _, _ = evaluate(model, val_loader)
            if epoch % 10 == 0: print("Epoch {:3d} | ValAUC={:.2f}%".format(epoch, val_auc))
            if epoch >= MIN_EPOCH and val_auc > best_val:
                best_val, best_epoch, best_state, patience_count = val_auc, epoch, {k: v.cpu().clone() for k,v in model.state_dict().items()}, 0
            elif epoch >= MIN_EPOCH:
                patience_count += 1
                if patience_count >= PATIENCE: print("Early stop epoch {}".format(epoch)); break

        model.load_state_dict(best_state)
        save_dir = Path("result_mlp/MLP_run{}_seed{}".format(run_idx, seed))
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model_mlp_run{}_seed{}.pt".format(run_idx, seed))
        ai,ci,si,spi = evaluate(model, test_loader)
        ae,ce,se,spe = evaluate(model, inter_loader)
        for k,v in zip(["ia","ic","is","isp","ea","ec","es","esp"],[ai,ci,si,spi,ae,ce,se,spe]): res[k].append(v)
        print("Run{} | Intra AUC={:.2f}% | Inter AUC={:.2f}%".format(run_idx, ai, ae))

    print("\n" + "="*60)
    print("RESUME MLP SOURCE-ONLY")
    print("="*60)
    print("\nINTRA-SITE :")
    for k,n in zip(["ia","ic","is","isp"],["AUC","Acc","Sen","Spec"]):
        print("  {} : {:.2f} +/- {:.2f}%".format(n, np.mean(res[k]), np.std(res[k])))
    print("\nINTER-SITE :")
    for k,n in zip(["ea","ec","es","esp"],["AUC","Acc","Sen","Spec"]):
        print("  {} : {:.2f} +/- {:.2f}%".format(n, np.mean(res[k]), np.std(res[k])))

if __name__ == "__main__":
    main()
