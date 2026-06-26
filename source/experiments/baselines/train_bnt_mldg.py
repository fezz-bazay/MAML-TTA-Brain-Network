import torch
import numpy as np
import random
import copy
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import torch.nn.functional as F
from omegaconf import OmegaConf
from source.models.BNT import BrainNetworkTransformer
from source.dataset.preprocess import StandardScaler

SOURCE_SITES = ["NYU", "UM_1", "USM", "UCLA_1"]
RUN_SEEDS    = [43, 44, 45, 46, 47]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS       = 200
LR           = 0.0001
META_LR      = 0.0001
INNER_LR     = 0.0001
PATIENCE     = 20
MIN_EPOCH    = 30

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, loader):
    model.eval()
    preds, lbls = [], []
    with torch.no_grad():
        for ts_b, pear_b, lbl_b in loader:
            logits = model(ts_b.to(DEVICE), pear_b.to(DEVICE))
            probs  = F.softmax(logits, dim=1)[:, 1]
            preds += probs.cpu().tolist()
            lbls  += lbl_b.tolist()
    auc      = roc_auc_score(lbls, preds) * 100
    pred_bin = (np.array(preds) > 0.5).astype(int)
    lbls_arr = np.array(lbls).astype(int)
    acc      = np.mean(pred_bin == lbls_arr) * 100
    rep      = classification_report(lbls_arr, pred_bin, output_dict=True, zero_division=0)
    sen      = rep.get("1", {}).get("recall", 0) * 100
    spec     = rep.get("0", {}).get("recall", 0) * 100
    return auc, acc, sen, spec

def get_batch(loader_iter, loader):
    try:
        return next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        return next(loader_iter)

cfg = OmegaConf.create({
    "dataset": {"node_sz": 200, "node_feature_sz": 200, "timeseries_sz": 200},
    "model": {
        "name": "BrainNetworkTransformer", "sizes": [360, 100],
        "pooling": [False, True], "pos_encoding": "none",
        "orthogonal": True, "freeze_center": True,
        "project_assignment": True, "pos_embed_dim": 360,
    },
})

def main():
    data    = np.load("data/abide.npy",
                      allow_pickle=True).item()
    ts_raw  = data["timeseires"]
    pear    = data["corr"]
    labels  = data["label"]
    sites   = data["site"]

    scaler  = StandardScaler(mean=np.mean(ts_raw), std=np.std(ts_raw))
    ts_norm = scaler.transform(ts_raw)
    ts_t    = torch.from_numpy(ts_norm).float()
    pear_t  = torch.from_numpy(pear).float()
    lbl_t   = torch.from_numpy(labels.astype(np.float32))

    source_mask = np.isin(sites, SOURCE_SITES)
    inter_mask  = ~source_mask
    inter_loader = DataLoader(
        TensorDataset(ts_t[inter_mask], pear_t[inter_mask], lbl_t[inter_mask]),
        batch_size=16, shuffle=False)

    all_intra_aucs, all_intra_accs, all_intra_sens, all_intra_specs = [], [], [], []
    all_inter_aucs, all_inter_accs, all_inter_sens, all_inter_specs = [], [], [], []

    for run_idx, seed in enumerate(RUN_SEEDS):
        set_seed(seed)
        print("\n" + "="*40)
        print("Run {} | Seed {}".format(run_idx, seed))
        print("="*40)

        src_idx     = np.where(source_mask)[0]
        src_lbl_int = labels[src_idx].astype(int)

        train_idx, valtest_idx = train_test_split(
            src_idx, test_size=0.30, stratify=src_lbl_int, random_state=seed)
        val_idx, test_idx = train_test_split(
            valtest_idx, test_size=0.50,
            stratify=src_lbl_int[np.isin(src_idx, valtest_idx)],
            random_state=seed)

        train_sites = sites[train_idx]
        site_loaders = {}
        for site in SOURCE_SITES:
            site_mask = train_sites == site
            site_idx  = train_idx[site_mask]
            if len(site_idx) >= 4:
                site_loaders[site] = DataLoader(
                    TensorDataset(ts_t[site_idx], pear_t[site_idx], lbl_t[site_idx]),
                    batch_size=16, shuffle=True, drop_last=False)

        site_list = list(site_loaders.keys())

        val_loader = DataLoader(
            TensorDataset(ts_t[val_idx], pear_t[val_idx], lbl_t[val_idx]),
            batch_size=16, shuffle=False)
        test_loader = DataLoader(
            TensorDataset(ts_t[test_idx], pear_t[test_idx], lbl_t[test_idx]),
            batch_size=16, shuffle=False)

        model     = BrainNetworkTransformer(cfg).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

        site_iters = {s: iter(site_loaders[s]) for s in site_list}

        best_val_auc   = -1.0
        best_epoch     = 0
        best_state     = None
        patience_count = 0

        n_batches = min(len(site_loaders[s]) for s in site_list)

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0

            for _ in range(n_batches):
                # Choisir aleatoirement 1 site comme meta-val
                meta_val_site   = random.choice(site_list)
                meta_train_sites = [s for s in site_list if s != meta_val_site]

                # Meta-train loss
                meta_train_loss = torch.tensor(0.0).to(DEVICE)
                for site in meta_train_sites:
                    try:
                        ts_b, pear_b, lbl_b = next(site_iters[site])
                    except StopIteration:
                        site_iters[site] = iter(site_loaders[site])
                        ts_b, pear_b, lbl_b = next(site_iters[site])
                    logits = model(ts_b.to(DEVICE), pear_b.to(DEVICE))
                    meta_train_loss = meta_train_loss + F.cross_entropy(logits, lbl_b.to(DEVICE).long())
                meta_train_loss = meta_train_loss / len(meta_train_sites)

                # Inner loop : creer modele temporaire
                model_temp = BrainNetworkTransformer(cfg).to(DEVICE)
                model_temp.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
                opt_temp   = torch.optim.SGD(model_temp.parameters(), lr=INNER_LR)
                model_temp.train()
                for site in meta_train_sites:
                    try:
                        ts_b, pear_b, lbl_b = next(site_iters[site])
                    except StopIteration:
                        site_iters[site] = iter(site_loaders[site])
                        ts_b, pear_b, lbl_b = next(site_iters[site])
                    logits_t = model_temp(ts_b.to(DEVICE), pear_b.to(DEVICE))
                    loss_t   = F.cross_entropy(logits_t, lbl_b.to(DEVICE).long())
                    opt_temp.zero_grad()
                    loss_t.backward()
                    opt_temp.step()

                # Meta-val loss sur modele adapte
                try:
                    ts_v, pear_v, lbl_v = next(site_iters[meta_val_site])
                except StopIteration:
                    site_iters[meta_val_site] = iter(site_loaders[meta_val_site])
                    ts_v, pear_v, lbl_v = next(site_iters[meta_val_site])

                model_temp.eval()
                with torch.no_grad():
                    logits_v = model_temp(ts_v.to(DEVICE), pear_v.to(DEVICE))
                meta_val_loss = F.cross_entropy(logits_v, lbl_v.to(DEVICE).long())

                # Total MLDG loss
                loss = meta_train_loss + meta_val_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_auc, _, _, _ = evaluate(model, val_loader)

            if epoch % 10 == 0:
                print("Epoch {:3d} | Loss={:.4f} | ValAUC={:.2f}%".format(
                    epoch, total_loss / n_batches, val_auc))

            if epoch >= MIN_EPOCH and val_auc > best_val_auc:
                best_val_auc   = val_auc
                best_epoch     = epoch
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            elif epoch >= MIN_EPOCH:
                patience_count += 1
                if patience_count >= PATIENCE:
                    print("Early stopping epoch {}".format(epoch))
                    break

        model.load_state_dict(best_state)

        save_dir = Path("result_bnt_mldg/BNT_mldg_run{}_seed{}".format(run_idx, seed))
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(),
                   save_dir / "model_bnt_mldg_run{}_seed{}.pt".format(run_idx, seed))

        auc_i, acc_i, sen_i, spec_i = evaluate(model, test_loader)
        auc_e, acc_e, sen_e, spec_e = evaluate(model, inter_loader)

        all_intra_aucs.append(auc_i); all_intra_accs.append(acc_i)
        all_intra_sens.append(sen_i); all_intra_specs.append(spec_i)
        all_inter_aucs.append(auc_e); all_inter_accs.append(acc_e)
        all_inter_sens.append(sen_e); all_inter_specs.append(spec_e)

        print("Run{} seed{} | Epoch {:3d} | Intra AUC={:.2f}% | Inter AUC={:.2f}%".format(
            run_idx, seed, best_epoch, auc_i, auc_e))

    sep = "=" * 60
    print("\n" + sep)
    print("RESUME BNT + MLDG")
    print(sep)
    print("\nINTRA-SITE (seen) :")
    print("  AUC  : {:.2f} +/- {:.2f}%".format(np.mean(all_intra_aucs), np.std(all_intra_aucs)))
    print("  Acc  : {:.2f} +/- {:.2f}%".format(np.mean(all_intra_accs), np.std(all_intra_accs)))
    print("  Sen  : {:.2f} +/- {:.2f}%".format(np.mean(all_intra_sens), np.std(all_intra_sens)))
    print("  Spec : {:.2f} +/- {:.2f}%".format(np.mean(all_intra_specs), np.std(all_intra_specs)))
    print("\nINTER-SITE (unseen) :")
    print("  AUC  : {:.2f} +/- {:.2f}%".format(np.mean(all_inter_aucs), np.std(all_inter_aucs)))
    print("  Acc  : {:.2f} +/- {:.2f}%".format(np.mean(all_inter_accs), np.std(all_inter_accs)))
    print("  Sen  : {:.2f} +/- {:.2f}%".format(np.mean(all_inter_sens), np.std(all_inter_sens)))
    print("  Spec : {:.2f} +/- {:.2f}%".format(np.mean(all_inter_specs), np.std(all_inter_specs)))
    print(sep)

if __name__ == "__main__":
    main()
