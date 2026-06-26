"""

TTA sans MAML — BNT source-only + fine-tune SGD au test

Memes parametres : lr=0.001, k=10, steps=3

"""

import torch

import numpy as np

import random

import copy

from sklearn.metrics import roc_auc_score, classification_report

from sklearn.model_selection import train_test_split

import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from omegaconf import OmegaConf

from source.models.BNT import BrainNetworkTransformer

from source.dataset.preprocess import StandardScaler



RUN_SEEDS  = [43, 44, 45, 46, 47]

SOURCE_SITES = ['NYU', 'UM_1', 'USM', 'UCLA_1']

K_SHOT     = 3

SGD_LR     = 0.001

SGD_STEPS  = 3

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'



cfg = OmegaConf.create({

    'dataset': {'node_sz': 200, 'node_feature_sz': 200, 'timeseries_sz': 200},

    'model': {

        'name': 'BrainNetworkTransformer', 'sizes': [360, 100],

        'pooling': [False, True], 'pos_encoding': 'none',

        'orthogonal': True, 'freeze_center': True,

        'project_assignment': True, 'pos_embed_dim': 360,

    },

})



def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)



def evaluate_loader(model, loader):

    model.eval()

    preds, lbls = [], []

    with torch.no_grad():

        for ts_b, pear_b, lbl_b in loader:

            logits = model(ts_b.to(DEVICE), pear_b.to(DEVICE))

            probs  = F.softmax(logits, dim=1)[:, 1]

            preds += probs.cpu().tolist()

            lbls  += lbl_b[:, 1].tolist()

    auc      = roc_auc_score(lbls, preds) * 100

    pred_bin = (np.array(preds) > 0.5).astype(int)

    lbls_arr = np.array(lbls).astype(int)

    acc      = np.mean(pred_bin == lbls_arr) * 100

    rep      = classification_report(lbls_arr, pred_bin, output_dict=True, zero_division=0)

    sen      = rep.get('1', {}).get('recall', 0) * 100

    spec     = rep.get('0', {}).get('recall', 0) * 100

    return auc, acc, sen, spec



def tta_sgd(model, support_ts, support_pear, support_lbl, lr=0.001, steps=3):

    """Fine-tune le classifier avec SGD standard — sans MAML"""

    model_copy = copy.deepcopy(model)

    # Freeze backbone — fine-tune seulement le classifier

    for name, param in model_copy.named_parameters():

        if 'fc' not in name:

            param.requires_grad = False

    optimizer = torch.optim.SGD(

        [p for p in model_copy.parameters() if p.requires_grad],

        lr=lr

    )

    model_copy.train()

    for _ in range(steps):

        logits = model_copy(support_ts.to(DEVICE), support_pear.to(DEVICE))

        loss   = F.cross_entropy(logits, support_lbl.to(DEVICE).long())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return model_copy



def evaluate_inter_with_tta(model, ts_all, pear_all, lbl_oh, sites_all):

    """Evaluation inter-site avec TTA SGD par site"""

    inter_sites = [s for s in np.unique(sites_all) if s not in SOURCE_SITES]

    all_preds, all_lbls = [], []



    for site in inter_sites:

        site_mask = sites_all == site

        ts_site   = ts_all[site_mask]

        pear_site = pear_all[site_mask]

        lbl_site  = lbl_oh[site_mask]

        n_site    = len(ts_site)



        if n_site < K_SHOT + 1:

            continue



        # k exemples support

        k = min(K_SHOT, n_site // 2)

        support_idx = np.random.choice(n_site, k, replace=False)

        query_mask  = np.ones(n_site, dtype=bool)

        query_mask[support_idx] = False



        support_ts   = torch.from_numpy(ts_site[support_idx]).float()

        support_pear = torch.from_numpy(pear_site[support_idx]).float()

        support_lbl  = torch.from_numpy(lbl_site[support_idx, 1].astype(int))



        query_ts     = torch.from_numpy(ts_site[query_mask]).float()

        query_pear   = torch.from_numpy(pear_site[query_mask]).float()

        query_lbl    = torch.from_numpy(lbl_site[query_mask]).float()



        # TTA avec SGD standard

        adapted_model = tta_sgd(model, support_ts, support_pear, support_lbl,

                                 lr=SGD_LR, steps=SGD_STEPS)

        adapted_model.eval()



        with torch.no_grad():

            logits = adapted_model(query_ts.to(DEVICE), query_pear.to(DEVICE))

            probs  = F.softmax(logits, dim=1)[:, 1]

            all_preds += probs.cpu().tolist()

            all_lbls  += query_lbl[:, 1].tolist()



    if len(set(all_lbls)) < 2:

        return 0.0, 0.0, 0.0, 0.0



    auc      = roc_auc_score(all_lbls, all_preds) * 100

    pred_bin = (np.array(all_preds) > 0.5).astype(int)

    lbls_arr = np.array(all_lbls).astype(int)

    acc      = np.mean(pred_bin == lbls_arr) * 100

    rep      = classification_report(lbls_arr, pred_bin, output_dict=True, zero_division=0)

    sen      = rep.get('1', {}).get('recall', 0) * 100

    spec     = rep.get('0', {}).get('recall', 0) * 100

    return auc, acc, sen, spec



def main():

    data     = np.load('data/abide.npy',

                       allow_pickle=True).item()

    ts_raw   = data['timeseires']

    pear_raw = data['corr']

    labels   = data['label']

    sites    = data['site']



    scaler   = StandardScaler(mean=np.mean(ts_raw), std=np.std(ts_raw))

    ts_norm  = scaler.transform(ts_raw)

    ts_t     = torch.from_numpy(ts_norm).float()

    pear_t   = torch.from_numpy(pear_raw).float()

    n        = len(labels)

    lbl_oh   = torch.zeros(n, 2)

    lbl_oh[torch.arange(n), torch.from_numpy(labels.astype(int))] = 1.0



    source_mask = np.isin(sites, SOURCE_SITES)

    intra_loader_full = DataLoader(

        TensorDataset(ts_t[source_mask], pear_t[source_mask], lbl_oh[source_mask]),

        batch_size=16, shuffle=False

    )



    all_intra_aucs, all_intra_accs = [], []

    all_intra_sens, all_intra_specs = [], []

    all_inter_aucs, all_inter_accs = [], []

    all_inter_sens, all_inter_specs = [], []



    for run_idx, seed in enumerate(RUN_SEEDS):

        set_seed(seed)

        model_path = (

            f""

            f"BrainNetworkTransformer/result_source_only/"

            f"BNT_source_only_run{run_idx}_seed{seed}/"

            f"model_BNT_source_only_run{run_idx}_seed{seed}.pt"

        )

        model = BrainNetworkTransformer(cfg).to(DEVICE)

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        print(f"Modele charge : {model_path}")



        # Intra — evaluation directe sans TTA

        src_idx      = np.where(source_mask)[0]

        src_lbl_int  = labels[src_idx].astype(int)

        _, valtest   = train_test_split(src_idx, test_size=0.30,

                                        stratify=src_lbl_int, random_state=seed)

        _, test_idx  = train_test_split(valtest, test_size=0.50,

                                        stratify=src_lbl_int[np.isin(src_idx, valtest)],

                                        random_state=seed)

        intra_loader = DataLoader(

            TensorDataset(ts_t[test_idx], pear_t[test_idx], lbl_oh[test_idx]),

            batch_size=16, shuffle=False

        )

        auc_i, acc_i, sen_i, spec_i = evaluate_loader(model, intra_loader)



        # Inter — TTA avec SGD standard (pas MAML)

        auc_e, acc_e, sen_e, spec_e = evaluate_inter_with_tta(

            model,

            ts_norm, pear_raw, lbl_oh.numpy(), sites

        )



        all_intra_aucs.append(auc_i); all_intra_accs.append(acc_i)

        all_intra_sens.append(sen_i); all_intra_specs.append(spec_i)

        all_inter_aucs.append(auc_e); all_inter_accs.append(acc_e)

        all_inter_sens.append(sen_e); all_inter_specs.append(spec_e)



        print(f"Run{run_idx} seed{seed} | Intra AUC={auc_i:.2f}% | Inter AUC={auc_e:.2f}%")



    print(f"\n{'='*60}")

    print(f"RESUME TTA SANS MAML (SGD fine-tune au test)")

    print(f"{'='*60}")

    print(f"\nINTRA-SITE (seen) :")

    print(f"  AUC  : {np.mean(all_intra_aucs):.2f} +/- {np.std(all_intra_aucs):.2f}%")

    print(f"  Acc  : {np.mean(all_intra_accs):.2f} +/- {np.std(all_intra_accs):.2f}%")

    print(f"  Sen  : {np.mean(all_intra_sens):.2f} +/- {np.std(all_intra_sens):.2f}%")

    print(f"  Spec : {np.mean(all_intra_specs):.2f} +/- {np.std(all_intra_specs):.2f}%")

    print(f"\nINTER-SITE (unseen) :")

    print(f"  AUC  : {np.mean(all_inter_aucs):.2f} +/- {np.std(all_inter_aucs):.2f}%")

    print(f"  Acc  : {np.mean(all_inter_accs):.2f} +/- {np.std(all_inter_accs):.2f}%")

    print(f"  Sen  : {np.mean(all_inter_sens):.2f} +/- {np.std(all_inter_sens):.2f}%")

    print(f"  Spec : {np.mean(all_inter_specs):.2f} +/- {np.std(all_inter_specs):.2f}%")

    print(f"{'='*60}\n")



if __name__ == '__main__':

    main()
