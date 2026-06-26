"""

Script pour entraîner BNT seulement sur les 4 sites sources.

NYU + UM_1 + USM + UCLA_1 = 424 sujets

Reproduit exactement le BNT baseline mais avec filtre source.

Seeds: 43, 44, 45, 46, 47

"""



import torch

import numpy as np

import random

from pathlib import Path

from omegaconf import OmegaConf, open_dict

from sklearn.model_selection import StratifiedShuffleSplit

from torch.utils.data import TensorDataset, DataLoader



from source.models.BNT import BrainNetworkTransformer

from source.training.training import Train

from source.components import lr_scheduler_factory, logger_factory



# ── Config ──────────────────────────────────────────────────

SOURCE_SITES = ['NYU', 'UM_1', 'USM', 'UCLA_1']

RUN_SEEDS    = [43, 44, 45, 46, 47]



CFG = OmegaConf.create({

    'dataset': {

        'name':            'abide',

        'path':            'data/abide.npy',

        'batch_size':      16,

        'train_set':       0.7,

        'val_set':         0.1,

        'stratified':      True,

        'drop_last':       True,

        'node_sz':         200,

        'node_feature_sz': 200,

        'timeseries_sz':   200,

    },

    'model': {

        'name':               'BrainNetworkTransformer',

        'sizes':              [360, 100],

        'pooling':            [False, True],

        'pos_encoding':       'none',

        'orthogonal':         True,

        'freeze_center':      True,

        'project_assignment': True,

        'pos_embed_dim':      360,

    },

    'training': {'name': 'Train', 'epochs': 200},

    'optimizer': [{

        'name':         'Adam',

        'lr':           0.0001,

        'weight_decay': 1e-4,

        'lr_scheduler': {

            'mode':          'cos',

            'base_lr':       0.0001,

            'target_lr':     1e-5,

            'warm_up_steps': 0,

            'warm_up_from':  0.0,

            'decay_factor':  0.1,

            'milestones':    [0.3, 0.6, 0.9],

            'poly_power':    2.0,

            'lr_decay':      0.98,

        }

    }],

    'datasz':             {'percentage': 1.0},

    'preprocess':         {'name': 'mixup', 'continus': True},

    'log_path':           'result_source_only',

    'save_learnable_graph': False,

    'project':            'brainnetworktransformer',

    'wandb_entity':       'your-wandb-entity',

})





def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark     = False





def load_source_data(cfg):

    """Charge seulement les 4 sites sources."""

    data             = np.load(cfg.dataset.path, allow_pickle=True).item()

    time_series      = data["timeseires"]

    pearson          = data["corr"]

    labels           = data["label"]

    sites            = data['site']



    # Filtre source-only

    mask        = np.isin(sites, SOURCE_SITES)

    time_series = time_series[mask]

    pearson     = pearson[mask]

    labels      = labels[mask]

    sites       = sites[mask]



    print(f"Sujets source-only : {mask.sum()} / {len(sites)} total")

    unique, counts = np.unique(sites, return_counts=True)

    for s, c in zip(unique, counts):

        print(f"  {s} : {c}")



    # Normalisation

    from source.dataset.preprocess import StandardScaler

    scaler      = StandardScaler(mean=np.mean(time_series), std=np.std(time_series))

    time_series = scaler.transform(time_series)



    time_series, pearson, labels = [

        torch.from_numpy(d).float()

        for d in (time_series, pearson, labels)

    ]



    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = pearson.shape[1:]

        cfg.dataset.timeseries_sz = time_series.shape[2]



    return time_series, pearson, labels





def make_dataloaders(cfg, time_series, pearson, labels, seed):
    """Split stratifié 70/10/20 comme BNT baseline."""
    n      = len(labels)
    idx    = np.arange(n)
    lbl_np = labels.numpy().astype(int)

    # One-hot encoding comme BNT baseline
    labels_onehot = torch.zeros(n, 2)
    labels_onehot[torch.arange(n), labels.long()] = 1.0

    # Train+val vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(idx, lbl_np))

    # Train vs val
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    train_idx, val_idx = next(sss2.split(trainval_idx, lbl_np[trainval_idx]))
    train_idx = trainval_idx[train_idx]
    val_idx   = trainval_idx[val_idx]

    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    def make_loader(idx, shuffle, drop_last=False):
        ds = TensorDataset(time_series[idx], pearson[idx], labels_onehot[idx])
        return DataLoader(ds, batch_size=cfg.dataset.batch_size,
                         shuffle=shuffle, drop_last=drop_last)

    return [
        make_loader(train_idx, shuffle=True,  drop_last=True),
        make_loader(val_idx,   shuffle=False, drop_last=False),
        make_loader(test_idx,  shuffle=False, drop_last=False),
    ]





def main():

    import wandb

    wandb.init(mode='disabled')



    print("=" * 60)

    print("BNT SOURCE-ONLY TRAINING")

    print(f"Sites: {SOURCE_SITES}")

    print("=" * 60)



    for run_idx, seed in enumerate(RUN_SEEDS):

        print(f"\n{'='*40}")

        print(f"Run {run_idx} | Seed {seed}")

        print(f"{'='*40}")



        set_seed(seed)



        cfg = OmegaConf.create(OmegaConf.to_container(CFG, resolve=True))

        with open_dict(cfg):

            cfg.unique_id       = f"BNT_source_only_run{run_idx}_seed{seed}"

            cfg.total_steps     = cfg.training.epochs * 20

            cfg.steps_per_epoch = 20



        # Charger données source-only

        time_series, pearson, labels = load_source_data(cfg)



        # DataLoaders

        dataloaders = make_dataloaders(cfg, time_series, pearson, labels, seed)



        # Modèle

        model = BrainNetworkTransformer(cfg).cuda()



        # Optimizer et scheduler

        optimizer     = torch.optim.Adam(

            model.parameters(), lr=1e-4, weight_decay=1e-4)

        lr_schedulers = lr_scheduler_factory(cfg.optimizer, cfg)

        logger        = logger_factory(cfg)



        # Trainer

        trainer = Train(

            cfg          = cfg,

            model        = model,

            optimizers   = [optimizer],

            lr_schedulers= lr_schedulers,

            dataloaders  = dataloaders,

            logger       = logger,

        )

        trainer.train()



        # Sauvegarder le modèle avec nom standard pour MAML

        save_dir = Path(cfg.log_path) / cfg.unique_id

        save_dir.mkdir(exist_ok=True, parents=True)

        model_path = save_dir / f"model_BNT_source_only_run{run_idx}_seed{seed}.pt"

        torch.save(model.state_dict(), model_path)

        print(f"Modele sauvegarde : {model_path}")



        # Evaluation inter-site apres chaque run

        import torch.nn.functional as F

        from sklearn.metrics import roc_auc_score, classification_report

        from source.dataset.preprocess import StandardScaler as SC



        data_all   = np.load(cfg.dataset.path, allow_pickle=True).item()

        ts_all     = data_all['timeseires']

        pear_all   = data_all['corr']

        lbl_all    = data_all['label']

        sites_all  = data_all['site']



        sc2        = SC(mean=np.mean(ts_all), std=np.std(ts_all))

        ts_all     = sc2.transform(ts_all)

        ts_t2      = torch.from_numpy(ts_all).float()

        pear_t2    = torch.from_numpy(pear_all).float()

        n2         = len(lbl_all)

        lbl_oh2    = torch.zeros(n2, 2)

        lbl_oh2[torch.arange(n2), torch.from_numpy(lbl_all.astype(int))] = 1.0



        inter_mask2 = ~np.isin(sites_all, SOURCE_SITES)

        inter_loader2 = DataLoader(

            TensorDataset(ts_t2[inter_mask2], pear_t2[inter_mask2], lbl_oh2[inter_mask2]),

            batch_size=16, shuffle=False)



        model.eval()

        preds2, lbls2 = [], []

        with torch.no_grad():

            for ts_b, pear_b, lbl_b in inter_loader2:

                logits = model(ts_b.cuda(), pear_b.cuda())

                probs  = F.softmax(logits, dim=1)[:, 1]

                preds2 += probs.cpu().tolist()

                lbls2  += lbl_b[:, 1].tolist()



        auc_inter  = roc_auc_score(lbls2, preds2) * 100

        pred_bin   = (np.array(preds2) > 0.5).astype(int)

        lbls_arr   = np.array(lbls2).astype(int)

        acc_inter  = np.mean(pred_bin == lbls_arr) * 100

        rep        = classification_report(lbls_arr, pred_bin,

                                           output_dict=True, zero_division=0)

        sen_inter  = rep.get('1', {}).get('recall', 0) * 100

        spec_inter = rep.get('0', {}).get('recall', 0) * 100



        print(f"  INTER Run{run_idx} seed{seed}: AUC={auc_inter:.2f}% ACC={acc_inter:.2f}% SEN={sen_inter:.2f}% SPEC={spec_inter:.2f}%")

        ALL_INTER_RESULTS.append({

            'run': run_idx, 'seed': seed,

            'auc': auc_inter, 'acc': acc_inter,

            'sen': sen_inter, 'spec': spec_inter

        })



    # Resume inter-site

    print("\n" + "=" * 60)

    print("BNT SOURCE-ONLY TRAINING COMPLETE")

    print("=" * 60)

    if ALL_INTER_RESULTS:

        aucs  = [r['auc']  for r in ALL_INTER_RESULTS]

        accs  = [r['acc']  for r in ALL_INTER_RESULTS]

        sens  = [r['sen']  for r in ALL_INTER_RESULTS]

        specs = [r['spec'] for r in ALL_INTER_RESULTS]

        print("\n INTER-SITE (15 unseen sites) :")

        for r in ALL_INTER_RESULTS:

            print(f"   Run{r['run']} seed{r['seed']}: AUC={r['auc']:.2f}% ACC={r['acc']:.2f}% SEN={r['sen']:.2f}% SPEC={r['spec']:.2f}%")

        print(f"   MOYENNE: AUC={np.mean(aucs):.2f}+-{np.std(aucs):.2f}% ACC={np.mean(accs):.2f}+-{np.std(accs):.2f}% SEN={np.mean(sens):.2f}+-{np.std(sens):.2f}% SPEC={np.mean(specs):.2f}+-{np.std(specs):.2f}%")





ALL_INTER_RESULTS = []

if __name__ == '__main__':

    main()

