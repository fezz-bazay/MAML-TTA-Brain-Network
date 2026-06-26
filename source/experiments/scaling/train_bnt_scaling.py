# source/sweeps/sweep_maml.py
# Sweep W&B — BNT+MAML avec protocole inter-site
# Sources : NYU + UM_1 + USM + UCLA_1
# Unseen  : tous les autres sites ABIDE

import wandb
import torch
import numpy as np
import random
from omegaconf import OmegaConf, open_dict

from source.dataset.abide import load_abide_data
from source.dataset.maml_dataloader import build_maml_datasets
from source.models.BNT import BrainNetworkTransformer
from source.models.BNT.maml_bnt import MAMLBNT
from source.training.maml_training import MAMLTrainer
from source.components import lr_scheduler_factory, logger_factory

RUN_SEEDS = [43, 44, 45, 46, 47]


import argparse as _argparse
_parser = _argparse.ArgumentParser()
_parser.add_argument("--inner_lr",        type=float, default=0.001)
_parser.add_argument("--k_shot",          type=int,   default=3)
_parser.add_argument("--num_inner_steps", type=int,   default=3)
_parser.add_argument("--seed",            type=int,   default=-1)
_parser.add_argument("--run_idx",         type=int,   default=-1)
_parser.add_argument("--log_path",        type=str,   default="result_maml_source_only")
_parser.add_argument("--unique_id_prefix",type=str,   default="sweep")
_args, _ = _parser.parse_known_args()
if _args.seed >= 0 and _args.run_idx >= 0:
    RUN_SEEDS = [_args.seed]
    _RUN_IDX_OFFSET = _args.run_idx
else:
    _RUN_IDX_OFFSET = 0

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


BASE_CFG = OmegaConf.create({
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
    'training': {
        'name':   'MAMLTrainer',
        'epochs': 200,
    },
    'optimizer': [{
        'name':         'Adam',
        'lr':           0.001,
        'weight_decay': 1e-4,
        'lr_scheduler': {
            'mode':          'cos',
            'base_lr':       0.001,
            'target_lr':     1e-5,
            'warm_up_steps': 0,
            'warm_up_from':  0.0,
            'decay_factor':  0.1,
            'milestones':    [0.3, 0.6, 0.9],
            'poly_power':    2.0,
            'lr_decay':      0.98,
        }
    }],
    'datasz':   {'percentage': 1.0},
    'preprocess': {'name': 'mixup', 'continus': True},
    'log_path':             _args.log_path,
    'save_learnable_graph': False,
    'project':              'brainnetworktransformer',
    'wandb_entity':         'your-wandb-entity',
    'meta_lr':         0.001,
    'meta_batch_size': 4,
    'q_query':         10,
    'feature_dim':     800,
    'enable_maml':     True,
})


def train(inner_lr=0.001, k_shot=10, num_inner_steps=3):
    run = wandb.init(mode='disabled')

    all_val_aucs    = []
    all_intra_aucs  = []
    all_intra_accs  = []
    all_intra_sens  = []
    all_intra_specs = []
    all_inter_aucs  = []
    all_inter_accs  = []
    all_inter_sens  = []
    all_inter_specs = []
    all_epochs      = []

    for _i, seed in enumerate(RUN_SEEDS):
        run_idx = _i + _RUN_IDX_OFFSET
        set_seed(seed)

        cfg = OmegaConf.create(OmegaConf.to_container(BASE_CFG, resolve=True))
        with open_dict(cfg):
            cfg.inner_lr        = inner_lr
            cfg.num_inner_steps = num_inner_steps
            cfg.k_shot          = k_shot
            cfg.unique_id       = f"{_args.unique_id_prefix}_run{run_idx}_seed{seed}"
            cfg.total_steps     = cfg.training.epochs
            cfg.steps_per_epoch = 1

        # Charger toutes les donnees ABIDE
        time_series, pearson, labels, sites = load_abide_data(cfg)

        # Afficher les sites uniques au premier run pour debug
        if run_idx == 0:
            sites_np = sites.numpy() if hasattr(sites, 'numpy') else np.array(sites)
            unique, counts = np.unique(sites_np, return_counts=True)
            print("\nSites ABIDE disponibles (id : count) :")
            for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:8]:
                print(f"  {u} : {c}")

        # Build datasets avec protocole inter-site
        train_ds, val_loader, test_intra_loader, test_inter_loader = \
            build_maml_datasets(cfg, time_series, pearson, labels, sites, seed=seed)

        # Modele avec poids BNT pre-entraines
        backbone = BrainNetworkTransformer(cfg).cuda()

        pretrained_path = (
            f""
            f"BrainNetworkTransformer/result_source_only/"
            f"BNT_source_only_run{run_idx}_seed{seed}/"
            f"model_BNT_source_only_run{run_idx}_seed{seed}.pt"
        )
        state_dict     = torch.load(pretrained_path, map_location='cuda')
        backbone_state = {k: v for k, v in state_dict.items()
                         if not k.startswith('fc.')}
        backbone.load_state_dict(backbone_state, strict=False)
        print(f"Poids BNT charges : {pretrained_path}")

        model = MAMLBNT(backbone, feature_dim=cfg.feature_dim).cuda()
        torch.nn.init.xavier_uniform_(model.classifier.fc.weight)
        torch.nn.init.zeros_(model.classifier.fc.bias)

        optimizer = torch.optim.Adam(
        model.classifier.parameters(),  # BNT frozen
        lr=cfg.meta_lr,
        weight_decay=1e-4
        )

        lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
        logger        = logger_factory(cfg)

        trainer = MAMLTrainer(
            cfg                = cfg,
            model              = model,
            optimizer          = optimizer,
            lr_scheduler       = lr_schedulers[0],
            train_dataset      = train_ds,
            val_dataset        = val_loader,
            test_intra_dataset = test_intra_loader,
            test_inter_dataset = test_inter_loader,
            logger             = logger,
        )

        trainer.train()

        if len(trainer.best_results) > 0:
            r = trainer.best_results
            all_val_aucs.append(r['val_auc'])
            all_intra_aucs.append(r['intra_auc'])
            all_intra_accs.append(r['intra_acc'])
            all_intra_sens.append(r['intra_sen'])
            all_intra_specs.append(r['intra_spec'])
            all_inter_aucs.append(r['inter_auc'])
            all_inter_accs.append(r['inter_acc'])
            all_inter_sens.append(r['inter_sen'])
            all_inter_specs.append(r['inter_spec'])
            all_epochs.append(r['best_epoch'])

    # Statistiques finales
    def ms(arr):
        return np.mean(arr) if arr else 0.0, np.std(arr) if arr else 0.0

    m_val,        _ = ms(all_val_aucs)
    m_intra_auc, s_intra_auc = ms(all_intra_aucs)
    m_intra_acc, s_intra_acc = ms(all_intra_accs)
    m_intra_sen, s_intra_sen = ms(all_intra_sens)
    m_intra_spec,s_intra_spec= ms(all_intra_specs)
    m_inter_auc, s_inter_auc = ms(all_inter_aucs)
    m_inter_acc, s_inter_acc = ms(all_inter_accs)
    m_inter_sen, s_inter_sen = ms(all_inter_sens)
    m_inter_spec,s_inter_spec= ms(all_inter_specs)

    print(f"\n{'='*60}")
    print(f"RESUME BNT+MAML — lr={inner_lr} steps={num_inner_steps} k={k_shot}")
    print(f"{'='*60}")
    print(f"\nRESULTATS PAR RUN :")
    for i, seed in enumerate(RUN_SEEDS):
        if i < len(all_intra_aucs):
            print(
                f"  Run {i} | Seed {seed} | Epoch {all_epochs[i]:3d} | "
                f"Intra AUC={all_intra_aucs[i]:.2f}% | "
                f"Inter AUC={all_inter_aucs[i]:.2f}%"
            )

    print(f"\nINTRA-SITE (seen) :")
    print(f"  AUC  : {m_intra_auc:.2f} ± {s_intra_auc:.2f}%")
    print(f"  Acc  : {m_intra_acc:.2f} ± {s_intra_acc:.2f}%")
    print(f"  Sen  : {m_intra_sen:.2f} ± {s_intra_sen:.2f}%")
    print(f"  Spec : {m_intra_spec:.2f} ± {s_intra_spec:.2f}%")
    print(f"\nINTER-SITE (unseen) :")
    print(f"  AUC  : {m_inter_auc:.2f} ± {s_inter_auc:.2f}%")
    print(f"  Acc  : {m_inter_acc:.2f} ± {s_inter_acc:.2f}%")
    print(f"  Sen  : {m_inter_sen:.2f} ± {s_inter_sen:.2f}%")
    print(f"  Spec : {m_inter_spec:.2f} ± {s_inter_spec:.2f}%")
    print(f"{'='*60}\n")

    wandb.log({
        'Val AUC':    m_val,
        'Intra AUC':  m_intra_auc,
        'Intra Acc':  m_intra_acc,
        'Inter AUC':  m_inter_auc,
        'Inter Acc':  m_inter_acc,
    })

    run.finish()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inner_lr',        type=float, default=0.001)
    parser.add_argument('--k_shot',          type=int,   default=10)
    parser.add_argument('--num_inner_steps', type=int,   default=3)
    parser.add_argument('--log_path',        type=str,   default='result_maml_source_only')
    parser.add_argument('--unique_id_prefix',type=str,   default='sweep')
    args = parser.parse_args()

    train(
        inner_lr=args.inner_lr,
        k_shot=args.k_shot,
        num_inner_steps=args.num_inner_steps
    )
