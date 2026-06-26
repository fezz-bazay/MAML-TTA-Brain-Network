import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from source.models.BNT import BrainNetworkTransformer
from source.models.BNT.maml_bnt import MAMLBNT
from source.dataset.preprocess import StandardScaler
from source.dataset.abide import load_abide_data
from source.dataset.maml_dataloader import build_maml_datasets
from source.training.maml_training import MAMLTrainer
from source.components import lr_scheduler_factory, logger_factory

ALL_SITES_SORTED = [
    "NYU", "UM_1", "UCLA_1", "USM", "PITT", "YALE", "MAX_MUN", "KKI",
    "TRINITY", "STANFORD", "CALTECH", "SDSU", "LEUVEN_2", "OLIN",
    "UM_2", "SBL", "LEUVEN_1", "CMU", "UCLA_2"
]

RUN_SEEDS   = [43, 44, 45, 46, 47]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_DIM = 800

cfg_bnt = OmegaConf.create({
    "dataset": {"node_sz": 200, "node_feature_sz": 200, "timeseries_sz": 200},
    "model": {
        "name": "BrainNetworkTransformer", "sizes": [360, 100],
        "pooling": [False, True], "pos_encoding": "none",
        "orthogonal": True, "freeze_center": True,
        "project_assignment": True, "pos_embed_dim": 360,
    },
    "feature_dim": 800,
})

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(n_sites):
    source_sites = ALL_SITES_SORTED[:n_sites]
    inter_sites  = ALL_SITES_SORTED[n_sites:]
    print("\nN={} | Source: {} | Test inter: {}".format(
        n_sites, source_sites, inter_sites))

    all_inter_aucs, all_inter_accs = [], []
    all_inter_sens, all_inter_specs = [], []

    for run_idx, seed in enumerate(RUN_SEEDS):
        set_seed(seed)

        # Charger BNT pre-entraine pour ce N
        # N=4 → modeles dans result_source_only, autres dans result_scaling_bnt
        if n_sites == 4:
            bnt_path = ("result_source_only/BNT_source_only_run{}_seed{}/"
                        "model_BNT_source_only_run{}_seed{}.pt".format(
                        run_idx, seed, run_idx, seed))
        else:
            bnt_path = ("result_scaling_bnt/BNT_source_only_N{}_run{}_seed{}/"
                        "model_BNT_source_only_run{}_seed{}.pt".format(
                        n_sites, run_idx, seed, run_idx, seed))

        backbone = BrainNetworkTransformer(cfg_bnt).to(DEVICE)
        backbone.load_state_dict(torch.load(bnt_path, map_location=DEVICE))
        print("BNT charge : {}".format(bnt_path))

        model = MAMLBNT(backbone, feature_dim=FEATURE_DIM).to(DEVICE)
        torch.nn.init.xavier_uniform_(model.classifier.fc.weight)
        torch.nn.init.zeros_(model.classifier.fc.bias)

        # Config MAML avec les bons sites sources
        cfg = OmegaConf.create({
            "dataset": {
                "name": "abide",
                "path": "data/abide.npy",
                "batch_size": 16, "train_set": 0.7, "val_set": 0.1,
                "stratified": True, "drop_last": True,
                "node_sz": 200, "node_feature_sz": 200, "timeseries_sz": 200,
            },
            "model": {"name": "BrainNetworkTransformer"},
            "training": {"name": "MAMLTrainer", "epochs": 200},
            "optimizer": [{"name": "Adam", "lr": 0.001, "weight_decay": 1e-4,
                "lr_scheduler": {
                    "mode": "cos", "base_lr": 0.001, "target_lr": 1e-5,
                    "warm_up_steps": 0, "warm_up_from": 0.0, "decay_factor": 0.1,
                    "milestones": [0.3, 0.6, 0.9], "poly_power": 2.0, "lr_decay": 0.98
                }}],
            "datasz": {"percentage": 1.0},
            "preprocess": {"name": "mixup", "continus": True},
            "log_path": "result_scaling_bnt_maml",
            "save_learnable_graph": False,
            "project": "brainnetworktransformer",
            "wandb_entity": "your-wandb-entity",
            "meta_lr": 0.001, "meta_batch_size": 4, "q_query": 10,
            "feature_dim": FEATURE_DIM, "enable_maml": True,
            "inner_lr": 0.001, "num_inner_steps": 3, "k_shot": 3,
            "total_steps": 200, "steps_per_epoch": 1,
            "unique_id": "scaling_bnt_maml_N{}_run{}_seed{}".format(n_sites, run_idx, seed),
            "total_steps": 200, "steps_per_epoch": 1,
            "source_sites": source_sites,
        })

        time_series, pearson, labels, sites = load_abide_data(cfg)
        train_ds, val_loader, test_intra_loader, test_inter_loader = \
            build_maml_datasets(cfg, time_series, pearson, labels, sites,
                               seed=seed, source_sites=source_sites)

        optimizer     = torch.optim.Adam(
            model.classifier.fc.parameters(), lr=cfg.meta_lr, weight_decay=1e-4)
        lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
        import wandb; wandb.init(mode="disabled")
        logger        = logger_factory(cfg)

        trainer = MAMLTrainer(
            cfg=cfg, model=model, optimizer=optimizer,
            lr_scheduler=lr_schedulers[0], train_dataset=train_ds,
            val_dataset=val_loader, test_intra_dataset=test_intra_loader,
            test_inter_dataset=test_inter_loader, logger=logger,
        )
        trainer.train()

        # Sauvegarder modele
        save_dir = Path("result_scaling_bnt_maml/N{}_run{}_seed{}".format(
            n_sites, run_idx, seed))
        save_dir.mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(),
                   save_dir / "model_maml_scaling_N{}_run{}_seed{}.pt".format(
                   n_sites, run_idx, seed))

        if len(trainer.best_results) > 0:
            r = trainer.best_results
            all_inter_aucs.append(r["inter_auc"])
            all_inter_accs.append(r["inter_acc"])
            all_inter_sens.append(r["inter_sen"])
            all_inter_specs.append(r["inter_spec"])
            print("Run{} seed{} | Epoch {:3d} | Inter AUC={:.2f}%".format(
                run_idx, seed, r["best_epoch"], r["inter_auc"]))

    sep = "=" * 60
    print("\n" + sep + "\nRESUME MAML+TTA N={}\n".format(n_sites) + sep)
    print("INTER-SITE :")
    print("  AUC  : {:.2f} +/- {:.2f}%".format(
        np.mean(all_inter_aucs), np.std(all_inter_aucs)))
    print("  Acc  : {:.2f} +/- {:.2f}%".format(
        np.mean(all_inter_accs), np.std(all_inter_accs)))
    print("  Sen  : {:.2f} +/- {:.2f}%".format(
        np.mean(all_inter_sens), np.std(all_inter_sens)))
    print("  Spec : {:.2f} +/- {:.2f}%".format(
        np.mean(all_inter_specs), np.std(all_inter_specs)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sites", type=int, required=True)
    args = parser.parse_args()
    main(args.n_sites)
