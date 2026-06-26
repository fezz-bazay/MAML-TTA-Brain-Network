import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import wandb
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from source.dataset.abide import load_abide_data
from source.dataset.maml_dataloader import build_maml_datasets
from source.models.mlp_brain import MLPBrain
from source.training.maml_training import MAMLTrainer
from source.components import lr_scheduler_factory, logger_factory

RUN_SEEDS   = [43, 44, 45, 46, 47]
FEATURE_DIM = 256

class MAMLMLPBrain(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone   = backbone.backbone
        class ClassifierWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(256, 2)
            def forward(self, x):
                return self.fc(x)
        self.classifier = ClassifierWrapper()

    def forward_features(self, ts, pear):
        bz = pear.shape[0]
        x  = pear.view(bz, -1)
        return self.backbone(x)

    def forward(self, ts, pear):
        return self.classifier.fc(self.forward_features(ts, pear))

BASE_CFG = OmegaConf.create({
    "dataset": {
        "name": "abide",
        "path": "data/abide.npy",
        "batch_size": 16, "train_set": 0.7, "val_set": 0.1,
        "stratified": True, "drop_last": True,
        "node_sz": 200, "node_feature_sz": 200, "timeseries_sz": 200,
    },
    "model": {"name": "MLPBrain"},
    "training": {"name": "MAMLTrainer", "epochs": 200},
    "optimizer": [{"name": "Adam", "lr": 0.001, "weight_decay": 1e-4,
        "lr_scheduler": {
            "mode": "cos", "base_lr": 0.001, "target_lr": 1e-5,
            "warm_up_steps": 0, "warm_up_from": 0.0, "decay_factor": 0.1,
            "milestones": [0.3, 0.6, 0.9], "poly_power": 2.0, "lr_decay": 0.98
        }}],
    "datasz": {"percentage": 1.0},
    "preprocess": {"name": "mixup", "continus": True},
    "log_path": "result_mlp_maml",
    "save_learnable_graph": False,
    "project": "brainnetworktransformer",
    "wandb_entity": "your-wandb-entity",
    "meta_lr": 0.001, "meta_batch_size": 4, "q_query": 10,
    "feature_dim": FEATURE_DIM, "enable_maml": True,
})

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(inner_lr=0.001, k_shot=3, num_inner_steps=3):
    all_intra_aucs, all_intra_accs, all_intra_sens, all_intra_specs = [], [], [], []
    all_inter_aucs, all_inter_accs, all_inter_sens, all_inter_specs = [], [], [], []

    import builtins
    seeds = getattr(builtins, 'RUN_SEEDS_OVERRIDE', RUN_SEEDS)

    for run_idx, seed in enumerate(seeds):
        set_seed(seed)
        cfg = OmegaConf.create(OmegaConf.to_container(BASE_CFG, resolve=True))
        with open_dict(cfg):
            cfg.inner_lr        = inner_lr
            cfg.num_inner_steps = num_inner_steps
            cfg.k_shot          = k_shot
            cfg.unique_id       = "mlp_maml_run{}_seed{}".format(run_idx, seed)
            cfg.total_steps     = cfg.training.epochs
            cfg.steps_per_epoch = 1

        mlp_cfg = OmegaConf.create({
            "dataset": {"node_sz": 200, "node_feature_sz": 200, "timeseries_sz": 200}
        })
        backbone = MLPBrain(mlp_cfg).cuda()
        pretrained_path = (
            "result_mlp/MLP_run{}_seed{}/".format(run_idx, seed) +
            "model_mlp_run{}_seed{}.pt".format(run_idx, seed)
        )
        backbone.load_state_dict(torch.load(pretrained_path, map_location="cuda"))
        print("MLP charge : {}".format(pretrained_path))

        model = MAMLMLPBrain(backbone).cuda()
        torch.nn.init.xavier_uniform_(model.classifier.fc.weight)
        torch.nn.init.zeros_(model.classifier.fc.bias)

        time_series, pearson, labels, sites = load_abide_data(cfg)
        train_ds, val_loader, test_intra_loader, test_inter_loader = \
            build_maml_datasets(cfg, time_series, pearson, labels, sites, seed=seed)

        optimizer     = torch.optim.Adam(
            model.classifier.fc.parameters(), lr=cfg.meta_lr, weight_decay=1e-4)
        lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)
        logger        = logger_factory(cfg)

        trainer = MAMLTrainer(
            cfg=cfg, model=model, optimizer=optimizer,
            lr_scheduler=lr_schedulers[0], train_dataset=train_ds,
            val_dataset=val_loader, test_intra_dataset=test_intra_loader,
            test_inter_dataset=test_inter_loader, logger=logger,
        )
        trainer.train()

        if len(trainer.best_results) > 0:
            r = trainer.best_results
            all_intra_aucs.append(r["intra_auc"]); all_intra_accs.append(r["intra_acc"])
            all_intra_sens.append(r["intra_sen"]); all_intra_specs.append(r["intra_spec"])
            all_inter_aucs.append(r["inter_auc"]); all_inter_accs.append(r["inter_acc"])
            all_inter_sens.append(r["inter_sen"]); all_inter_specs.append(r["inter_spec"])
            print("Run{} seed{} | Epoch {:3d} | Intra AUC={:.2f}% | Inter AUC={:.2f}%".format(
                run_idx, seed, r["best_epoch"], r["intra_auc"], r["inter_auc"]))

    sep = "=" * 60
    print("\n" + sep + "\nRESUME MLP + MAML + TTA\n" + sep)
    print("\nINTRA-SITE :")
    print("  AUC  : {:.2f} +/- {:.2f}%".format(np.mean(all_intra_aucs), np.std(all_intra_aucs)))
    print("  Acc  : {:.2f} +/- {:.2f}%".format(np.mean(all_intra_accs), np.std(all_intra_accs)))
    print("  Sen  : {:.2f} +/- {:.2f}%".format(np.mean(all_intra_sens), np.std(all_intra_sens)))
    print("  Spec : {:.2f} +/- {:.2f}%".format(np.mean(all_intra_specs), np.std(all_intra_specs)))
    print("\nINTER-SITE :")
    print("  AUC  : {:.2f} +/- {:.2f}%".format(np.mean(all_inter_aucs), np.std(all_inter_aucs)))
    print("  Acc  : {:.2f} +/- {:.2f}%".format(np.mean(all_inter_accs), np.std(all_inter_accs)))
    print("  Sen  : {:.2f} +/- {:.2f}%".format(np.mean(all_inter_sens), np.std(all_inter_sens)))
    print("  Spec : {:.2f} +/- {:.2f}%".format(np.mean(all_inter_specs), np.std(all_inter_specs)))
    print(sep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner_lr",        type=float, default=0.001)
    parser.add_argument("--k_shot",          type=int,   default=3)
    parser.add_argument("--num_inner_steps", type=int,   default=3)
    parser.add_argument("--single_seed",     type=int,   default=-1)
    args = parser.parse_args()
    wandb.init(mode="disabled")
    if args.single_seed >= 0:
        import builtins
        builtins.RUN_SEEDS_OVERRIDE = [args.single_seed]
    main(inner_lr=args.inner_lr, k_shot=args.k_shot,
         num_inner_steps=args.num_inner_steps)
