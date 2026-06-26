import torch

import numpy as np

import random

import argparse

import wandb

from omegaconf import OmegaConf, open_dict

from sklearn.metrics import roc_auc_score, classification_report

import torch.nn.functional as F

from source.dataset.abide import load_abide_data

from source.dataset.maml_dataloader import build_maml_datasets

from source.models.BNT import BrainNetworkTransformer

from source.models.BNT.maml_bnt import MAMLBNT

from source.training.maml_training import MAMLTrainer

from source.components import lr_scheduler_factory, logger_factory



RUN_SEEDS = [43, 44, 45, 46, 47]



def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False



BASE_CFG = OmegaConf.create({

    "dataset": {

        "name": "abide",

        "path": "data/abide.npy",

        "batch_size": 16, "train_set": 0.7, "val_set": 0.1,

        "stratified": True, "drop_last": True,

        "node_sz": 200, "node_feature_sz": 200, "timeseries_sz": 200,

    },

    "model": {

        "name": "BrainNetworkTransformer", "sizes": [360, 100],

        "pooling": [False, True], "pos_encoding": "none",

        "orthogonal": True, "freeze_center": True,

        "project_assignment": True, "pos_embed_dim": 360,

    },

    "training": {"name": "MAMLTrainer", "epochs": 200},

    "optimizer": [{"name": "Adam", "lr": 0.001, "weight_decay": 1e-4,

        "lr_scheduler": {

            "mode": "cos", "base_lr": 0.001, "target_lr": 1e-5,

            "warm_up_steps": 0, "warm_up_from": 0.0, "decay_factor": 0.1,

            "milestones": [0.3, 0.6, 0.9], "poly_power": 2.0, "lr_decay": 0.98

        }}],

    "datasz": {"percentage": 1.0},

    "preprocess": {"name": "mixup", "continus": True},

    "log_path": "result_maml_no_tta",

    "save_learnable_graph": False,

    "project": "brainnetworktransformer",

    "wandb_entity": "your-wandb-entity",

    "meta_lr": 0.001, "meta_batch_size": 4, "q_query": 10,

    "feature_dim": 800, "enable_maml": True,

})



def evaluate_direct(model, loader, device="cuda"):

    model.eval()

    preds, lbls = [], []

    with torch.no_grad():

        for batch in loader:

            ts_b, pear_b, lbl_b = batch[0], batch[1], batch[2]

            logits = model(ts_b.to(device), pear_b.to(device))

            probs = F.softmax(logits, dim=1)[:, 1]

            preds += probs.cpu().tolist()

            if lbl_b.dim() == 2:
                lbls += lbl_b[:, 1].tolist()
            else:
                lbls += lbl_b.tolist()

    auc = roc_auc_score(lbls, preds) * 100

    pred_bin = (np.array(preds) > 0.5).astype(int)

    lbls_arr = np.array(lbls).astype(int)

    acc = np.mean(pred_bin == lbls_arr) * 100

    rep = classification_report(lbls_arr, pred_bin, output_dict=True, zero_division=0)

    sen = rep.get("1", {}).get("recall", 0) * 100

    spec = rep.get("0", {}).get("recall", 0) * 100

    return auc, acc, sen, spec



def main(inner_lr=0.001, k_shot=10, num_inner_steps=3):

    all_intra_aucs, all_intra_accs, all_intra_sens, all_intra_specs = [], [], [], []

    all_inter_aucs, all_inter_accs, all_inter_sens, all_inter_specs = [], [], [], []

    all_epochs = []



    for run_idx, seed in enumerate(RUN_SEEDS):

        set_seed(seed)

        cfg = OmegaConf.create(OmegaConf.to_container(BASE_CFG, resolve=True))

        with open_dict(cfg):

            cfg.inner_lr = inner_lr

            cfg.num_inner_steps = num_inner_steps

            cfg.k_shot = k_shot

            cfg.unique_id = "no_tta_run{}_seed{}".format(run_idx, seed)

            cfg.total_steps = cfg.training.epochs

            cfg.steps_per_epoch = 1



        time_series, pearson, labels, sites = load_abide_data(cfg)

        train_ds, val_loader, test_intra_loader, test_inter_loader = build_maml_datasets(

            cfg, time_series, pearson, labels, sites, seed=seed)



        backbone = BrainNetworkTransformer(cfg).cuda()

        pretrained_path = (

            ""

            "BrainNetworkTransformer/result_source_only/"

            "BNT_source_only_run{}_seed{}/".format(run_idx, seed) +

            "model_BNT_source_only_run{}_seed{}.pt".format(run_idx, seed)

        )

        state_dict = torch.load(pretrained_path, map_location="cuda")

        backbone_state = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}

        backbone.load_state_dict(backbone_state, strict=False)

        print("Poids BNT charges : {}".format(pretrained_path))



        model = MAMLBNT(backbone, feature_dim=cfg.feature_dim).cuda()

        torch.nn.init.xavier_uniform_(model.classifier.fc.weight)

        torch.nn.init.zeros_(model.classifier.fc.bias)



        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=cfg.meta_lr, weight_decay=1e-4)

        lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer, cfg=cfg)

        logger = logger_factory(cfg)



        trainer = MAMLTrainer(

            cfg=cfg, model=model, optimizer=optimizer,

            lr_scheduler=lr_schedulers[0], train_dataset=train_ds,

            val_dataset=val_loader, test_intra_dataset=test_intra_loader,

            test_inter_dataset=test_inter_loader, logger=logger,

        )

        trainer.train()



        if len(trainer.best_results) > 0:

            r = trainer.best_results

            best_epoch = r["best_epoch"]

            all_epochs.append(best_epoch)

            auc_i, acc_i, sen_i, spec_i = evaluate_direct(model, test_intra_loader)

            auc_e, acc_e, sen_e, spec_e = evaluate_direct(model, test_inter_loader)

            all_intra_aucs.append(auc_i); all_intra_accs.append(acc_i)

            all_intra_sens.append(sen_i); all_intra_specs.append(spec_i)

            all_inter_aucs.append(auc_e); all_inter_accs.append(acc_e)

            all_inter_sens.append(sen_e); all_inter_specs.append(spec_e)

            print("Run{} seed{} | Epoch {:3d} | Intra AUC={:.2f}% | Inter AUC={:.2f}%".format(

                run_idx, seed, best_epoch, auc_i, auc_e))



    sep = "=" * 60

    print("\n" + sep)

    print("RESUME MAML SANS TTA — lr={} k={} steps={}".format(inner_lr, k_shot, num_inner_steps))

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

    print(sep + "\n")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--inner_lr", type=float, default=0.001)

    parser.add_argument("--k_shot", type=int, default=10)

    parser.add_argument("--num_inner_steps", type=int, default=3)

    args = parser.parse_args()

    wandb.init(mode="disabled")

    main(inner_lr=args.inner_lr, k_shot=args.k_shot, num_inner_steps=args.num_inner_steps)

