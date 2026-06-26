# ================================================================
# source/training/training.py
# Trainer unifié — compatible BNT standard ET modules avancés
# (DisentangledBNT, ImprovedAdversarialBNT, AttentionConsistentBNT, ImprovedMDBNT)
#
# AJOUTS vs version originale :
#   ✅ Sélection meilleure epoch basée sur Val AUC
#   ✅ Sauvegarde final_metrics.txt par run
#   ✅ Modèle sauvegardé avec nom unique (model_{run_id}.pt)
#   ✅ Résumé moyenne ± std sur tous les runs à la fin
# ================================================================

from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
from datetime import datetime


# ================================================================
# Stockage global pour le résumé final des N runs
# ================================================================
ALL_RUNS_RESULTS = []


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph

        self.is_advanced_model = self._detect_advanced_model()
        self.logger.info(f'Advanced model mode: {self.is_advanced_model}')

        self.needs_site_labels = self._detect_site_labels_needed()

        self.init_meters()

    def _detect_advanced_model(self) -> bool:
        return hasattr(self.model, 'backbone')

    def _detect_site_labels_needed(self) -> bool:
        model_class = self.model.__class__.__name__
        return model_class in [
            'ImprovedAdversarialBNT',
            'DisentangledBNT',
            'AttentionConsistentBNT',
            'ImprovedMDBNT',
        ]

    def _get_logits_from_output(self, output) -> torch.Tensor:
        if isinstance(output, dict):
            if 'classification_output' in output:
                return output['classification_output']
            elif 'task_output' in output:
                return output['task_output']
            else:
                raise KeyError(
                    f"Dict output ne contient ni 'classification_output' ni 'task_output'. "
                    f"Clés disponibles: {list(output.keys())}"
                )
        return output

    def _get_auxiliary_loss(self, output, label) -> torch.Tensor:
        if not isinstance(output, dict):
            return 0.0

        aux_loss = 0.0

        if 'total_loss' in output and torch.is_tensor(output['total_loss']):
            aux_loss += output['total_loss']

        if 'total_disentangle_loss' in output and torch.is_tensor(output['total_disentangle_loss']):
            aux_loss += output['total_disentangle_loss']

        if 'domain_loss' in output and torch.is_tensor(output['domain_loss']):
            current_lambda = output.get('current_lambda', self.config.get('lambda_domain', 0.2))
            aux_loss += current_lambda * output['domain_loss']

        if 'consistency_loss' in output and torch.is_tensor(output['consistency_loss']):
            lambda_c = self.config.get('lambda_consistency', 0.5)
            aux_loss += lambda_c * output['consistency_loss']

        return aux_loss

    def init_meters(self):
        self.train_loss, self.val_loss, \
            self.test_loss, self.train_accuracy, \
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()

        for time_series, node_feature, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            time_series  = time_series.cuda()
            node_feature = node_feature.cuda()
            label        = label.cuda()

            if self.config.preprocess.continus:
                time_series, node_feature, label = continus_mixup_data(
                    time_series, node_feature, y=label)

            if self.is_advanced_model:
                # Passer labels_int comme proxy de site au discriminateur
                output = self.model(
                    time_series, node_feature,
                    labels_int=label.argmax(dim=1).long() if label.dim() > 1 else label.long()
                )
            else:
                output = self.model(time_series, node_feature)

            logits    = self._get_logits_from_output(output)
            main_loss = self.loss_fn(logits, label)
            aux_loss  = self._get_auxiliary_loss(output, label)

            if torch.is_tensor(aux_loss):
                total_loss = main_loss + aux_loss
            else:
                total_loss = main_loss

            self.train_loss.update_with_weight(total_loss.item(), label.shape[0])

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            top1 = accuracy(logits, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for time_series, node_feature, label in dataloader:
            time_series  = time_series.cuda()
            node_feature = node_feature.cuda()
            label        = label.cuda()

            with torch.no_grad():
                if self.is_advanced_model:
                    output = self.model(time_series, node_feature)
                else:
                    output = self.model(time_series, node_feature)

            logits = self._get_logits_from_output(output)
            label  = label.float()

            loss = self.loss_fn(logits, label)
            loss_meter.update_with_weight(loss.item(), label.shape[0])

            top1 = accuracy(logits, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])

            result += F.softmax(logits, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5]  = 1
        result[result <= 0.5] = 0

        metric = precision_recall_fscore_support(
            labels, result, average='micro')

        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']

        return [auc] + list(metric) + recall

    # ================================================================
    # ✅ AJOUT 1 — Sauvegarde final_metrics.txt (meilleure epoch Val AUC)
    # ================================================================

    def save_final_metrics(self, best_epoch_data):
        """
        Sauvegarde les métriques de la meilleure epoch
        et enregistre dans ALL_RUNS_RESULTS pour le résumé final.
        """
        self.save_path.mkdir(exist_ok=True, parents=True)
        metrics_file = self.save_path / "final_metrics.txt"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "=" * 60,
            f"RÉSULTATS FINAUX - {self.model.__class__.__name__}",
            f"Date: {now}",
            f"Run ID: {self.config.unique_id}",
            "=" * 60, "",
            "📊 MEILLEURE EPOCH (basée sur Val AUC):",
            f"   Epoch            : {best_epoch_data['Epoch']}", "",
            "📈 MÉTRIQUES DE PERFORMANCE:",
            f"   Test Accuracy    : {best_epoch_data['Test Accuracy']:.2f}%",
            f"   Test AUC (AUROC) : {best_epoch_data['Test AUC'] * 100:.2f}%",
            f"   Test Sensitivity : {best_epoch_data['Test Sensitivity'] * 100:.2f}%",
            f"   Test Specificity : {best_epoch_data['Test Specificity'] * 100:.2f}%",
            f"   micro F1         : {best_epoch_data['micro F1'] * 100:.2f}%",
            f"   micro Recall     : {best_epoch_data['micro recall'] * 100:.2f}%",
            f"   micro Precision  : {best_epoch_data['micro precision'] * 100:.2f}%", "",
            "✅ MÉTRIQUES DE VALIDATION:",
            f"   Val AUC          : {best_epoch_data['Val AUC'] * 100:.2f}%",
            f"   Val Loss         : {best_epoch_data['Val Loss']:.4f}", "",
            "🔧 PARAMÈTRES D'ENTRAÎNEMENT:",
            f"   Train Accuracy   : {best_epoch_data['Train Accuracy']:.2f}%",
            f"   Train Loss       : {best_epoch_data['Train Loss']:.4f}",
            f"   Test Loss        : {best_epoch_data['Test Loss']:.4f}",
            f"   Model            : {self.model.__class__.__name__}",
            "=" * 60,
        ]

        with open(metrics_file, 'w') as f:
            f.write("\n".join(lines))

        self.logger.info(f"✅ Métriques sauvegardées : {metrics_file}")
        self.logger.info("\n" + "\n".join(lines))

        # Enregistrer pour le résumé final
        ALL_RUNS_RESULTS.append({
            'run_id':     self.config.unique_id,
            'best_epoch': best_epoch_data['Epoch'],
            'test_acc':   best_epoch_data['Test Accuracy'],
            'test_auc':   best_epoch_data['Test AUC'] * 100,
            'test_sen':   best_epoch_data['Test Sensitivity'] * 100,
            'test_spec':  best_epoch_data['Test Specificity'] * 100,
            'val_auc':    best_epoch_data['Val AUC'] * 100,
        })

    # ================================================================
    # ✅ AJOUT 2 — Résumé finale moyenne ± std sur tous les runs
    # ================================================================

    def save_summary(self):
        """
        Affiche et sauvegarde le résumé moyenne ± std
        sur tous les runs. Appelé après chaque run.
        """
        if len(ALL_RUNS_RESULTS) < 1:
            return

        accs  = [r['test_acc']  for r in ALL_RUNS_RESULTS]
        aucs  = [r['test_auc']  for r in ALL_RUNS_RESULTS]
        sens  = [r['test_sen']  for r in ALL_RUNS_RESULTS]
        specs = [r['test_spec'] for r in ALL_RUNS_RESULTS]

        lines = [
            "",
            "=" * 60,
            f"RÉSUMÉ — {len(ALL_RUNS_RESULTS)} RUN(S) COMPLÉTÉS",
            f"Modèle : {self.model.__class__.__name__}",
            "=" * 60,
            "",
            "📊 RÉSULTATS PAR RUN (meilleure epoch / Val AUC) :",
        ]

        for i, r in enumerate(ALL_RUNS_RESULTS):
            lines.append(
                f"   Run {i} | Epoch {r['best_epoch']:3d} | "
                f"Acc={r['test_acc']:.2f}%  "
                f"AUC={r['test_auc']:.2f}%  "
                f"Sen={r['test_sen']:.2f}%  "
                f"Spec={r['test_spec']:.2f}%  "
                f"ValAUC={r['val_auc']:.2f}%"
            )

        lines += [
            "",
            "📈 MOYENNE ± STD :",
            f"   Test Accuracy    : {np.mean(accs):.2f} ± {np.std(accs):.2f}%",
            f"   Test AUC (AUROC) : {np.mean(aucs):.2f} ± {np.std(aucs):.2f}%",
            f"   Test Sensitivity : {np.mean(sens):.2f} ± {np.std(sens):.2f}%",
            f"   Test Specificity : {np.mean(specs):.2f} ± {np.std(specs):.2f}%",
            "=" * 60,
        ]

        summary_text = "\n".join(lines)
        self.logger.info(summary_text)

        # Sauvegarder dans le dossier parent
        summary_file = self.save_path.parent / "summary_all_runs.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        self.logger.info(f"✅ Résumé sauvegardé : {summary_file}")

    def generate_save_learnable_matrix(self):
        learable_matrixs = []
        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label        = label.long()
            time_series  = time_series.cuda()
            node_feature = node_feature.cuda()
            label        = label.cuda()

            with torch.no_grad():
                if not self.is_advanced_model:
                    _, learable_matrix, _ = self.model(time_series, node_feature)
                    learable_matrixs.append(learable_matrix.cpu().detach().numpy())
                    labels += label.tolist()

        if learable_matrixs:
            self.save_path.mkdir(exist_ok=True, parents=True)
            np.save(self.save_path / "learnable_matrix.npy",
                    {'matrix': np.vstack(learable_matrixs), "label": np.array(labels)},
                    allow_pickle=True)

    # ================================================================
    # ✅ AJOUT 3 — Modèle sauvegardé avec nom unique
    # ================================================================

    def _get_model_name(self):
        """
        Génère automatiquement le nom du modèle selon :
          - La méthode utilisée (BNT_baseline, BNT_Dis, etc.)
          - L'atlas (cc200, aal, dos160)
          - Le run et seed (depuis unique_id)

        Exemples :
          model_BNT_baseline_cc200_run0_seed43.pt
          model_BNT_Dis_cc200_run0_seed43.pt
          model_BNT_Att_aal_run2_seed45.pt
        """
        # ── Détecter la méthode ──────────────────────────────────
        model_class = self.model.__class__.__name__

        if model_class == 'BrainNetworkTransformer':
            method = "BNT_baseline"
        elif model_class == 'ImprovedMDBNT':
            # Détecter quels modules sont actifs
            cfg = self.model.config
            use_dis  = cfg.get('use_disentanglement', False)
            use_att  = cfg.get('use_attention',       False)
            use_cont = cfg.get('use_contrastive',     False)
            use_adv  = cfg.get('use_adversarial',     False)

            if use_dis and not use_att and not use_cont:
                method = "BNT_Dis"
            elif use_att and not use_dis and not use_cont:
                method = "BNT_Att"
            elif use_cont and not use_dis and not use_att:
                method = "BNT_Cont"
            elif use_dis and use_att and not use_cont:
                method = "BNT_Dis_Att"
            elif use_dis and use_cont and not use_att:
                method = "BNT_Dis_Cont"
            elif use_att and use_cont and not use_dis:
                method = "BNT_Att_Cont"
            elif use_dis and use_att and use_cont:
                method = "BNT_Dis_Att_Cont"
            else:
                method = "BNT_MD"
        elif model_class == 'DisentangledBNT':
            method = "BNT_Dis"
        elif model_class == 'ImprovedAdversarialBNT':
            method = "BNT_Adv"
        elif model_class == 'AttentionConsistentBNT':
            method = "BNT_Att"
        elif model_class == 'ContrastiveBNT':
            method = "BNT_Cont"
        else:
            method = model_class

        # ── Détecter l'atlas depuis le chemin des données ────────
        data_path = str(self.config.get('dataset', {}).get('path', ''))
        if 'cc200' in data_path:
            atlas = "cc200"
        elif 'aal' in data_path:
            atlas = "aal"
        elif 'dos160' in data_path:
            atlas = "dos160"
        else:
            atlas = "atlas"

        # ── Extraire run et seed depuis unique_id ────────────────
        # unique_id format : "04-11-10-30-15_run0_seed43"
        uid = self.config.unique_id
        try:
            parts   = uid.split('_')
            run_str  = [p for p in parts if p.startswith('run')][0]
            seed_str = [p for p in parts if p.startswith('seed')][0]
        except Exception:
            run_str  = "run0"
            seed_str = "seed0"

        return f"model_{method}_{atlas}_{run_str}_{seed_str}.pt"

    def save_result(self, results):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "training_process.npy",
                results, allow_pickle=True)

        # ✅ Nom automatique selon méthode + atlas + run + seed
        model_filename = self._get_model_name()
        model_path     = self.save_path / model_filename
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"✅ Modèle sauvegardé : {model_path}")
        self.logger.info(f"   Nom : {model_filename}")

        # ✅ Renommer le dossier avec la même convention
        # Ex: 04-11-14-53-57_run3_seed46 → BNT_baseline_cc200_run3_seed46
        try:
            uid = self.config.unique_id
            parts    = uid.split('_')
            run_str  = [p for p in parts if p.startswith('run')][0]
            seed_str = [p for p in parts if p.startswith('seed')][0]

            # Extraire méthode et atlas depuis le nom du modèle
            # model_BNT_baseline_cc200_run3_seed46.pt
            model_stem   = model_filename.replace('model_', '').replace('.pt', '')
            # model_stem = BNT_baseline_cc200_run3_seed46
            # Enlever run et seed pour garder méthode+atlas
            method_atlas = model_stem.replace(f'_{run_str}_{seed_str}', '')
            # method_atlas = BNT_baseline_cc200

            new_folder_name = f"{method_atlas}_{run_str}_{seed_str}"
            new_save_path   = self.save_path.parent / new_folder_name

            if not new_save_path.exists():
                self.save_path.rename(new_save_path)
                self.logger.info(f"✅ Dossier renommé : {new_save_path.name}")
                self.save_path = new_save_path
            else:
                self.logger.info(f"⚠️  Dossier déjà existant : {new_save_path.name}")
        except Exception as e:
            self.logger.warning(f"⚠️  Renommage dossier échoué : {e}")

    def train(self):
        training_process = []
        self.current_step = 0

        # ✅ Variables pour la sélection best epoch / Val AUC
        best_val_auc    = -1.0
        best_epoch_data = None

        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])

            val_result  = self.test_per_epoch(
                self.val_dataloader,  self.val_loss,  self.val_accuracy)
            test_result = self.test_per_epoch(
                self.test_dataloader, self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test Sen:{test_result[-1]:.4f}',
                f'Test Spec:{test_result[-2]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}',
                f'Model:{self.model.__class__.__name__}',
            ]))

            wandb.log({
                "Train Loss":        self.train_loss.avg,
                "Train Accuracy":    self.train_accuracy.avg,
                "Test Loss":         self.test_loss.avg,
                "Test Accuracy":     self.test_accuracy.avg,
                "Val AUC":           val_result[0],
                "Test AUC":          test_result[0],
                'Test Sensitivity':  test_result[-1],
                'Test Specificity':  test_result[-2],
                'micro F1':          test_result[-4],
                'micro recall':      test_result[-5],
                'micro precision':   test_result[-6],
            })

            epoch_data = {
                "Epoch":             epoch,
                "Train Loss":        self.train_loss.avg,
                "Train Accuracy":    self.train_accuracy.avg,
                "Test Loss":         self.test_loss.avg,
                "Test Accuracy":     self.test_accuracy.avg,
                "Test AUC":          test_result[0],
                'Test Sensitivity':  test_result[-1],
                'Test Specificity':  test_result[-2],
                'micro F1':          test_result[-4],
                'micro recall':      test_result[-5],
                'micro precision':   test_result[-6],
                "Val AUC":           val_result[0],
                "Val Loss":          self.val_loss.avg,
                "Model":             self.model.__class__.__name__,
            }

            training_process.append(epoch_data)

            # ✅ Sélection meilleure epoch basée sur Val AUC (min 10 epochs)
            if epoch >= 30 and val_result[0] > best_val_auc:
                best_val_auc    = val_result[0]
                best_epoch_data = epoch_data.copy()

        # ✅ Fin des epochs — sauvegarder métriques de la meilleure epoch
        if best_epoch_data is not None:
            self.save_final_metrics(best_epoch_data)

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()

        self.save_result(training_process)

        # ✅ Afficher résumé après chaque run
        self.save_summary()
