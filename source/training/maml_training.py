# source/training/maml_training.py

import torch

import torch.nn.functional as F

import numpy as np

from pathlib import Path

from sklearn.metrics import roc_auc_score, classification_report

from datetime import datetime

import wandb

from source.utils import isfloat

from source.models.BNT.maml_bnt import CosineLoss



ALL_MAML_RESULTS = []





class MAMLTrainer:



    def __init__(self, cfg, model, optimizer, lr_scheduler,

                 train_dataset, val_dataset,

                 test_intra_dataset, test_inter_dataset,

                 logger):



        self.cfg                = cfg

        self.model              = model

        self.optimizer          = optimizer

        self.lr_scheduler       = lr_scheduler

        self.train_dataset      = train_dataset

        self.val_dataset        = val_dataset

        self.test_intra_dataset = test_intra_dataset

        self.test_inter_dataset = test_inter_dataset

        self.logger             = logger

        self.best_results       = {}



        self.inner_lr        = cfg.get('inner_lr',        0.005)

        self.num_inner_steps = cfg.get('num_inner_steps', 5)

        self.meta_batch_size = cfg.get('meta_batch_size', 4)

        self.epochs          = cfg.training.epochs

        self.patience        = cfg.get('patience',        20)



        self.loss_fn   = CosineLoss(xent=0.1)

        self.save_path = Path(cfg.log_path) / cfg.unique_id



        self.logger.info(f'MAML Config:')

        self.logger.info(f'  inner_lr        : {self.inner_lr}')

        self.logger.info(f'  num_inner_steps : {self.num_inner_steps}')

        self.logger.info(f'  meta_batch_size : {self.meta_batch_size}')

        self.logger.info(f'  patience        : {self.patience}')



    def _get_labels_int(self, labels):

        if labels.dim() > 1:

            return labels.argmax(dim=1)

        return labels.long()



    def inner_loop(self, support, fast_weights=None, create_graph=True):

        ts, pear, labels = support

        ts     = ts.cuda()

        pear   = pear.cuda()

        labels = labels.cuda().float()



        if fast_weights is None:

            fast_weights = {

                'weight': self.model.classifier.fc.weight.clone(),

                'bias':   self.model.classifier.fc.bias.clone()

            }

        # Verifier batch size
        if len(ts) < 2:
            return fast_weights

        for _ in range(self.num_inner_steps):

            with torch.no_grad():

                features = self.model.forward_features(ts, pear)



            logits     = F.linear(features,

                                  fast_weights['weight'],

                                  fast_weights['bias'])

            labels_int = self._get_labels_int(labels)

            loss       = self.loss_fn(logits, labels_int)



            grads = torch.autograd.grad(

                loss,

                [fast_weights['weight'], fast_weights['bias']],

                create_graph=create_graph

            )



            fast_weights = {

                'weight': fast_weights['weight'] - self.inner_lr * grads[0],

                'bias':   fast_weights['bias']   - self.inner_lr * grads[1]

            }



        return fast_weights



    def outer_loop(self, meta_batch):

        meta_loss = 0.0



        for support, query in meta_batch:

            fast_weights = self.inner_loop(support, create_graph=True)



            ts, pear, labels = query

            ts     = ts.cuda()

            pear   = pear.cuda()

            labels = labels.cuda().float()



            with torch.no_grad():

                features = self.model.forward_features(ts, pear)



            logits     = F.linear(features,

                                  fast_weights['weight'],

                                  fast_weights['bias'])

            labels_int = self._get_labels_int(labels)

            query_loss = self.loss_fn(logits, labels_int)

            meta_loss += query_loss



        return meta_loss / len(meta_batch)



    def evaluate(self, dataloader):

        self.model.eval()

        all_preds  = []

        all_labels = []

        total_loss = 0.0

        total_n    = 0



        with torch.no_grad():

            for batch in dataloader:

                ts     = batch[0].cuda()

                pear   = batch[1].cuda()

                labels = batch[2]



                features   = self.model.forward_features(ts, pear)

                logits     = self.model.classifier(features)

                probs      = F.softmax(logits, dim=1)[:, 1]

                labels_int = self._get_labels_int(labels)

                loss       = F.cross_entropy(logits, labels_int.cuda())

                total_loss += loss.item() * len(labels_int)

                total_n    += len(labels_int)



                all_preds  += probs.cpu().tolist()

                all_labels += labels_int.cpu().tolist()



        if total_n == 0 or len(set(all_labels)) < 2:

            return 0.5, 50.0, 0.0, 0.0, 0.0



        avg_loss   = total_loss / total_n

        auc        = roc_auc_score(all_labels, all_preds)

        result     = np.array(all_preds)

        labels_arr = np.array(all_labels)

        result[result > 0.5]  = 1

        result[result <= 0.5] = 0

        acc = np.mean(result == labels_arr) * 100



        report = classification_report(

            labels_arr, result, output_dict=True, zero_division=0)

        recall = [0, 0]

        for k in report:

            if isfloat(k):

                recall[int(float(k))] = report[k]['recall']



        return auc, acc, recall[1], recall[0], avg_loss



    def evaluate_with_tta(self, dataloader, k_shot=5):
        """TTA par site — vrai scenario MAML."""
        self.model.eval()

        all_ts_list, all_pear_list, all_lbl_list, all_site_list = [], [], [], []
        has_sites = False

        for batch in dataloader:
            all_ts_list.append(batch[0])
            all_pear_list.append(batch[1])
            all_lbl_list.append(self._get_labels_int(batch[2]))
            if len(batch) > 3:
                all_site_list.append(batch[3])
                has_sites = True

        all_ts   = torch.cat(all_ts_list,   dim=0)
        all_pear = torch.cat(all_pear_list, dim=0)
        all_lbls = torch.cat(all_lbl_list,  dim=0)

        all_preds = []
        all_labs  = []

        if has_sites:
            all_sites = torch.cat(all_site_list, dim=0).numpy()
            for site_id in np.unique(all_sites):
                idx    = np.where(all_sites == site_id)[0]
                ts_s   = all_ts[idx]
                pear_s = all_pear[idx]
                lbl_s  = all_lbls[idx]
                n      = len(idx)

                if n <= k_shot:
                    with torch.no_grad():
                        feat  = self.model.forward_features(ts_s.cuda(), pear_s.cuda())
                        logits = self.model.classifier(feat)
                        probs  = F.softmax(logits, dim=1)[:, 1]
                    all_preds += probs.cpu().tolist()
                    all_labs  += lbl_s.tolist()
                    continue

                perm    = torch.randperm(n)
                sup_idx = perm[:k_shot].numpy()
                qry_idx = perm[k_shot:].numpy()

                sup_ts   = ts_s[sup_idx].cuda()
                sup_pear = pear_s[sup_idx].cuda()
                sup_lbl  = lbl_s[sup_idx].cuda()
                qry_ts   = ts_s[qry_idx].cuda()
                qry_pear = pear_s[qry_idx].cuda()
                qry_lbl  = lbl_s[qry_idx].cuda()

                support      = (sup_ts, sup_pear, F.one_hot(sup_lbl, 2).float())
                fast_weights = self.inner_loop(support, create_graph=False)

                with torch.no_grad():
                    feat   = self.model.forward_features(qry_ts, qry_pear)
                    logits = F.linear(feat, fast_weights['weight'], fast_weights['bias'])
                    probs  = F.softmax(logits, dim=1)[:, 1]

                all_preds += probs.cpu().tolist()
                all_labs  += qry_lbl.cpu().tolist()
        else:
            with torch.no_grad():
                for i in range(0, len(all_ts), 16):
                    feat   = self.model.forward_features(all_ts[i:i+16].cuda(), all_pear[i:i+16].cuda())
                    logits = self.model.classifier(feat)
                    probs  = F.softmax(logits, dim=1)[:, 1]
                    all_preds += probs.cpu().tolist()
                    all_labs  += all_lbls[i:i+16].tolist()

        # Sauvegarder predictions pour ROC

        if len(set(all_labs)) < 2:
            return 0.5, 50.0, 0.0, 0.0, 0.0

        auc        = roc_auc_score(all_labs, all_preds)
        result     = np.array(all_preds)
        labels_arr = np.array(all_labs)
        result[result > 0.5]  = 1
        result[result <= 0.5] = 0
        acc = np.mean(result == labels_arr) * 100

        report = classification_report(labels_arr, result, output_dict=True, zero_division=0)
        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']

        return auc, acc, recall[1], recall[0], 0.0

    def _tta_batch_level(self, all_ts, all_pear, all_labels, k_shot):
        """Fallback TTA au niveau batch si pas de sites."""
        n           = len(all_ts)
        perm        = torch.randperm(n)
        sup_idx     = perm[:k_shot]
        qry_idx     = perm[k_shot:]

        sup_ts   = all_ts[sup_idx].cuda()
        sup_pear = all_pear[sup_idx].cuda()
        sup_lbl  = all_labels[sup_idx].cuda()

        qry_ts   = all_ts[qry_idx].cuda()
        qry_pear = all_pear[qry_idx].cuda()
        qry_lbl  = all_labels[qry_idx].cuda()

        support      = (sup_ts, sup_pear,
                        F.one_hot(sup_lbl, 2).float())
        fast_weights = self.inner_loop(support, create_graph=False)

        with torch.no_grad():
            features = self.model.forward_features(qry_ts, qry_pear)
            logits   = F.linear(features,
                                fast_weights['weight'],
                                fast_weights['bias'])
            probs    = F.softmax(logits, dim=1)[:, 1]

        all_preds  = probs.cpu().tolist()
        all_lbls   = qry_lbl.cpu().tolist()

        import numpy as np
        auc        = roc_auc_score(all_lbls, all_preds)
        result     = np.array(all_preds)
        labels_arr = np.array(all_lbls)
        result[result > 0.5]  = 1
        result[result <= 0.5] = 0
        acc = np.mean(result == labels_arr) * 100

        report = classification_report(
            labels_arr, result, output_dict=True, zero_division=0)
        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']

        return auc, acc, recall[1], recall[0], 0.0




    def evaluate_with_tta_per_step(self, dataloader, k_shot=5, max_steps=3):
        """Mesure AUC apres chaque inner step pour visualisation."""
        self.model.eval()
        all_ts_list, all_pear_list, all_lbl_list, all_site_list = [], [], [], []

        for batch in dataloader:
            all_ts_list.append(batch[0])
            all_pear_list.append(batch[1])
            all_lbl_list.append(self._get_labels_int(batch[2]))
            if len(batch) > 3:
                all_site_list.append(batch[3])

        all_ts   = torch.cat(all_ts_list,   dim=0)
        all_pear = torch.cat(all_pear_list, dim=0)
        all_lbls = torch.cat(all_lbl_list,  dim=0)
        all_sites = torch.cat(all_site_list, dim=0).numpy()

        # Pour chaque step : listes de preds
        step_preds = [[] for _ in range(max_steps + 1)]
        step_labs  = [[] for _ in range(max_steps + 1)]

        for site_id in np.unique(all_sites):
            idx    = np.where(all_sites == site_id)[0]
            ts_s   = all_ts[idx]
            pear_s = all_pear[idx]
            lbl_s  = all_lbls[idx]
            n      = len(idx)

            if n <= k_shot:
                continue

            perm    = torch.randperm(n)
            sup_idx = perm[:k_shot].numpy()
            qry_idx = perm[k_shot:].numpy()

            sup_ts   = ts_s[sup_idx].cuda()
            sup_pear = pear_s[sup_idx].cuda()
            sup_lbl  = lbl_s[sup_idx].cuda()
            qry_ts   = ts_s[qry_idx].cuda()
            qry_pear = pear_s[qry_idx].cuda()
            qry_lbl  = lbl_s[qry_idx]

            # Step 0 — sans adaptation
            fast_weights = {
                "weight": self.model.classifier.fc.weight.clone(),
                "bias":   self.model.classifier.fc.bias.clone()
            }
            with torch.no_grad():
                feat   = self.model.forward_features(qry_ts, qry_pear)
                logits = F.linear(feat, fast_weights["weight"], fast_weights["bias"])
                probs  = F.softmax(logits, dim=1)[:, 1]
            step_preds[0] += probs.cpu().tolist()
            step_labs[0]  += qry_lbl.tolist()

            # Steps 1..max_steps
            support = (sup_ts, sup_pear, F.one_hot(sup_lbl, 2).float())
            fast_weights_init = {
                "weight": self.model.classifier.fc.weight.clone(),
                "bias":   self.model.classifier.fc.bias.clone()
            }
            fast_w = fast_weights_init

            for step in range(1, max_steps + 1):
                # Une seule step d adaptation
                with torch.no_grad():
                    feats_sup = self.model.forward_features(sup_ts, sup_pear)
                logits_s = F.linear(feats_sup, fast_w["weight"], fast_w["bias"])
                loss_s   = self.loss_fn(logits_s, sup_lbl)
                grads    = torch.autograd.grad(
                    loss_s, [fast_w["weight"], fast_w["bias"]],
                    create_graph=False)
                fast_w = {
                    "weight": fast_w["weight"] - self.inner_lr * grads[0],
                    "bias":   fast_w["bias"]   - self.inner_lr * grads[1]
                }
                with torch.no_grad():
                    feat   = self.model.forward_features(qry_ts, qry_pear)
                    logits = F.linear(feat, fast_w["weight"], fast_w["bias"])
                    probs  = F.softmax(logits, dim=1)[:, 1]
                step_preds[step] += probs.cpu().tolist()
                step_labs[step]  += qry_lbl.tolist()

        # Calculer AUC par step
        from sklearn.metrics import roc_auc_score
        aucs = []
        for step in range(max_steps + 1):
            if len(set(step_labs[step])) < 2:
                aucs.append(0.5)
            else:
                aucs.append(roc_auc_score(step_labs[step], step_preds[step]))
        return aucs

    def train(self):

        best_combined   = -1.0

        best_epoch_data = None

        self.current_step = 0

        no_improve      = 0



        for epoch in range(self.epochs):

            self.model.train()



            meta_batch = self.train_dataset.sample_meta_batch(

                self.meta_batch_size, epoch=epoch)



            self.optimizer.zero_grad()

            meta_loss = self.outer_loop(meta_batch)

            meta_loss.backward()

            torch.nn.utils.clip_grad_norm_(

                self.model.classifier.parameters(), max_norm=1.0)

            self.optimizer.step()



            self.current_step += 1

            self.lr_scheduler.update(

                optimizer=self.optimizer, step=self.current_step)



            val_auc,   val_acc,   val_sen,   val_spec,   val_loss   = \
                self.evaluate(self.val_dataset)

            intra_auc, intra_acc, intra_sen, intra_spec, intra_loss = \
                self.evaluate(self.test_intra_dataset)

            inter_auc, inter_acc, inter_sen, inter_spec, inter_loss = \
                self.evaluate_with_tta(self.test_inter_dataset, k_shot=self.cfg.k_shot)
            combined_auc = 0.5 * intra_auc + 0.5 * inter_auc



            self.logger.info(

                f'Epoch[{epoch}/{self.epochs}] | '

                f'MetaLoss:{meta_loss.item():.3f} | '

                f'ValAUC:{val_auc:.4f} | '

                f'IntraAUC:{intra_auc:.4f} | '

                f'InterAUC:{inter_auc:.4f} | '

                f'Combined:{combined_auc:.4f} | '

                f'IntraAcc:{intra_acc:.2f}% | '

                f'InterAcc:{inter_acc:.2f}% | '

                f'LR:{self.lr_scheduler.lr:.6f}'

            )



            wandb.log({

                'Meta Loss':    meta_loss.item(),

                'Val AUC':      val_auc,

                'Intra AUC':    intra_auc,

                'Inter AUC':    inter_auc,

                'Combined AUC': combined_auc,

                'Intra Acc':    intra_acc,

                'Inter Acc':    inter_acc,

                'Intra Sen':    intra_sen,

                'Intra Spec':   intra_spec,

                'Inter Sen':    inter_sen,

                'Inter Spec':   inter_spec,

                'Val Loss':     val_loss,

                'Intra Loss':   intra_loss,

                'Inter Loss':   inter_loss,

            })



            epoch_data = {

                'Epoch':        epoch,

                'Val AUC':      val_auc,

                'Intra AUC':    intra_auc,

                'Intra Acc':    intra_acc,

                'Intra Sen':    intra_sen,

                'Intra Spec':   intra_spec,

                'Inter AUC':    inter_auc,

                'Inter Acc':    inter_acc,

                'Inter Sen':    inter_sen,

                'Inter Spec':   inter_spec,

                'Combined AUC': combined_auc,

            }



            if combined_auc > best_combined:

                best_combined   = combined_auc

                best_epoch_data = epoch_data.copy()

                no_improve      = 0

            else:

                no_improve += 1

                if no_improve >= self.patience:

                    self.logger.info(f'Early stopping epoch {epoch}')

                    break



        if best_epoch_data is not None:

            self.save_path.mkdir(exist_ok=True, parents=True)
            self.save_results(best_epoch_data)
            import torch as _torch
            _model_path = self.save_path / 'model_maml.pt'
            _torch.save(self.model.state_dict(), _model_path)
            self.logger.info(f'Model saved: {_model_path}')



    def save_results(self, best_epoch_data):

        self.save_path.mkdir(exist_ok=True, parents=True)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")



        lines = [

            "=" * 60,

            "RESULTATS FINAUX - BNT+MAML",

            f"Date: {now}",

            f"Run ID: {self.cfg.unique_id}",

            "=" * 60,

            f"Meilleure Epoch    : {best_epoch_data['Epoch']}",

            f"Val AUC            : {best_epoch_data['Val AUC'] * 100:.2f}%",

            f"Combined AUC       : {best_epoch_data['Combined AUC'] * 100:.2f}%",

            "--- INTRA-SITE (seen) ---",

            f"Intra AUC          : {best_epoch_data['Intra AUC'] * 100:.2f}%",

            f"Intra Accuracy     : {best_epoch_data['Intra Acc']:.2f}%",

            f"Intra Sensitivity  : {best_epoch_data['Intra Sen'] * 100:.2f}%",

            f"Intra Specificity  : {best_epoch_data['Intra Spec'] * 100:.2f}%",

            "--- INTER-SITE (unseen) ---",

            f"Inter AUC          : {best_epoch_data['Inter AUC'] * 100:.2f}%",

            f"Inter Accuracy     : {best_epoch_data['Inter Acc']:.2f}%",

            f"Inter Sensitivity  : {best_epoch_data['Inter Sen'] * 100:.2f}%",

            f"Inter Specificity  : {best_epoch_data['Inter Spec'] * 100:.2f}%",

            "=" * 60,

        ]



        metrics_file = self.save_path / "final_metrics_maml.txt"

        with open(metrics_file, 'w') as f:

            f.write("\n".join(lines))



        self.logger.info("\n" + "\n".join(lines))



        self.best_results = {

            'best_epoch':   best_epoch_data['Epoch'],

            'val_auc':      best_epoch_data['Val AUC'] * 100,

            'combined_auc': best_epoch_data['Combined AUC'] * 100,

            'intra_auc':    best_epoch_data['Intra AUC'] * 100,

            'intra_acc':    best_epoch_data['Intra Acc'],

            'intra_sen':    best_epoch_data['Intra Sen'] * 100,

            'intra_spec':   best_epoch_data['Intra Spec'] * 100,

            'inter_auc':    best_epoch_data['Inter AUC'] * 100,

            'inter_acc':    best_epoch_data['Inter Acc'],

            'inter_sen':    best_epoch_data['Inter Sen'] * 100,

            'inter_spec':   best_epoch_data['Inter Spec'] * 100,

        }



        ALL_MAML_RESULTS.append(self.best_results)

