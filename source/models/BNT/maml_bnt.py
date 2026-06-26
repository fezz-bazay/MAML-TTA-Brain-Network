# source/models/BNT/maml_bnt.py
# Architecture : BNT backbone (pre-entraine) + MAML Classifier Head
# Pas de modulation — trop instable avec 800 dim
# Contribution : BNT + MAML selectif sur classifier + protocole inter-site

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineLoss(nn.Module):
    """
    Loss de Lee et al. TNNLS 2023.
    cosine_embedding_loss + lambda * cross_entropy
    Evite l'overfitting sur petits datasets.
    """
    def __init__(self, xent=0.1):
        super().__init__()
        self.xent = xent

    def forward(self, logits, targets):
        # targets = LongTensor
        one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        y       = torch.ones(logits.size(0)).to(logits.device)
        cosine_loss = F.cosine_embedding_loss(
            logits, one_hot, y, reduction='mean')
        cent_loss = F.cross_entropy(
            F.normalize(logits, dim=1), targets, reduction='mean')
        return cosine_loss + self.xent * cent_loss


class MAMLClassifier(nn.Module):
    """
    Task network — equivalent de task_network de Lee et al.
    Linear(800, 2) — adapte par inner loop MAML par site.
    """
    def __init__(self, feature_dim=800, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MAMLBNT(nn.Module):
    """
    BNT backbone (pre-entraine) + MAML Classifier Head.

    Difference vs Lee et al. :
    - Lee : MLP(6670->8) + Modulation + Classifier
    - Nous : BNT Transformer(ts+pearson->800) + Classifier MAML

    Inner loop : adapte seulement MAMLClassifier (comme meta_train de Lee)
    Outer loop : met a jour BNT + Classifier
    """
    def __init__(self, backbone, feature_dim=800, num_classes=2):
        super().__init__()
        self.backbone   = backbone
        self.classifier = MAMLClassifier(feature_dim, num_classes)

    def forward(self, ts, pear):
        fF = self.backbone.forward_features(ts, pear)
        return self.classifier(fF)

    def forward_features(self, ts, pear):
        """Retourne features BNT [batch, 800]."""
        return self.backbone.forward_features(ts, pear)
