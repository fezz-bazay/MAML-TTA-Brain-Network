import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from .base import BaseModel

class MLPBrain(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()
        input_dim = config.dataset.node_sz * config.dataset.node_sz
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 800),
            nn.BatchNorm1d(800),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(800, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Linear(256, 2)

    def forward_features(self, time_series, node_feature):
        bz = node_feature.shape[0]
        x  = node_feature.view(bz, -1)
        return self.backbone(x)

    def forward(self, time_series, node_feature):
        return self.classifier(self.forward_features(time_series, node_feature))
