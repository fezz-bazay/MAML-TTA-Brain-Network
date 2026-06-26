import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel


class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num,
                 pooling=True, orthogonal=True, freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(
            d_model=input_feature_size, nhead=4,
            dim_feedforward=hidden_size,
            batch_first=True
        )

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
            )
            self.dec = DEC(
                cluster_number=output_node_num,
                hidden_dimension=input_feature_size,
                encoder=self.encoder,
                orthogonal=orthogonal,
                freeze_center=freeze_center,
                project_assignment=project_assignment
            )

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class BrainNetworkTransformer(BaseModel):

    def __init__(self, config: DictConfig):
        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(
                torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim),
                requires_grad=True
            )
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling

        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(
                    input_feature_size=forward_dim,
                    input_node_num=in_sizes[index],
                    hidden_size=1024,
                    output_node_num=size,
                    pooling=do_pooling[index],
                    orthogonal=config.model.orthogonal,
                    freeze_center=config.model.freeze_center,
                    project_assignment=config.model.project_assignment
                )
            )

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        # ✅ AJOUT 1: embed_dim requis par tous les modules avancés
        # 8 * sizes[-1] = dimension après dim_reduction + reshape
        self.embed_dim = 8 * sizes[-1]  # = 8 * 100 = 800 avec config par défaut

        self.fc = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    # ✅ AJOUT 2: forward_features — retourne les features AVANT fc
    # Requis par DisentangledBNT, AdversarialBNT, AttentionConsistentBNT, ImprovedMDBNT
    def forward_features(self,
                         time_seires: torch.tensor,
                         node_feature: torch.tensor) -> torch.tensor:
        """
        Extrait les features du BNT sans la couche de classification finale.
        Retourne un tenseur de shape [batch_size, embed_dim] = [batch_size, 800]
        """
        bz, _, _ = node_feature.shape

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []
        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape((bz, -1))

        # Retourne les features AVANT fc — shape: [batch_size, embed_dim]
        return node_feature

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor) -> torch.tensor:
        """
        Forward pass complet avec classification finale.
        Retourne logits de shape [batch_size, 2]
        """
        # ✅ Réutilise forward_features pour éviter la duplication de code
        features = self.forward_features(time_seires, node_feature)
        return self.fc(features)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        decs = list(filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
