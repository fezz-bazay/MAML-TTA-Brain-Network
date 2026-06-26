from .transformer import GraphTransformer
from .BNT import BrainNetworkTransformer
from .BNT.maml_bnt import MAMLBNT
from .mlp_brain import MLPBrain
from omegaconf import DictConfig


def model_factory(config: DictConfig):

    if config.model.name in ["LogisticRegression", "SVC"]:
        return None

    # ================================================================
    # CAS MAML — MAML-Head (Selective Parameter Adaptation)
    # ================================================================
    if config.get("enable_maml", False):
        backbone    = BrainNetworkTransformer(config).cuda()
        feature_dim = config.get("feature_dim", 800)
        model       = MAMLBNT(backbone, feature_dim=feature_dim)
        return model.cuda()

    # ================================================================
    # CAS STANDARD — BNT, BrainNetCNN, FBNETGEN, Transformer
    # ================================================================
    return eval(config.model.name)(config).cuda()
