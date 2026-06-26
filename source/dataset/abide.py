
import numpy as np

import torch

from .preprocess import StandardScaler

from omegaconf import DictConfig, open_dict



# Sites sources fixes — comme Lee et al.

SOURCE_SITES = ['NYU', 'UM_1', 'USM', 'UCLA_1']



def load_abide_data(cfg: DictConfig):

    data = np.load(cfg.dataset.path, allow_pickle=True).item()

    final_timeseires = data["timeseires"]

    final_pearson    = data["corr"]

    labels           = data["label"]

    site             = data['site']



    # Filtre source-only si demande

    source_only = cfg.dataset.get('source_only', False)

    if source_only:

        mask = np.isin(site, SOURCE_SITES)

        final_timeseires = final_timeseires[mask]

        final_pearson    = final_pearson[mask]

        labels           = labels[mask]

        site             = site[mask]

        print(f"[source_only] Filtre actif : {mask.sum()} sujets ({SOURCE_SITES})")



    scaler = StandardScaler(

        mean=np.mean(final_timeseires),

        std=np.std(final_timeseires)

    )

    final_timeseires = scaler.transform(final_timeseires)



    final_timeseires, final_pearson, labels = [

        torch.from_numpy(d).float()

        for d in (final_timeseires, final_pearson, labels)

    ]



    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]

        cfg.dataset.timeseries_sz = final_timeseires.shape[2]



    return final_timeseires, final_pearson, labels, site

