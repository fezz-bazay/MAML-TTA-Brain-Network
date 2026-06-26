# source/dataset/maml_dataloader.py
# Protocole inter-site inspire de Lee et al. TNNLS 2023
# Sources : NYU + UM_1 + USM + UCLA_1 (4 grands sites)
# Unseen  : tous les autres sites
# Seed dynamique par epoch pour diversifier les episodes

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Sites sources fixes — comme Lee et al.
SOURCE_SITES = ['NYU', 'UM_1', 'USM', 'UCLA_1']


class SiteTaskDataset:
    """
    Dataset MAML pour le training.
    Chaque site source = une task.
    Support equilibre : k_shot//2 ASD + k_shot//2 TD
    Query             : q_query sujets restants du meme site
    Seed dynamique    : varie par epoch pour diversifier les episodes
    """

    def __init__(self, time_series, pearson, labels,
                 sites, k_shot=8, q_query=10, seed=42):

        self.time_series = time_series
        self.pearson     = pearson
        self.labels      = labels
        self.sites       = sites
        self.k_shot      = k_shot
        self.q_query     = q_query
        self.seed        = seed

        self.site_ids  = np.unique(sites)
        self.site_data = self._organize_by_site()

        print(f"Sites valides pour MAML training : {list(self.site_data.keys())}")

    def _organize_by_site(self):
        site_data = {}
        k = self.k_shot // 2

        for site_id in self.site_ids:
            idx        = np.where(self.sites == site_id)[0]
            labels_int = self.labels[idx]
            if labels_int.dim() > 1:
                labels_int = labels_int.argmax(dim=1)
            labels_np = labels_int.numpy()

            idx_asd = idx[labels_np == 1]
            idx_td  = idx[labels_np == 0]

            if len(idx_asd) >= k + 2 and len(idx_td) >= k + 2:
                site_data[site_id] = {
                    'asd': idx_asd,
                    'td':  idx_td,
                    'all': idx
                }

        return site_data

    def sample_task(self, site_id, epoch=0):
        """
        Support equilibre k//2 ASD + k//2 TD.
        Query = sujets restants du site.
        Seed dynamique : self.seed + epoch * 100
        → diversifie les episodes a chaque epoch
        → reste reproductible si on relance avec meme seed
        """
        rng  = np.random.RandomState(self.seed + epoch * 100)
        data = self.site_data[site_id]
        k    = self.k_shot // 2

        idx_asd = data['asd'].copy()
        idx_td  = data['td'].copy()
        rng.shuffle(idx_asd)
        rng.shuffle(idx_td)

        sup_idx   = np.concatenate([idx_asd[:k], idx_td[:k]])
        sup_set   = set(sup_idx.tolist())
        query_idx = np.array([i for i in data['all'] if i not in sup_set])
        rng.shuffle(query_idx)
        query_idx = query_idx[:self.q_query]

        support = (
            self.time_series[sup_idx],
            self.pearson[sup_idx],
            self.labels[sup_idx]
        )
        query = (
            self.time_series[query_idx],
            self.pearson[query_idx],
            self.labels[query_idx]
        )
        return support, query

    def get_all_tasks(self):
        return list(self.site_data.keys())

    def sample_meta_batch(self, meta_batch_size, epoch=0):
        """
        Echantillonne meta_batch_size sites.
        Seed dynamique par epoch.
        """
        rng     = np.random.RandomState(self.seed + epoch * 100)
        tasks   = self.get_all_tasks()
        sampled = rng.choice(
            tasks,
            size=min(meta_batch_size, len(tasks)),
            replace=False
        )
        return [self.sample_task(s, epoch=epoch) for s in sampled]


def build_maml_datasets(cfg, time_series, pearson, labels, sites, seed=42, source_sites=None):
    """
    Protocole inter-site de Lee et al. adapte a BNT.

    Sources : NYU + UM_1 + USM + UCLA_1 (4 grands sites = 424 sujets)
    Split   : 70% train / 15% val intra / 15% test intra
    Unseen  : 15 autres sites → test inter-site (585 sujets)

    Retourne :
        train_ds          : SiteTaskDataset pour MAML training
        val_loader        : DataLoader val intra-site
        test_intra_loader : DataLoader test intra-site
        test_inter_loader : DataLoader test inter-site (sites unseen)
    """
    k_shot     = cfg.get('k_shot',  8)
    q_query    = cfg.get('q_query', 10)
    batch_size = cfg.dataset.batch_size

    # Convertir sites en numpy
    if hasattr(sites, 'numpy'):
        sites_np = sites.numpy()
    else:
        sites_np = np.array(sites)

    # Afficher les sites disponibles
    unique_sites, counts = np.unique(sites_np, return_counts=True)
    print("\nSites disponibles dans ABIDE :")
    _src_sites = source_sites if source_sites is not None else SOURCE_SITES
    for s, c in sorted(zip(unique_sites, counts), key=lambda x: -x[1]):
        marker = " <- SOURCE" if s in _src_sites else ""
        print(f"  Site {s:>12} : {c:4d} sujets{marker}")

    # Masques source / inter
    source_mask = np.isin(sites_np, _src_sites)
    inter_mask  = ~source_mask

    print(f"\nSujets sources  : {source_mask.sum()}")
    print(f"Sujets inter    : {inter_mask.sum()}")

    # Donnees sources
    src_ts    = time_series[source_mask]
    src_pear  = pearson[source_mask]
    src_lbls  = labels[source_mask]
    src_sites = sites_np[source_mask]

    # Donnees inter-site (unseen)
    int_ts   = time_series[inter_mask]
    int_pear = pearson[inter_mask]
    int_lbls = labels[inter_mask]

    # Labels int pour stratification
    src_lbl_int = src_lbls.argmax(dim=1).numpy() \
        if src_lbls.dim() > 1 else src_lbls.numpy().astype(int)

    # Split sources : 70% train / 15% val / 15% test intra
    src_idx = np.arange(len(src_ts))

    train_idx, valtest_idx = train_test_split(
        src_idx, test_size=0.30,
        stratify=src_lbl_int,
        random_state=seed
    )
    val_idx, test_intra_idx = train_test_split(
        valtest_idx, test_size=0.50,
        stratify=src_lbl_int[valtest_idx],
        random_state=seed
    )

    print(f"\nSplit sources :")
    print(f"  Train MAML   : {len(train_idx)} sujets")
    print(f"  Val intra    : {len(val_idx)} sujets")
    print(f"  Test intra   : {len(test_intra_idx)} sujets")
    print(f"  Test inter   : {len(int_ts)} sujets (sites unseen)\n")

    # SiteTaskDataset pour MAML training
    train_ds = SiteTaskDataset(
        src_ts[train_idx],
        src_pear[train_idx],
        src_lbls[train_idx],
        src_sites[train_idx],
        k_shot=k_shot,
        q_query=q_query,
        seed=seed
    )

    # DataLoaders standard pour evaluation
    val_loader = DataLoader(
        TensorDataset(
            src_ts[val_idx],
            src_pear[val_idx],
            src_lbls[val_idx]
        ),
        batch_size=batch_size, shuffle=False
    )

    test_intra_loader = DataLoader(
        TensorDataset(
            src_ts[test_intra_idx],
            src_pear[test_intra_idx],
            src_lbls[test_intra_idx]
        ),
        batch_size=batch_size, shuffle=False
    )

    int_sites_np = sites_np[inter_mask]
    unique_inter = np.unique(int_sites_np)
    site2idx     = {s: i for i, s in enumerate(unique_inter)}
    int_sites_t  = torch.tensor([site2idx[s] for s in int_sites_np],
                                 dtype=torch.long)

    test_inter_loader = DataLoader(
        TensorDataset(int_ts, int_pear, int_lbls, int_sites_t),
        batch_size=batch_size, shuffle=False
    )

    return train_ds, val_loader, test_intra_loader, test_inter_loader
