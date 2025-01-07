# @Author: CAG
# @Github: CAgAG
# @Encode: UTF-8
# @FileName: model.py

import random

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from base import BaseDetector
from feat_aug import data_augmentation_by_channel
from basic_model import FIAD_Base
from pygod.utils import validate_device


class FIAD(BaseDetector):
    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.relu,
                 alpha=None,
                 beta=.5,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=5,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 f=10):
        super(FIAD, self).__init__(contamination=contamination)

        # model param
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha
        self.beta = beta

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)
        self.batch_size = batch_size
        self.num_neigh = num_neigh

        # other param
        self.f = f
        self.model = None

        self.loss_theta = random.choice([10., 40., 90.])  # struct
        self.loss_eta = random.choice([3., 5., 8.])  # attr

        # loss
        self.loss_func = self.loss_func_ade

    def fit(self, G, y_true=None):
        """
        Fit detector
        """
        G.node_idx = torch.arange(G.x.shape[0])
        self.num_nodes, self.node_feat_dim = G.x.shape
        if self.hid_dim > self.num_nodes:
            self.hid_dim = self.num_nodes

        if self.batch_size == 0:
            self.batch_size = G.x.shape[0]

        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        self.model = FIAD_Base(in_dim=G.x.shape[1],
                               hid_dim=self.hid_dim,
                               num_layers=self.num_layers,
                               dropout=self.dropout,
                               act=self.act,
                               num_nodes=self.num_nodes).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        self.model.train()
        decision_scores = np.zeros(G.x.shape[0])
        for _ in tqdm(range(self.epoch)):
            epoch_loss = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.node_idx

                # source data
                x, s, edge_index = self.process_graph(sampled_data)  # node feat, adj, edge index

                # encode
                h = self.model.embed(x, edge_index)

                # Anomaly Injection (Feature)
                h_ = h
                h_aug = data_augmentation_by_channel(h, rate=1.0, scale_factor=self.f)

                # reconstruct
                x_, s_ = self.model.reconstruct(h_, edge_index)
                x_aug, s_aug = self.model.reconstruct(h_aug, edge_index)

                # loss compute
                score = self.loss_func(x[:batch_size],
                                       x_[:batch_size],
                                       s[:batch_size],
                                       s_[:batch_size])
                score_aug = self.loss_func(x[:batch_size],
                                           x_aug[:batch_size],
                                           s[:batch_size],
                                           s_aug[:batch_size])
                # feature loss
                feat_loss = self.feat_loss_func(h_, h_aug)
                loss = self.beta * ((torch.mean(score) + torch.mean(score_aug)) / 2) + \
                       (1 - self.beta) * torch.mean(feat_loss)

                score = score.detach().cpu().numpy()
                if not np.isnan(score).any():
                    decision_scores[node_idx[:batch_size]] = score
                epoch_loss += loss.item() * batch_size

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.decision_scores_ = decision_scores
        self._process_decision_scores()
        return self

    def decision_function(self, G):
        """
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.
        """
        check_is_fitted(self, ['model'])
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]

        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        self.model.eval()
        outlier_scores = np.zeros(G.x.shape[0])
        for sampled_data in loader:
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.node_idx

            x, s, edge_index = self.process_graph(sampled_data)

            x_, s_ = self.model(x, edge_index)
            score = self.loss_func(x[:batch_size],
                                   x_[:batch_size],
                                   s[:batch_size],
                                   s_[:batch_size])

            outlier_scores[node_idx[:batch_size]] = score.detach() \
                .cpu().numpy()
        return outlier_scores

    def process_graph(self, G):
        """
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.
        """
        s = to_dense_adj(G.edge_index)[0].to(self.device)
        edge_index = G.edge_index.to(self.device)
        x = G.x.to(self.device)
        return x, s, edge_index

    def loss_func_ade(self, x, x_, s, s_):
        # generate hyperparameter - structure penalty
        reversed_adj = 1 - s
        thetas = torch.where(reversed_adj > 0, reversed_adj,
                             torch.full(s.shape, self.loss_theta).to(self.device))

        # generate hyperparameter - node penalty
        reversed_attr = 1 - x
        etas = torch.where(reversed_attr == 1, reversed_attr,
                           torch.full(x.shape, self.loss_eta).to(self.device))  # 相比 Dominant_19不同的地方

        # attribute reconstruction loss
        diff_attribute = torch.pow(x - x_, 2) * etas
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        # structure reconstruction loss
        diff_structure = torch.pow(s - s_, 2) * thetas
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = self.alpha * attribute_errors \
                + (1 - self.alpha) * structure_errors
        return score

    def feat_loss_func(self, feat_1, feat_2):
        diff_ = torch.pow(feat_1 - feat_2, 2)
        errors = torch.sqrt(torch.sum(diff_, 1))
        return errors

    def save(self, path_dir: str, info: list):
        if path_dir[-1] != '/':
            path_dir += '/'
        unique_str = '-'.join(info)

        torch.save(self.model.state_dict(), "{}{}.pth".format(path_dir, unique_str))

    def load(self, path_dir: str, info: list):
        if path_dir[-1] != '/':
            path_dir += '/'
        unique_str = '-'.join(info)

        self.model.load_state_dict(torch.load("{}{}.pth".format(path_dir, unique_str)))
