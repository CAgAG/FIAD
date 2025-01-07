# @Author: CAG
# @Github: CAgAG
# @Encode: UTF-8
# @FileName: basic_model.py

import torch
import torch.nn as nn

from basic_nn import GCN


class FIAD_Base(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_layers,
                 dropout,
                 act,
                 num_nodes):
        super(FIAD_Base, self).__init__()

        decoder_layers = int(num_layers / 2)
        encoder_layers = num_layers - decoder_layers

        self.shared_encoder = GCN(in_channels=in_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=encoder_layers,
                                  out_channels=hid_dim,
                                  dropout=dropout,
                                  act=act)

        self.attr_decoder = nn.Sequential(
            nn.Linear(num_nodes, hid_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hid_dim, in_dim),
            nn.Dropout(dropout)
        )
        self.struct_decoder = GCN(in_channels=hid_dim,
                                  hidden_channels=hid_dim,
                                  num_layers=decoder_layers,
                                  out_channels=hid_dim,
                                  dropout=dropout,
                                  act=act)

    def embed(self, x, edge_index):
        h = self.shared_encoder(x, edge_index)
        return h

    def reconstruct(self, h, edge_index):
        # decode attribute matrix
        x_ = self.attr_decoder(h.T)
        x_ = h @ x_

        # decode structure matrix
        h_ = self.struct_decoder(h, edge_index)
        s_ = torch.sigmoid(h_ @ h_.T)
        return x_, s_

    def forward(self, x, edge_index):
        # encode
        h = self.embed(x, edge_index)
        # reconstruct
        x_, s_ = self.reconstruct(h, edge_index)
        return x_, s_
