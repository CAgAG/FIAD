# @Author: CAG
# @Github: CAgAG
# @Encode: UTF-8
# @FileName: feat_aug.py

import torch


def data_augmentation_by_channel(x, rate=0.2, scale_factor=10, device='cuda'):
    x_aug = x.clone()
    num_node = x_aug.shape[0]
    num_channel = x_aug.shape[1]
    prob = torch.rand(num_channel)

    n1_range = prob < rate / 3  # prob 小于 1/3
    n2_range = torch.logical_and(rate / 3 <= prob, prob < rate * 2 / 3)  # prob 在 [1/3, 2/3)
    n3_range = torch.logical_and(rate * 2 / 3 <= prob, prob < rate * 1.0)  # prob 在 [2/3, 1)

    # 置换, 打乱 x_aug[:, n1_range] 的通道顺序
    shuffle_mat = x_aug[:, n1_range]
    x_aug[:, n1_range] = shuffle_mat[:, torch.randperm(torch.sum(n1_range)).tolist()]

    # 值修改, 修改 x_aug[:, n2_range] 的通道值 * scale_factor
    x_aug[:, n2_range] = (x_aug[:, n2_range].T * scale_factor).T

    # 值修改, 修改 x_aug[:, n3_range] 的通道值 / scale_factor
    x_aug[:, n3_range] = (x_aug[:, n3_range].T / scale_factor).T

    x_aug = x_aug.to(device)
    return x_aug
