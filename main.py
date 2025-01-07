# @Author  : CAgAG
# @Version : 1.0
# @Function:

import os
import argparse
import itertools
from datetime import datetime
import warnings

import numpy as np
from sklearn.metrics import f1_score
from pygod.utils.utility import load_data
from pygod.metrics import eval_roc_auc

from model import FIAD
from utils import PrintToFile, set_top_k_to_one

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='weibo', required=False,
                    help="Default: weibo")
parser.add_argument("--dataset_dir", type=str, default='./data/', required=False,
                    help="Default: ./data/")
parser.add_argument("--batch_size", type=int, default=0, required=False,
                    help="0 for total graph"
                         "Default: 0")
parser.add_argument("--model_name", type=str, default='FIAD', required=False)  # Previous Project Name: ChannelAD
args = parser.parse_args()

# Parameters
hid_dim = [8, 16, 32, 64, 128]
dropout = [0, 0.1, 0.3]
lr = [0.1, 0.05, 0.01]
alpha = [0.8, 0.5, 0.2, 0.3, 0.7]
beta = [0.8, 0.5, 0.2, 0.3, 0.7]

weight_decay = 0.01
batch_size = 0
num_neigh = -1
epoch = 300
gpu = 0

dataset = args.dataset
dataset_dir = args.dataset_dir
model_name = args.model_name
log_path = "./log/train-{}_test.log".format(dataset)
best_auc = -1
save_dir = './models/{}/'.format(dataset)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
combinations = itertools.product(hid_dim, dropout, lr, alpha, beta)

if __name__ == '__main__':
    auc, F1 = [], []
    print(dataset)

    for trial_i in range(20):
        for comb in combinations:
            comb_hid_dim, comb_dropout, comb_lr, comb_alpha, comb_beta = comb
            model = FIAD(hid_dim=comb_hid_dim,
                         weight_decay=weight_decay,
                         dropout=comb_dropout,
                         lr=comb_lr,
                         epoch=epoch,
                         gpu=gpu,
                         alpha=comb_alpha,
                         beta=comb_beta,
                         batch_size=batch_size,
                         num_neigh=num_neigh)
            data = load_data(dataset, dataset_dir)

            model.fit(data)
            score = model.decision_scores_

            y = data.y.bool()
            k = sum(y)

            if np.isnan(score).any():
                warnings.warn('contains NaN, skip one trial.')
                continue

            # ROC-AUC
            auc.append(eval_roc_auc(y, score))
            if auc[-1] > best_auc:
                best_auc = auc[-1]
                best_auc = round(best_auc, ndigits=6)
                Now = datetime.now()
                model.save(path_dir=save_dir,
                           info=[str(best_auc), str(comb_hid_dim),
                                 str(Now.year), str(Now.month), str(Now.day), str(Now.hour)])

            # F1 Score
            src_y_true = data.y.numpy()
            src_pred = score
            anomaly_node_count = np.count_nonzero(src_y_true)
            y_true = src_y_true.astype(int)
            y_pred = set_top_k_to_one(src_pred, anomaly_node_count)
            cur_f1_score = f1_score(y_true=y_true, y_pred=y_pred)
            F1.append(cur_f1_score)

            with PrintToFile(log_path):
                print(model)
                print(auc)
                # print(cur_f1_score)
            print(model)
            print(trial_i, "==>", auc[-1], "|", best_auc)

    with PrintToFile(log_path):
        print(f'\ntrain {len(auc)} loop: ')
        print("AUC: {:.4f}±{:.4f} ({:.4f}) \n".format(np.mean(auc), np.std(auc), np.max(auc)))
        print("F1: {:.4f}±{:.4f} ({:.4f}) \n".format(np.mean(F1), np.std(F1), np.max(F1)))

    print(f'\ntrain {len(auc)} loop: ')
    print("AUC: {:.4f}±{:.4f} ({:.4f}) \n".format(np.mean(auc), np.std(auc), np.max(auc)))
    print("F1: {:.4f}±{:.4f} ({:.4f}) \n".format(np.mean(F1), np.std(F1), np.max(F1)))
