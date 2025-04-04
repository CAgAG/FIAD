# FIAD: Graph anomaly detection framework based feature injection 
You can check the paper here [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S0957417424020839).

# Environment Setup
Before you start, install Pytorch and torch-geometric with appropriate CUDA support. Please refer to the PyTorch and torch-geometric websites for the specific guidelines.

My environment is as follows:
```bash
torch==1.13.0
torchaudio==0.13.0
torchvision==0.14.0
torch-cluster==1.6.1
torch-scatter==2.1.1
torch-sparse==0.6.17
torch-spline-conv==1.2.2
torch-geometric==2.3.1

```

Install other dependencies:

```bash
pip install -r requirements.txt
```

# Dataset

- books, weibo, and reddit datasets. please refer to [here](https://github.com/pygod-team/data).

- cora, citeseer, and pubmed datasets. please refer to [here](https://github.com/TrustAGI-Lab/CoLA).

- ogbn-arxiv dataset. please refer to [here](https://ogb.stanford.edu/docs/nodeprop/).

# Experiments
Parameters:

- dataset: dataset name.
- dataset_dir: the directory where the dataset is located.
- batch_size: batch size of graphs. 0 for total graph.

Example:
```bash
python main.py --dataset weibo --dataset_dir ./data/ --batch_size 0
```

You can further modify the parameters in the paper within the `main.py` file.
```python
hid_dim = [...]         # Embedding dimension: ℎ
lr = [...]              # Learning Rate
injection_rate = [...]  # Proportion of injection: 𝑝
alpha = [...]           # Proportion between attribute and structure: 𝛼
beta = [...]            # Proportion between two losses: 𝛽
```
> The `log` folder contains a portion of the training model records.

# Citing FIAD:
```bash
@article{FIAD2025CHEN,
	title = {FIAD: Graph Anomaly Detection Framework Based Feature Injection},
	author = {Chen, Aoge and Wu, Jianshe and Zhang, Hongtao},
	year = {2025},
	journal = {Expert Systems with Applications},
	volume = {259},
	pages = {125216},
	doi = {10.1016/j.eswa.2024.125216}
}
```
