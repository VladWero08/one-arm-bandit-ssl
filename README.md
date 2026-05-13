# one-arm-bandit-ssl

This repository implements the reinforcement-learning-guided semi-supervised learning (RLGSSL) algorithm from the NeurIPS 2024 paper [*Reinforcement Learning Guided Semi-Supervised Learning: A review*](https://proceedings.neurips.cc/paper_files/paper/2024/file/f7a7bb369e48f10e85fce85b67d8c516-Paper-Conference.pdf). The method advances SSL evolution by using an RL policy to dynamically label unlabeled data, outperforming baselines like FixMatch and Meta Pseudo-Labels on CIFAR-10/100 and SVHN with limited labels.

## Key Components

- **Teacher-Student Framework**: EMA-updated teacher generates consistent pseudo-labels under augmentations.
- **Mixup Augmentation**: Linear interpolation in latent space for regularization, as in MixMatch/ReMixMatch.

## RL Loss

Our RL loss adapts the original RLGSSL policy to balance **exploration** and **exploitation** in pseudo-labeling. The reward function combines prediction accuracy with an **entropy bonus**:

<img width="824" height="64" alt="image" src="https://github.com/user-attachments/assets/a7ef9090-a355-4219-a2dd-f68af6c14604" />

which is more detailed in the implementation and documentation of the project.
