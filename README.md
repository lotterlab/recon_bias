# Bias in Medical Image Reconstruction 

This repository contains code to train models for bias evaluation and mitigation of medical image reconstruction models. It is split into multiple branches, each with a different purpose:

## Available Branches

- `ucsf`: Reconstruction and classification and segmentation for the UCSF-PDGM dataset
- `chexpert`: Reconstruction and classification for the UCSF-PDGM dataset
- `reweighting`: Reconstruction training for a mitigation technique based on reweighting of samples
- `eodd`: Reconstruction training for a mitigation technique based on an equalized constraint
- `adv`: Reconstruction training for a mitigation technique based on adversarial traing

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your/repo.git
2. Checkout the branch you need: 
    ```bash
    git checkout feature-x
3.	Follow the instructions in the branch’s README.md