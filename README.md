## freqdiscriminet

Code to accompany the manuscript "Models optimized for real-world tasks reveal the necessity of precise temporal coding in hearing" by Mark R. Saddler and Josh H. McDermott (2024). This repository contains the code, models, and analyses for our comparisons between deep neural network and ideal observer models for pure tone frequency discrimination.

## Dependencies

This is a repository of Python (3.11.7) code. A complete list of Python dependencies is contained in [`requirements.txt`](requirements.txt). The models were developed in `torch-2.2.1` on machines running CentOS Linux 7.

## Model Weights

Trained model weights and and raw model evaluation outputs for each model configuration are too large to include here, but can be downloaded from our [Google Drive](https://drive.google.com/drive/folders/1YgC7x6Ot84XZInlSyHK-9NQ0jhhGUS2z?usp=share_link). The file structure of the Google Drive (which mirrors this repository) should be preserved for code to run without altering file paths.

## Contents

The [`figures_schematics.ipynb`](figures_schematics.ipynb) and [`figures_results.ipynb`](figures_results.ipynb) Jupyter Notebooks generate all figures shown in the paper and give a minimal example of how to run our deep neural network models. Code for training and evaluating the models at scale is located in [`run_model.py`](run_model.py), which is called from the example SLURM script [`run_model_job.sh`](run_model_job.sh). Code to generate stimui and build the model can be found in:
- [`pure_tone_dataset.py`](pure_tone_dataset.py) (stimulus generation for optimization / evaluation)
- [`peripheral_model.py`](peripheral_model.py) (PyTorch implementation of the auditory nerve model)
- [`perceptual_model.py`](perceptual_model.py) (Build feedforward convolutional neural network model)
- [`util_torch.py`](util_torch.py) (utility functions for PyTorch models)
The [`util.py`](util.py) file contains helpful functions for analyzing and plotting results (i.e., inferring frequency discrimination thresholds from model evaluation outputs).

## Notes

To directly compare our deep neural network models with ideal observers, we strived to simulate the frequency discrimination experiment from [Heinz et al. (2001, Neural Computation)](https://doi.org/10.1162/089976601750541804) as closely as we could. We ported their [auditory nerve model](https://modeldb.science/37436) to PyTorch and trained deep neural networks to make pure tone frequency discrimination judgments (with stimulus parameters matched as closely as possible to the original paper). Frequency discrimination thresholds for the Siebert (1970) and Heinz et al. (2001) ideal observer models were scanned from Fig. 4a in the original paper.

## Contact
Mark R. Saddler (msaddler@mit.edu)
