# CrossInfoGAN-MI

CrossInfoGAN-MI is a research codebase implementing variants of cross-generative models and tools for estimating mutual information. We show that existing MI estimators failed to accurately capture high-dimensional dependencies due to the inherent 'curse of dimensionality' and significant bias in finite-sample regimes. 


## Repository layout

- `model/` — core model implementations and helper modules (e.g., `aemi.py`, `bmi.py`, `info_vae.py`, `nn.py`, `utils.py`).
- `checkpoints/` — pretrained model weights (PyTorch `.pt` files).
- `*.ipynb` — example notebooks demonstrating training, evaluation, and MI estimation (e.g., `CrossInfoGAN-mnist.ipynb`, `mi_mnist.ipynb`).

## Requirements

This project was developed for Python 3.10 and uses common scientific and deep learning libraries. A minimal set of dependencies is provided in `requirements.txt`.

Recommended packages include (see `requirements.txt`):

- torch
- torchvision
- numpy
- scipy
- matplotlib
- jupyter
- tqdm

## Quick start

1. Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```


2. In the Jupyter UI open `CrossInfoGAN-mnist.ipynb` (or `mi_mnist.ipynb`) and run cells to reproduce experiment snippets.

Notes:
- Some notebooks expect the `data/` and `checkpoints/` folders to be present in the repository root (they are included here). If you want to use your own datasets, adapt the data-loading cells in the notebooks.
- Pretrained weights are PyTorch `.pt` files in `checkpoints/`; notebook cells show how to load them into the models in `model/`.




