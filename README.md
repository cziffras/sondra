# Sondra – Semantic Segmentation for SAR Imagery

Sondra is a student project conducted by three students from CentraleSupélec: Emmanuel Benichou, Rodolphe Durand, Lazare Plisson Arcos during supervised by Jérémy Fix (teacher on CentraleSupélec Metz Campus). This project is aiming to develop Deep Learning architecture compatible with SAR data in order to perform semantic segmentation on famous SAR imaging datasets. This repo focuses on Polarimetric San Francisco dataset that can be found on this [link](https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/).

---

## Features

- Implementation of state-of-the-art segmentation models: **UNet** and **SegFormer**
- Integrated support for SAR data formats
- Seamless training pipeline using `torchcvnn` (find [here](https://torchcvnn.github.io/torchcvnn/))
- Experiment tracking via **Weights & Biases (WandB)**

## Running an Experiment

To launch a training run, navigate to the project root and execute:

```bash
cd ~/sondra
python -m src.torchtmpl.main configs/baseline_unet.yaml train
```

You can modify the configuration by editing or switching files in the `configs/` directory.

## Installation

Set up your environment by creating and activating a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Then install the required Python packages:

```bash
pip install -r requirements.txt
```

> The `requirements.txt` file was generated using:
> 
> ```bash
> pip list --format=freeze > requirements.txt
> ```

## Result Visualization

All training metrics and outputs are logged using **Weights & Biases (WandB)**. Once logged in, you can monitor:

- Confusion matrices
- Predicted segmentation masks
- Training and validation metrics
- And soon more !

To enable logging:

```bash
wandb login
```

You can then view your runs at [wandb.ai](https://wandb.ai).

## Repository Structure

```bash
sondra/
├── configs/              # YAML config files for experiments
├── src/torchtmpl/        # Core implementation (datasets, models, training, etc.)
├── logs/                 # logging outputs
└── piprequirements.txt   # Dependency list compatible with pip
```

---

For questions, improvements, or contributions, feel free to open an issue or pull request.

Happy segmenting!

