# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo
from torch.utils.tensorboard import SummaryWriter

# Local imports
from . import models
from . import optim
from . import losses
from .data import get_dataloaders
from .utils import  training_contrastive_utils, training_utils


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if use_cuda:
        torch.cuda.empty_cache()

    logging.info(f"Using device : {device}")

    if config.get("contrastive", False):
        run_type = "ContrastivePretraining"
    elif "pretrained_weights" in config.get("model", {}):
        run_type = "SegmentationFromPretrained"
    else:
        run_type = "SegmentationBaseline(NoContrastive)"

    # Initialiser WandB
    wandb.init(
        project="segmentation-polsf",
        entity="SONDRA_2024-2025",
        config=config,
        name=run_type + "_" + models.__name__,
        tags=[run_type],
    )

    contrastive = config.get("contrastive", False)

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader, input_size, num_classes = get_dataloaders(
        data_config, use_cuda, contrastive
    )

    print(f"data input size : {input_size}, num_classes : {num_classes}")

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = losses.get_loss(config["loss"]["name"])

    # Build the optimizer
    logging.info("= Optimizer")
    optimizer = optim.get_optimizer(config["optimizer"], model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # On utilise la classe du modèle comme base pour le nom du log
    logname = model_config["class"]
    logdir = training_utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    tensorboard_writer = SummaryWriter(logdir)

    # Copie du fichier de config dans le dossier log
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Création d'un résumé de l'expérience (peut être désactivé avec verbose=False)
    verbose = config.get("verbose", True)

    if verbose:
        input_size = next(iter(train_loader))[0].shape
        summary_text = (
            f"Logdir : {logdir}\n"
            + "## Command \n"
            + " ".join(sys.argv)
            + "\n\n"
            + f" Config : {config} \n\n"
            + "## Summary of the model architecture\n"
            + f"{torchinfo.summary(model, input_size=input_size, dtypes=[torch.complex64] if bool(config['model'].get('is_complex', False)) else None)}\n\n"
            + "## Loss\n\n"
            + f"{loss}\n\n"
            + "## Datasets : \n"
            + f"Train : {train_loader.dataset}\n"
            + f"Validation : {valid_loader.dataset}"
        )
        with open(logdir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(summary_text)
        logging.info(summary_text)

    # Définition du callback d'early stopping
    model_checkpoint = training_utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    train_epoch_func = (
        training_contrastive_utils.train_one_contrastive_epoch
        if contrastive
        else training_utils.train_one_epoch
    )
    valid_func = training_utils.valid_contrastive_epoch if contrastive else training_utils.valid_epoch

    accumulation_steps = config["data"].get("accumulation_steps", 1)

    for e in range(config["nepochs"]):
        # Entraînement pour une époque
        train_metrics =  train_epoch_func(
            model, train_loader, loss, optimizer, device, accumulation_steps
        )
        train_loss = train_metrics["train_loss"]

        # Évaluation sur le set de validation
        valid_metrics =  valid_func(
            model, train_loader, loss, optimizer, device, accumulation_steps
        )
        valid_loss = valid_metrics["valid_loss"]

        if not contrastive:
            valid_accuracy = valid_metrics.get("valid_overall_accuracy", None)
            accuracy_msg = f", {valid_accuracy:.3f}%" if valid_accuracy is not None else ""
        else:
            accuracy_msg = ""

        updated = model_checkpoint.update(valid_loss)
        logging.info(
            "[%d/%d] Train loss : %.3f, Test loss : %.3f %s%s"
            % (
                e,
                config["nepochs"],
                train_loss,
                valid_loss,
                accuracy_msg,
                "[>> BETTER <<]" if updated else "",
            )
        )

        # Mise à jour des dashboards
        metrics = {"train_loss": train_loss, "valid_loss": valid_loss}

        if not contrastive:
            metrics["valid_overall_accuracy"] = valid_metrics["valid_overall_accuracy"]

        wandb.log(metrics)

        # Mise à jour du dashboard TensorBoard
        for key, value in metrics.items():
            tensorboard_writer.add_scalar(key, value, e)


def test(config):
    raise NotImplementedError("La fonction test n'est pas encore implémentée voir training_utils.")


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage: {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    config_file = sys.argv[1]
    command = sys.argv[2]

    logging.info(f"Loading config from {config_file}")
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier de config: {e}")
        sys.exit(-1)

    if command == "train":
        train(config)
    elif command == "test":
        test(config)
    else:
        logging.error(f"Commande inconnue: {command}")
        sys.exit(-1)


if __name__ == "__main__":
    main()
