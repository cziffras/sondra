# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib
import json

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
from .utils import (
    training_contrastive_utils, 
    training_utils, 
    log_confusion_matrix,
    check_model_params_validity,
    count_parameters,
    log_predictions_on_wandb,
)


def train(config, wandb_run, visualize):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if use_cuda:
        torch.cuda.empty_cache()

    logging.info(f"Using device : {device}")

    contrastive = config["model"].get("contrastive", False)

    logging.info(f"= Attempting a forward pass with given config")
    check_model_params_validity(config, use_cuda, contrastive)

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
    num_params = count_parameters(model)
    model.to(device)

    logging.info(f"= Model has {num_params} parameters")

    # Build the loss
    logging.info("= Loss")
    if contrastive:
        loss = losses.get_loss(config["loss"]["name"])
    else:
        loss = losses.get_loss(config["loss"]["name"], ignore_index=0)

    # Build the optimizer
    logging.info("= Optimizer")
    optimizer = optim.get_optimizer(config, model.parameters())

    # Build the scheduler
    logging.info("= Scheduler")
    steps_per_epoch = len(train_loader)
    scheduler = optim.get_scheduler(config, optimizer, steps_per_epoch)

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

    num_input_dims = len(input_size)

    # Définition du callback d'early stopping
    model_checkpoint = training_utils.ModelCheckpoint(
        model, 
        optimizer,
        str(logdir),
        num_input_dims,
        min_is_best=True
    )

    train_epoch_func = (
        training_contrastive_utils.train_one_contrastive_epoch
        if contrastive
        else training_utils.train_one_epoch
    )
    valid_func = (
        training_contrastive_utils.valid_contrastive_epoch 
        if contrastive 
        else training_utils.valid_epoch
    )

    for e in range(config["nepochs"]):
        # Entraînement pour une époque

        if contrastive: 
            train_metrics = train_epoch_func(
                model=model,
                loader=train_loader,
                f_loss=loss,
                optim=optimizer,
                scheduler=scheduler,
                device=device,
                epoch=e,
                loss_weights=config["model"].get("loss_weights", None)
            )
        
        else: 
            train_metrics =  train_epoch_func(
                model=model,
                loader=train_loader,
                f_loss=loss,
                optim=optimizer,
                scheduler=scheduler,
                device=device,
                number_classes=num_classes,
                epoch=e
        )

        train_loss = train_metrics["train_loss"]

        if contrastive: 
            valid_metrics =  valid_func(
                model=model,
                loader=valid_loader,
                f_loss=loss,
                device=device,
            )
        else: 
            valid_metrics =  valid_func(
                model=model,
                loader=valid_loader,
                f_loss=loss,
                device=device,
                number_classes=num_classes
            )
            
        valid_loss = valid_metrics["valid_loss"]

        if not contrastive:
            valid_accuracy = valid_metrics.get("valid_overall_accuracy", None)
            accuracy_msg = f", Accuracy : {valid_accuracy:.3f}% " if valid_accuracy is not None else ""
        else:
            accuracy_msg = ""

        updated = model_checkpoint.update(
            score=valid_loss,
            epoch=e
        )
        logging.info(
            "[%d/%d] Train loss: %.3f, Validation loss: %.3f %s%s",
            e,
            config["nepochs"],
            train_loss,
            valid_loss,
            accuracy_msg,  # Remplacé accuracy_msg par accuracy
            "[>> BETTER <<]" if updated else "",
        )
        # Mise à jour des dashboards
        metrics = {
            "train_loss": train_loss, 
            "valid_loss": valid_loss,
        }

        if not contrastive:
            metrics["valid_overall_accuracy"] = valid_metrics["valid_overall_accuracy"]

        wandb_run.log(metrics)

        for key, value in metrics.items():
            tensorboard_writer.add_scalar(key, value, e)
    
    if contrastive :

        logging.info("###################### Finished contrastive pre-training ######################")
    
    else: 
        logging.info("###################### Final evaluation on valid loader ######################")

        model, _, score = model_checkpoint.load_best_checkpoint()

        logging.info(f"Loaded best model with training loss : {score:.3f}")

        test_metrics, _, test_cm = training_utils.test_epoch(
            model=model,
            loader=valid_loader,  
            device=device,
            number_classes=num_classes,
            ignore_index=0
        )

        log_confusion_matrix(
            wandb_run=wandb_run,
            cm=test_cm,
            title="Confusion Matrix",
            xlabel="Predictions",
            ylabel="Ground Truth",
        )

        if visualize:
            log_predictions_on_wandb(
                model,
                num_classes,
                device,
                data_config,
                logdir,
                ignore_index=0,
                training_metrics=metrics,
                use_cuda=use_cuda
            )
        else:
            wandb.log(metrics)


        logging.info("###################### End of training ######################")


def test(config, model, wandb_run):
    raise NotImplementedError

def main():
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 4:
        logging.error(f"Usage: {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    config_file = sys.argv[1]
    command = sys.argv[2]
    visualize = sys.argv[3]

    logging.info(f"Loading config from {config_file}")
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier de config: {e}")
        sys.exit(-1)
    
    if config.get("contrastive", False):
        run_type = "ContrastivePretraining"
    elif "pretrained_weights" in config.get("model", {}):
        run_type = "SegmentationFromPretrained"
    else:
        run_type = "SegmentationBaseline(NoContrastive)"

    # Initialiser WandB
    wandb_run = wandb.init(
            project="segmentation-polsf",
            entity="SONDRA_2024-2025",
            config=config,
            name=run_type + "_" + models.__name__,
            tags=[run_type],
    )

    if command == "train":
        if visualize == "visualize":
            train(config, wandb_run=wandb_run, visualize=True)
        else:
            train(config, wandb_run=wandb_run, visualize=False)
    elif command == "test":
        test(config, wandb_run=wandb_run)
    else:
        logging.error(f"Commande inconnue: {command}")
        sys.exit(-1)


if __name__ == "__main__":
    main()
