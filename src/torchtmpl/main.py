import json
import logging
import os
import pathlib
import sys

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchinfo import torchinfo

import wandb

from . import losses, models, optim
from .data import get_dataloaders
from .utils import (
    check_model_params_validity,
    count_parameters,
    log_confusion_matrix,
    log_predictions_on_wandb,
    set_seed,
    training_contrastive_utils,
    training_utils,
)


def train(config, wandb_run, visualize):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if use_cuda:
        torch.cuda.empty_cache()

    logging.info(f"Using device : {device}")

    # before anything is built: the weights are drawn at construction
    set_seed(config["seed"])
    logging.info(f"= Seed {config['seed']}")

    contrastive = config["model"].get("contrastive", False)

    logging.info("= Attempting a forward pass with given config")
    check_model_params_validity(config, use_cuda)

    logging.info("= Building the dataloaders")

    data = get_dataloaders(config, use_cuda)
    train_loader, valid_loader = data.train, data.valid
    input_size, num_classes = data.input_size, data.num_classes

    logging.info(f"= Input size {input_size}, {num_classes} classes")

    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)

    if "pretrained_weights" in model_config:
        path = model_config["pretrained_weights"]
        transferred = models.load_pretrained_encoder(model, path)
        logging.info(
            f"= Encoder initialised from {path} ({transferred} tensors), "
            "decoder and head start from scratch"
        )

    num_params = count_parameters(model)
    model.to(device)

    logging.info(f"= Model has {num_params} parameters")

    logging.info("= Loss")
    if contrastive:
        loss = losses.get_loss(
            config["loss"]["name"],
            num_stages=len(model_config.get("widths", [])),
            lambda_reg=model_config.get("lambda_reg", 1.0),
        ).to(device)
    else:
        loss = losses.get_loss(config["loss"]["name"], ignore_index=0)

    logging.info("= Optimizer")
    optimizer = optim.get_optimizer(config, model.parameters())

    logging.info("= Scheduler")
    steps_per_epoch = len(train_loader)
    scheduler = optim.get_scheduler(config, optimizer, steps_per_epoch)

    logging_config = config["logging"]
    logname = model_config["class"]
    logdir = training_utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    tensorboard_writer = SummaryWriter(logdir)

    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    verbose = config.get("verbose", True)

    if verbose:
        batch_shape = next(iter(train_loader))[0].shape
        summary_text = (
            f"Logdir : {logdir}\n"
            + "## Command \n"
            + " ".join(sys.argv)
            + "\n\n"
            + f" Config : {config} \n\n"
            + "## Summary of the model architecture\n"
            + f"{torchinfo.summary(model, input_size=batch_shape, dtypes=[torch.complex64] if bool(config['model'].get('is_complex', False)) else None)}\n\n"
            + "## Loss\n\n"
            + f"{loss}\n\n"
            + "## Datasets : \n"
            + f"Train : {len(train_loader.dataset)} patches\n"
            + f"Validation : {len(valid_loader.dataset) if valid_loader else 'none (pre-training)'}"
        )
        with open(logdir / "summary.txt", "w", encoding="utf-8") as f:
            f.write(summary_text)
        logging.info(summary_text)

    num_input_dims = len(input_size)

    model_checkpoint = training_utils.ModelCheckpoint(
        model, optimizer, str(logdir), num_input_dims, min_is_best=True
    )

    train_epoch_func = (
        training_contrastive_utils.train_one_contrastive_epoch
        if contrastive
        else training_utils.train_one_epoch
    )
    for e in range(config["nepochs"]):
        if contrastive:
            train_metrics = train_epoch_func(
                model=model,
                loader=train_loader,
                f_loss=loss,
                optim=optimizer,
                scheduler=scheduler,
                device=device,
                epoch=e,
            )

        else:
            train_metrics = train_epoch_func(
                model=model,
                loader=train_loader,
                f_loss=loss,
                optim=optimizer,
                scheduler=scheduler,
                device=device,
                number_classes=num_classes,
                epoch=e,
            )

        train_loss = train_metrics["train_loss"]
        metrics = {"train_loss": train_loss}

        if contrastive:
            # pre-training holds nothing out, so the checkpoint follows the
            # training loss and the run is really a fixed budget
            selection_score = train_loss
        else:
            valid_metrics = training_utils.valid_epoch(
                model=model,
                loader=valid_loader,
                f_loss=loss,
                device=device,
                number_classes=num_classes,
            )
            selection_score = valid_metrics["valid_loss"]
            metrics["valid_loss"] = selection_score
            metrics["valid_overall_accuracy"] = valid_metrics["valid_overall_accuracy"]
            metrics["valid_mean_iou"] = valid_metrics["valid_mean_iou"]

        # every hierarchical loss exposes the same `weights` property: we
        # watch no stage is collapsing
        if hasattr(loss, "weights"):
            stage_weights = loss.weights.cpu()
            metrics |= {f"stage_weights/{i}": w.item() for i, w in enumerate(stage_weights)}

        for key, value in metrics.items():
            tensorboard_writer.add_scalar(key, value, e)
        wandb_run.log(metrics, step=e)

        if contrastive:
            model_checkpoint.save(score=train_loss, epoch=e)
            logging.info("[%d/%d] train %.3f", e, config["nepochs"], train_loss)
        else:
            updated = model_checkpoint.update(score=selection_score, epoch=e)
            logging.info(
                "[%d/%d] train %.3f, valid %.3f%s",
                e,
                config["nepochs"],
                train_loss,
                selection_score,
                " [>> BETTER <<]" if updated else "",
            )

        if scheduler is not None and config["scheduler"]["name"] == "CosineAnnealingLR":
            scheduler.step()

    if contrastive:
        logging.info(
            "###################### Finished contrastive pre-training ######################"
        )

    else:
        logging.info("###################### Final evaluation on test ######################")

        model, _, score = model_checkpoint.load_best_checkpoint()

        logging.info(f"Loaded best model, validation loss : {score:.3f}")

        # the test split never took part in selecting this checkpoint
        test_metrics, _, test_cm = training_utils.test_epoch(
            model=model,
            loader=data.test,
            device=device,
            number_classes=num_classes,
            ignore_index=0,
        )

        # next to the checkpoint, so that a run directory holds its own results
        with open(logdir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2, default=lambda array: array.tolist())

        wandb_run.log(test_metrics)

        wandb_run.log(
            {
                "test_per_class": wandb.Table(
                    columns=["class", "precision", "recall", "f1", "iou"],
                    data=[
                        [name, p, r, f, i]
                        for name, p, r, f, i in zip(
                            data.classes[1:],
                            test_metrics["test_precision_per_class"],
                            test_metrics["test_recall_per_class"],
                            test_metrics["test_f1_per_class"],
                            test_metrics["test_iou_per_class"],
                        )
                    ],
                )
            }
        )

        logging.info(
            "Test : accuracy %.2f%%, mean IoU %.2f%%, macro F1 %.2f%%, kappa %.2f%%",
            test_metrics["test_overall_accuracy"],
            test_metrics["test_mean_iou"],
            test_metrics["test_macro_f1"],
            test_metrics["test_kappa_score"],
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
                config,
                logdir,
                ignore_index=0,
                training_metrics=metrics,
                use_cuda=use_cuda,
            )

        logging.info("###################### End of training ######################")


def test(config, model, wandb_run):
    raise NotImplementedError


def main():

    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 4:
        logging.error(f"Usage: {sys.argv[0]} config.yaml <train|test> <visualize|novis>")
        sys.exit(-1)

    config_file = sys.argv[1]
    command = sys.argv[2]
    visualize = sys.argv[3]

    logging.info(f"Loading config from {config_file}")
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        sys.exit(-1)

    model_config = config["model"]
    if model_config.get("contrastive", False):
        run_type = "ContrastivePretraining"
    elif "pretrained_weights" in model_config:
        run_type = "SegmentationFromPretrained"
    else:
        run_type = "SegmentationBaseline"

    wandb_config = config.get("wandb", {})
    wandb_run = wandb.init(
        project=wandb_config.get("project", "segmentation-polsf"),
        entity=wandb_config.get("entity"),
        config=config,
        name=f"{run_type}_{model_config['class']}",
        tags=[run_type, model_config["class"]],
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
