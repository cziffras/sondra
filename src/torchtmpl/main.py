import logging
import os
import pathlib
import sys

import torch
import wandb
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchinfo import torchinfo

from . import losses, models, optim
from .data import get_dataloaders
from .utils import (
    check_model_params_validity,
    count_parameters,
    log_confusion_matrix,
    log_predictions_on_wandb,
    training_contrastive_utils,
    training_utils,
)


def train(config, wandb_run, visualize):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if use_cuda:
        torch.cuda.empty_cache()

    logging.info(f"Using device : {device}")

    contrastive = config["model"].get("contrastive", False)

    logging.info("= Attempting a forward pass with given config")
    check_model_params_validity(config, use_cuda, contrastive)

    logging.info("= Building the dataloaders")
    data_config = config["data"]

    data = get_dataloaders(data_config, use_cuda, contrastive)
    train_loader, valid_loader = data.train, data.valid
    input_size, num_classes = data.input_size, data.num_classes

    logging.info(f"= Input size {input_size}, {num_classes} classes")

    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    num_params = count_parameters(model)
    model.to(device)

    logging.info(f"= Model has {num_params} parameters")

    logging.info("= Loss")
    if contrastive:
        loss = losses.get_loss(
            config["loss"]["name"],
            num_stages=len(list(config["model"]["widths"])),
            lambda_reg=config["model"].get("lambda_reg", 1.0),
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

    model_checkpoint = training_utils.ModelCheckpoint(
        model, optimizer, str(logdir), num_input_dims, min_is_best=True
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
        if contrastive:
            train_metrics = train_epoch_func(
                model=model,
                loader=train_loader,
                f_loss=loss,
                optim=optimizer,
                scheduler=scheduler,
                device=device,
                epoch=e,
                lambda_reg=config["model"].get("lambda_reg", 1),
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

        if contrastive:
            valid_metrics = valid_func(
                model=model,
                loader=valid_loader,
                f_loss=loss,
                device=device,
                lambda_reg=config["model"].get("lambda_reg", 1),
            )
        else:
            valid_metrics = valid_func(
                model=model,
                loader=valid_loader,
                f_loss=loss,
                device=device,
                number_classes=num_classes,
            )

        valid_loss = valid_metrics["valid_loss"]

        metrics = {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }

        if not contrastive:
            metrics["valid_overall_accuracy"] = valid_metrics["valid_overall_accuracy"]

        log_vars_dict = {}
        beta_dict = {}
        weights_dict = {}

        if contrastive:
            if hasattr(loss, "log_vars"):
                lv = loss.weights.cpu()
                log_vars_dict = {f"log_vars/stage_{i}": lv[i].item() for i in range(len(lv))}
                for k, v in log_vars_dict.items():
                    tensorboard_writer.add_scalar(k, v, e)

            if hasattr(loss, "log_beta"):
                bw = loss.weights.cpu()
                beta_dict = {f"beta_weights/stage_{i}": bw[i].item() for i in range(len(bw))}
                for k, v in beta_dict.items():
                    tensorboard_writer.add_scalar(k, v, e)

            if hasattr(loss, "logits"):
                ws = loss.weights.cpu()
                weights_dict = {f"weights/stage_{i}": ws[i].item() for i in range(len(ws))}
                for k, v in weights_dict.items():
                    tensorboard_writer.add_scalar(k, v, e)

        all_logs = {**metrics, **log_vars_dict, **beta_dict, **weights_dict}
        wandb_run.log(all_logs, step=e)

        if not contrastive:
            valid_accuracy = valid_metrics.get("valid_overall_accuracy", None)
            accuracy_msg = (
                f", Accuracy : {valid_accuracy:.3f}% " if valid_accuracy is not None else ""
            )
        else:
            accuracy_msg = ""

        updated = model_checkpoint.update(score=valid_loss, epoch=e)
        logging.info(
            "[%d/%d] Train loss: %.3f, Validation loss: %.3f %s%s",
            e,
            config["nepochs"],
            train_loss,
            valid_loss,
            accuracy_msg,
            "[>> BETTER <<]" if updated else "",
        )

        for key, value in metrics.items():
            tensorboard_writer.add_scalar(key, value, e)

        if config["model"].get("scheduler", None) == "CosineAnnealingLR":
            scheduler.step()

    if contrastive:
        logging.info(
            "###################### Finished contrastive pre-training ######################"
        )

    else:
        logging.info(
            "###################### Final evaluation on valid loader ######################"
        )

        model, _, score = model_checkpoint.load_best_checkpoint()

        logging.info(f"Loaded best model with training loss : {score:.3f}")

        test_metrics, _, test_cm = training_utils.test_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            number_classes=num_classes,
            ignore_index=0,
        )

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
                data_config,
                logdir,
                ignore_index=0,
                training_metrics=metrics,
                use_cuda=use_cuda,
            )
        else:
            wandb_run.log(metrics)

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
