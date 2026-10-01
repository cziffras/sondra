import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import tqdm
from matplotlib.colors import BoundaryNorm, ListedColormap
from torch import nn

import wandb

from .metrics_utils import predict


def plot_segmentation_images(
    to_be_vizualized: list,
    confusion_matrix: np.ndarray,
    number_classes: int,
    logdir: str,
    class_names: list,
    ignore_index: int = None,
    sets_masks: np.ndarray = None,
    other_metrics=None,
) -> None:
    """
    Plots segmentation images with an optional test mask overlay to indicate dataset splits.

    Args:
        to_be_vizualized (list): Array of shape (N, 3, H, W), where:
                                       - First channel: Ground truth.
                                       - Second channel: Prediction.
                                       - Third channel: Original image (optional).
        confusion_matrix (np.ndarray): Row-normalised confusion matrix of the test split, over
                                       the evaluated classes (the ignored one left out).
        number_classes (int): Number of classes for segmentation.
        logdir (str): Directory to save the plot.
        class_names (list): Name of every class, the ignored one included.
        wandb_log (bool): Whether to log the plot to Weights & Biases.
        ignore_index (int, optional): Value in the ground truth to be ignored in the masked prediction.
        sets_masks (np.ndarray, optional): Array of shape (N, H, W) with integer values indicating dataset splits:
                                           - 1: Train
                                           - 2: Validation
                                           - 3: Test
    """
    class_colors = {
        7: {
            0: "black",
            1: "purple",
            2: "blue",
            3: "green",
            4: "red",
            5: "cyan",
            6: "yellow",
        },
        5: {
            0: "black",
            1: "green",
            2: "brown",
            3: "blue",
            4: "yellow",
        },
    }.get(number_classes, {})

    cmap = ListedColormap([class_colors[key] for key in sorted(class_colors.keys())])
    bounds = np.arange(len(class_colors) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    patches = [
        mpatches.Patch(color=class_colors[i], label=class_names[i])
        for i in sorted(class_colors.keys())
    ]

    sets_mask_colors = {
        1: "red",
        2: "green",
        3: "blue",
    }
    sets_mask_cmap = ListedColormap(
        [sets_mask_colors[key] for key in sorted(sets_mask_colors.keys())]
    )
    sets_mask_bounds = np.arange(len(sets_mask_colors) + 1) - 0.5
    sets_mask_norm = BoundaryNorm(sets_mask_bounds, sets_mask_cmap.N)
    sets_mask_patches = [
        mpatches.Patch(color=sets_mask_colors[i], label=f"Set {i}")
        for i in sorted(sets_mask_colors.keys())
    ]

    num_samples = to_be_vizualized[0].shape[0]
    nrows = num_samples + 1
    ncols = 4 if sets_masks is not None else 3

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 5 * nrows),
        constrained_layout=True,
    )

    for i in range(num_samples):
        img = to_be_vizualized[0][i]
        g_t = to_be_vizualized[1][i]
        pred = to_be_vizualized[2][i]

        if ignore_index is not None:
            masked_pred = pred.copy()
            masked_pred[g_t == ignore_index] = ignore_index
        else:
            masked_pred = pred

        g_t = np.squeeze(g_t)
        axes[i][0].imshow(g_t, cmap=cmap, norm=norm, origin="lower")
        axes[i][0].set_title(f"Ground Truth {i + 1}")
        axes[i][0].axis("off")

        pred = np.squeeze(pred)
        axes[i][1].imshow(pred, cmap=cmap, norm=norm, origin="lower")
        axes[i][1].set_title(f"Prediction {i + 1}")
        axes[i][1].axis("off")

        masked_pred = np.squeeze(masked_pred)
        axes[i][2].imshow(masked_pred, cmap=cmap, norm=norm, origin="lower")
        axes[i][2].set_title(f"Masked Prediction {i + 1}")
        axes[i][2].axis("off")

        if sets_masks is not None:
            axes[i][3].imshow(sets_masks[i], cmap=sets_mask_cmap, norm=sets_mask_norm)
            axes[i][3].set_title(f"Sets Mask {i + 1}")
            axes[i][3].axis("off")

    evaluated_names = [name for i, name in enumerate(class_names) if i != ignore_index]
    sns.heatmap(
        confusion_matrix.round(decimals=3),
        annot=True,
        fmt=".2g",
        cmap="Blues",
        ax=axes[-1][0],
        xticklabels=evaluated_names,
        yticklabels=evaluated_names,
    )
    axes[-1][0].tick_params(axis="x", labelrotation=45)
    axes[-1][0].set_xlabel("Predicted Class")
    axes[-1][0].set_ylabel("Ground Truth Class")
    axes[-1][0].set_title("Test Confusion Matrix")

    legend_ax = axes[-1][1]
    legend_ax.axis("off")
    legend_ax.legend(handles=patches, loc="center", title="Classes")

    if sets_masks is not None:
        test_mask_legend_ax = axes[-1][2]
        test_mask_legend_ax.axis("off")
        test_mask_legend_ax.legend(handles=sets_mask_patches, loc="center", title="Test Masks")
    else:
        axes[-1][2].axis("off")

    if ncols == 4:
        axes[-1][3].axis("off")

    path = f"{logdir}/segmentation_images.png"
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    logs = {
        "segmentation_images": [
            wandb.Image(path, caption="Segmentation of the labelled area and test confusion matrix")
        ]
    }
    if other_metrics is not None:
        logs.update(other_metrics)

    logging.info("Logging image and metrics to wandb...")
    wandb.log(logs)


def reassemble_image(
    segments,
    samples_per_col,
    samples_per_row,
    num_channels,
    segment_size,
    real_indices,
    sets_indices=None,
):
    """
    Reassemble an image from its segments using real_indices to determine their positions.

    Args:
        segments: List or array of image segments.
        samples_per_col: Number of segments per column in the reassembled image.
        samples_per_row: Number of segments per row in the reassembled image.
        num_channels: Number of channels in the image.
        segment_size: Height/width of each square segment.
        real_indices: List of real indices corresponding to the segments.
        sets_indices: List of sets of indices for mask assignment (optional).

    Returns:
        reassembled_image: The reconstructed image tensor.
        mask: A mask indicating the set each segment belongs to (if sets_indices is provided).
    """
    img_height = samples_per_row * segment_size
    img_width = samples_per_col * segment_size

    reassembled_image = np.zeros((num_channels, img_height, img_width), dtype=segments[0].dtype)
    if sets_indices is None:
        mask = None
    else:
        mask = np.zeros_like(reassembled_image, dtype=np.uint8)

    index_to_position = {
        real_index: (row, col)
        for row in range(samples_per_row)
        for col in range(samples_per_col)
        for real_index in [row * samples_per_col + col]
    }

    for segment_index, real_index in enumerate(real_indices):
        if real_index not in index_to_position:
            raise ValueError(f"Real index {real_index} is out of bounds for the image grid.")

        row, col = index_to_position[real_index]
        h_start = row * segment_size
        w_start = col * segment_size

        reassembled_image[:, h_start : h_start + segment_size, w_start : w_start + segment_size] = (
            segments[segment_index]
        )

        if sets_indices is not None and mask is not None:
            if real_index in sets_indices[0]:
                mask[
                    :,
                    h_start : h_start + segment_size,
                    w_start : w_start + segment_size,
                ] = 0
            elif real_index in sets_indices[1]:
                mask[
                    :,
                    h_start : h_start + segment_size,
                    w_start : w_start + segment_size,
                ] = 1
            elif real_index in sets_indices[2]:
                mask[
                    :,
                    h_start : h_start + segment_size,
                    w_start : w_start + segment_size,
                ] = 2

    return reassembled_image, mask


def predict_patches(model, loader, device, ignore_index=0):
    """Predicted class map of every patch the loader yields, with the patch indices."""
    outputs = []
    list_of_indices = []
    model.eval()
    model.to(device)

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for inputs, _, idx in tqdm.tqdm(loader):
            pred_outputs = model(inputs.to(device))
            pred_outputs = predict(
                softmax(torch.abs(pred_outputs).type(torch.float64)), ignore_index
            )
            outputs.extend(pred_outputs.cpu().numpy())
            list_of_indices.extend(idx.cpu().numpy().tolist())

    return outputs, list_of_indices


def log_predictions_on_wandb(
    model,
    num_classes,
    device,
    config,
    logdir,
    test_confusion_matrix,
    class_names,
    ignore_index=0,
    training_metrics=None,
    use_cuda=False,
) -> None:
    """
    Maps of the prediction over the whole labelled area, train patches included, next to
    the confusion matrix of the test split alone: `test_confusion_matrix`, from test_epoch.
    """

    from ..data.wrappers import get_full_image_dataloader

    logging.info("Computing model predictions on the dataset...")
    img_size = config["data"].get("patch_size", (128, 128))[0]

    model.eval()

    (
        data_loader,
        nsamples_per_cols,
        nsamples_per_rows,
    ) = get_full_image_dataloader(config, use_cuda=use_cuda)

    reconstructed_tensors, list_of_indices = predict_patches(
        model=model,
        loader=data_loader,
        device=device,
        ignore_index=ignore_index,
    )

    image_tensors = []
    ground_truth_tensors = []
    indice_tensors = []

    for data in tqdm.tqdm(data_loader):
        image_tensors.extend(data[0].cpu().detach().numpy())
        ground_truth_tensors.extend(data[1].cpu().detach().numpy())
        indice_tensors.extend(data[2].cpu().detach().numpy())

    ground_truth, sets_masks = reassemble_image(
        segments=ground_truth_tensors,
        samples_per_col=nsamples_per_cols,
        samples_per_row=nsamples_per_rows,
        num_channels=(
            ground_truth_tensors[0].shape[0] if len(ground_truth_tensors[0].shape) > 2 else 1
        ),
        segment_size=img_size,
        real_indices=indice_tensors,
    )

    image_input, _ = reassemble_image(
        segments=image_tensors,
        samples_per_col=nsamples_per_cols,
        samples_per_row=nsamples_per_rows,
        num_channels=(image_tensors[0].shape[0] if len(image_tensors[0].shape) > 2 else 1),
        segment_size=img_size,
        real_indices=indice_tensors,
        sets_indices=None,
    )

    predicted, _ = reassemble_image(
        segments=reconstructed_tensors,
        samples_per_col=nsamples_per_cols,
        samples_per_row=nsamples_per_rows,
        num_channels=(
            reconstructed_tensors[0].shape[0] if len(reconstructed_tensors[0].shape) > 2 else 1
        ),
        segment_size=img_size,
        real_indices=list_of_indices,
        sets_indices=None,
    )

    to_be_vizualized = [
        image_input[np.newaxis, ...],
        ground_truth[np.newaxis, ...],
        predicted[np.newaxis, ...],
    ]

    plot_segmentation_images(
        to_be_vizualized=to_be_vizualized,
        confusion_matrix=test_confusion_matrix,
        number_classes=num_classes,
        ignore_index=ignore_index,
        logdir=logdir,
        class_names=class_names,
        sets_masks=sets_masks,
        other_metrics=training_metrics,
    )
