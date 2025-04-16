# From torchcvnn/examples/polsf_unet


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import wandb
from torch.autograd import Variable
import torch
import torch.nn as nn
import tqdm
from .metrics_utils import (
    compute_batch_confusion_matrix,
    normalize_confusion_matrix,
)
import logging


def plot_segmentation_images(
    to_be_vizualized: list,
    confusion_matrix: np.ndarray,
    number_classes: int,
    logdir: str,
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
        confusion_matrix (np.ndarray): Confusion matrix of shape (number_classes, number_classes).
        number_classes (int): Number of classes for segmentation.
        logdir (str): Directory to save the plot.
        wandb_log (bool): Whether to log the plot to Weights & Biases.
        ignore_index (int, optional): Value in the ground truth to be ignored in the masked prediction.
        sets_masks (np.ndarray, optional): Array of shape (N, H, W) with integer values indicating dataset splits:
                                           - 1: Train
                                           - 2: Validation
                                           - 3: Test
    """
    # Define colormap for segmentation classes
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
        mpatches.Patch(color=class_colors[i], label=f"Class {i}")
        for i in sorted(class_colors.keys())
    ]

    # Define colormap for sets masks
    sets_mask_colors = {
        1: "red",  # Train
        2: "green",  # Validation
        3: "blue",  # Test
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

    # Limit number of samples to visualize
    num_samples = to_be_vizualized[0].shape[0]
    nrows = num_samples + 1  # +1 for confusion matrix
    ncols = 4 if sets_masks is not None else 3  # Add test mask column if available

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 5 * nrows),
        constrained_layout=True,
    )

    # Plot ground truth, predictions, masked predictions, and optionally test masks
    for i in range(num_samples):
        img = to_be_vizualized[0][i]
        g_t = to_be_vizualized[1][i]
        pred = to_be_vizualized[2][i]

        # Mask prediction if ignore_index is provided
        if ignore_index is not None:
            masked_pred = pred.copy()
            masked_pred[g_t == ignore_index] = ignore_index
        else:
            masked_pred = pred

        # Plot ground truth
        g_t = np.squeeze(g_t)
        axes[i][0].imshow(g_t, cmap=cmap, norm=norm, origin="lower")
        axes[i][0].set_title(f"Ground Truth {i+1}")
        axes[i][0].axis("off")

        # Plot prediction
        pred = np.squeeze(pred)
        axes[i][1].imshow(pred, cmap=cmap, norm=norm, origin="lower")
        axes[i][1].set_title(f"Prediction {i+1}")
        axes[i][1].axis("off")

        # Plot masked prediction
        masked_pred = np.squeeze(masked_pred)
        axes[i][2].imshow(masked_pred, cmap=cmap, norm=norm, origin="lower")
        axes[i][2].set_title(f"Masked Prediction {i+1}")
        axes[i][2].axis("off")

        # Plot test mask if available
        if sets_masks is not None:
            axes[i][3].imshow(sets_masks[i], cmap=sets_mask_cmap, norm=sets_mask_norm)
            axes[i][3].set_title(f"Sets Mask {i+1}")
            axes[i][3].axis("off")

    # Plot confusion matrix in the last row
    sns.heatmap(
        confusion_matrix.round(decimals=3),
        annot=True,
        fmt=".2g",
        cmap="Blues",
        ax=axes[-1][0],
        xticklabels=np.setdiff1d(
            np.arange(0, number_classes), np.array([ignore_index])
        ),
        yticklabels=np.setdiff1d(
            np.arange(0, number_classes), np.array([ignore_index])
        ),
    )
    axes[-1][0].set_xlabel("Predicted Class")
    axes[-1][0].set_ylabel("Ground Truth Class")
    axes[-1][0].set_title("Confusion Matrix")

    # Add legends
    legend_ax = axes[-1][1]
    legend_ax.axis("off")
    legend_ax.legend(handles=patches, loc="center", title="Classes")

    # Add sets mask legend if applicable
    if sets_masks is not None:
        test_mask_legend_ax = axes[-1][2]
        test_mask_legend_ax.axis("off")
        test_mask_legend_ax.legend(
            handles=sets_mask_patches, loc="center", title="Test Masks"
        )
    else:
        axes[-1][2].axis("off")

    # Leave extra columns blank for symmetry
    if ncols == 4:
        axes[-1][3].axis("off")

    # Save the figure
    path = f"{logdir}/segmentation_images.png"
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    # Log to Weights & Biases if enabled

    logs = {
        "segmentation_images": [
            wandb.Image(path, caption="Segmentation Images and Confusion Matrix")
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
    # Calculate total image dimensions
    img_height = samples_per_row * segment_size
    img_width = samples_per_col * segment_size

    # Initialize the empty image tensor with the correct shape
    reassembled_image = np.zeros(
        (num_channels, img_height, img_width), dtype=segments[0].dtype
    )
    if sets_indices is None:
        mask = None
    else:
        mask = np.zeros_like(reassembled_image, dtype=np.uint8)

    # Map real_indices to their positions
    index_to_position = {
        real_index: (row, col)
        for row in range(samples_per_row)
        for col in range(samples_per_col)
        for real_index in [row * samples_per_col + col]
    }

    # Place each segment into the correct position
    for segment_index, real_index in enumerate(real_indices):
        if real_index not in index_to_position:
            raise ValueError(
                f"Real index {real_index} is out of bounds for the image grid."
            )

        # Get the target row and column
        row, col = index_to_position[real_index]
        h_start = row * segment_size
        w_start = col * segment_size

        # Insert the segment into the image
        reassembled_image[
            :, h_start : h_start + segment_size, w_start : w_start + segment_size
        ] = segments[segment_index]

        # Update the mask if sets_indices is provided
        if mask is not None:
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


def one_forward_with_conf_mat(model, loader, device, number_classes, ignore_index=0):
    outputs = []
    model.eval()
    model.to(device)

    softmax = nn.Softmax(dim=1)

    list_of_indices = []

    size = np.setdiff1d(np.arange(0, number_classes), np.array([ignore_index]))
    conf_matrix_accum = np.zeros((len(size), len(size)))

    with torch.no_grad():
        for _, data in enumerate(tqdm.tqdm(loader)):

            # Handle different data structures (tuple, list, or otherwise)
            if isinstance(data, (tuple, list)):
                if len(data) == 2:
                    inputs, labels = data  # For standard datasets
                elif len(data) == 3:
                    inputs, labels, idx = data  # For wrapped datasets
                    list_of_indices.extend(idx.cpu().numpy().tolist())
                else:
                    raise ValueError("Unexpected data format in loader.")
            else:
                inputs = data
                labels = None
            # Need to adapt the wrapper and the collect of the indices for reconstruction datasets

            inputs = Variable(inputs).to(device)

            # Forward propagate through the model
            pred_outputs = model(inputs)

            pred_outputs = (
                softmax(torch.abs(pred_outputs).type(torch.float64))
                .argmax(dim=1)
                .cpu()
                .numpy()
            )
            outputs.extend(pred_outputs)

            labels_flat = labels.cpu().numpy().flatten()
            batch_cm = compute_batch_confusion_matrix(
                predictions=pred_outputs.flatten(),
                labels=labels_flat,
                ignore_index=ignore_index,
                size=size,
            )

            conf_matrix_accum += batch_cm

    conf_matrix_accum = normalize_confusion_matrix(conf_matrix_accum)
    return (
        outputs,
        list_of_indices,
        conf_matrix_accum,
    )


def log_predictions_on_wandb(
    model,
    num_classes,
    device,
    data_config,
    logdir,
    ignore_index=0,
    training_metrics=None,
    use_cuda=False,
) -> None:
    """
    Test the model based on the given configuration.
    """

    from ..data.wrappers import get_full_image_dataloader

    logging.info("Computing model predictions on the dataset...")
    img_size = data_config.get("patch_size", (128, 128))[0]

    model.eval()

    (
        data_loader,
        nsamples_per_cols,
        nsamples_per_rows,
    ) = get_full_image_dataloader(data_config, use_cuda=use_cuda) # 

    (
        reconstructed_tensors,
        list_of_indices,
        cm,
    ) = one_forward_with_conf_mat(
        model=model,
        loader=data_loader,
        device=device,
        number_classes=num_classes,
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
            ground_truth_tensors[0].shape[0]
            if len(ground_truth_tensors[0].shape) > 2
            else 1
        ),
        segment_size=img_size,
        real_indices=indice_tensors,
    )

    image_input, _ = reassemble_image(
        segments=image_tensors,
        samples_per_col=nsamples_per_cols,
        samples_per_row=nsamples_per_rows,
        num_channels=(
            image_tensors[0].shape[0] if len(image_tensors[0].shape) > 2 else 1
        ),
        segment_size=img_size,
        real_indices=indice_tensors,
        sets_indices=None,
    )

    predicted, _ = reassemble_image(
        segments=reconstructed_tensors,
        samples_per_col=nsamples_per_cols,
        samples_per_row=nsamples_per_rows,
        num_channels=(
            reconstructed_tensors[0].shape[0]
            if len(reconstructed_tensors[0].shape) > 2
            else 1
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
        confusion_matrix=cm,
        number_classes=num_classes,
        ignore_index=ignore_index,
        logdir=logdir,
        sets_masks=sets_masks,
        other_metrics=training_metrics,
    )
