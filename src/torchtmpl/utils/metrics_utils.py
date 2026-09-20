# MIT License
#
# Copyright (c) 2023 Jeremy Fix
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Code refactored by Emmanuel Benichou

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    jaccard_score,
)


def log_confusion_matrix(wandb_run, cm, title="Confusion Matrix", xlabel="Preds", ylabel="Labels"):

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    wandb_run.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()


def compute_metrics(predictions, ground_truth, ignore_index, num_classes=None, test=False):
    mask = ground_truth != ignore_index
    filtered_predictions = predictions[mask]
    filtered_ground_truth = ground_truth[mask]

    iou = jaccard_score(filtered_ground_truth, filtered_predictions, average="weighted")

    accuracy = accuracy_score(filtered_ground_truth, filtered_predictions)

    if test:
        conf_matrix = confusion_matrix(
            filtered_ground_truth,
            filtered_predictions,
            labels=np.setdiff1d(np.arange(0, num_classes), np.array([ignore_index])),
            normalize="true",
        )
        average_accuracy = balanced_accuracy_score(filtered_ground_truth, filtered_predictions)

        kappa = cohen_kappa_score(filtered_ground_truth, filtered_predictions)

        return iou, accuracy, average_accuracy, kappa, conf_matrix
    else:
        return iou, accuracy


def normalize_confusion_matrix(conf_matrix):
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = conf_matrix / (row_sums + 1e-6)
    return normalized_matrix


def compute_batch_iou(predictions, labels, ignore_index=None):
    mask = labels != ignore_index
    filtered_predictions = predictions[mask]
    filtered_ground_truth = labels[mask]

    return jaccard_score(filtered_ground_truth, filtered_predictions, average="weighted")


def empty_confusion_matrix(number_classes, ignore_index):
    classes = np.setdiff1d(np.arange(number_classes), np.array([ignore_index]))
    return classes, np.zeros((len(classes), len(classes)))


def compute_batch_confusion_matrix(predictions, targets, classes, ignore_index):
    mask = targets != ignore_index
    return confusion_matrix(
        targets[mask],
        predictions[mask],
        labels=classes,
    )


def compute_overall_accuracy(conf_matrix):
    return np.trace(conf_matrix) / conf_matrix.sum()


def compute_kappa(conf_matrix):
    row_sums = conf_matrix.sum(axis=1)
    col_sums = conf_matrix.sum(axis=0)
    expected_agreement = (row_sums @ col_sums) / conf_matrix.sum() ** 2
    observed_agreement = np.trace(conf_matrix) / conf_matrix.sum()
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def compute_iou(conf_matrix):
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    union = tp + fp + fn

    iou_per_class = np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union > 0)

    return iou_per_class, iou_per_class.mean()


def compute_classification_metrics(conf_matrix):
    precision = np.diag(conf_matrix) / (conf_matrix.sum(axis=0) + 1e-6)
    recall = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "macro_precision": precision.mean(),
        "macro_recall": recall.mean(),
        "macro_f1": f1.mean(),
    }
