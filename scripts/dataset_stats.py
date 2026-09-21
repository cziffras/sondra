"""
uv run python -m scripts.dataset_stats configs/baseline_segformer.yaml
"""

import sys
from pathlib import Path

import torch
import yaml

from src.torchtmpl.data import get_dataloaders


def count_labels(loader, num_classes):
    """
    Pixel count per class, and how many patches hold a single class (a network
    computing embeddings for mixed labels patches will probably struggle to learn
    relevant information).
    """
    counts = torch.zeros(num_classes, dtype=torch.long)
    single_class = 0
    patches = 0

    for _, targets in loader:
        counts += torch.bincount(targets.reshape(-1), minlength=num_classes)
        for target in targets:
            patches += 1
            single_class += int(target.min() == target.max())

    return counts, single_class, patches


def main(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    data = get_dataloaders(config, use_cuda=False)

    counts, single_class, patches = count_labels(data.train, data.num_classes)
    share = (100 * counts / counts.sum()).tolist()

    assert hasattr(data, "valid"), "get_dataloaders returned an instance without `valid` attribute"
    
    lines = [
        f"train {len(data.train.dataset)} patches, valid {len(data.valid.dataset)}", # type: ignore
        f"input {data.input_size}, {counts.sum().item()} pixels read",
        "",
        *(f"{name} : {percent:.2f}%" for name, percent in zip(data.classes, share)),
        "",
        f"annotated : {100 - share[0]:.2f}%",
        f"single-class patches : {100 * single_class / patches:.2f}%",
    ]

    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} config.yaml")
    main(sys.argv[1])
