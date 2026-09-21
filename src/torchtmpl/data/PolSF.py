import logging
import os
import pathlib

import numpy as np
import torch
from PIL import Image
from torchcvnn.datasets import ALOSDataset
from torchcvnn.transforms import LogAmplitude, PolSARtoTensor
from torchvision.transforms import v2

from ..transforms import SARContrastiveAugmentations
from ..utils import ToTensor


class EnhancedPolSFDataset(ALOSDataset):
    """
    Hybrid Dataset that behaves as PolSFDataset yielding patches and labels
    in supervised mode and conversely provides two augmented views with contrastive
    transformations.
    """

    ALOS_PATH_SUFFIX = "VOL-ALOS2044980750-150324-HBQR1.1__A"

    def __init__(
        self,
        root,
        contrastive_mode=False,
        transform=None,
        transform_contrastive=None,
        augment_transform=None,
        patch_size=(128, 128),
        crop_coordinates=None,
        patch_stride=None,
        index_tracking=False,
        **kwargs,
    ):
        if not contrastive_mode:
            crop_coordinates = ((2832, 736), (7888, 3520))

        root = pathlib.Path(root)
        if root.name != self.ALOS_PATH_SUFFIX:
            root = root / self.ALOS_PATH_SUFFIX
        if not root.is_file():
            raise FileNotFoundError(
                f"ALOS-2 volume file not found at {root}. Point data.root_dir at the "
                f"directory holding {self.ALOS_PATH_SUFFIX} and SF-ALOS2-label2d.png."
            )

        super().__init__(
            volpath=root,
            transform=transform,
            patch_size=patch_size,
            patch_stride=patch_stride,
            crop_coordinates=crop_coordinates,
            **kwargs,
        )

        self.contrastive_mode = contrastive_mode
        self.transform_contrastive = transform_contrastive
        self.augment_transform = augment_transform
        self.index_tracking = index_tracking
        self.to_tensor_labels = ToTensor(dtype=torch.int64)

        self.classes = [
            "0 - Unlabeled",
            "1 - Mountain",
            "2 - Water",
            "3 - Vegetation",
            "4 - High-Density Urban",
            "5 - Low-Density Urban",
            "6 - Developed",
        ]

        if not contrastive_mode:
            labels_path = root.parent / "SF-ALOS2-label2d.png"
            self.labels = np.array(Image.open(labels_path))[::-1, :].copy()

    def __getitem__(self, idx):
        if self.contrastive_mode:
            return self._get_contrastive_item(idx)
        else:
            return self._get_supervised_item(idx)

    def _get_contrastive_item(self, idx):

        data = super().__getitem__(idx)

        if self.transform_contrastive is not None:
            view1, view2 = self.transform_contrastive(data)
        else:
            view1, view2 = data, data

        view1 = view1.to(torch.complex64)
        view2 = view2.to(torch.complex64)

        if self.index_tracking:
            return view1, view2, idx
        return view1, view2

    def _get_supervised_item(self, idx):
        """
        Supervised mode : get the patch and corresponding label behaving like PolSFDataset.
        """
        patch = super().__getitem__(idx)

        row_stride, col_stride = self.patch_stride
        nsamples_per_cols = self.nsamples_per_cols

        start_row = (idx // nsamples_per_cols) * row_stride
        start_col = (idx % nsamples_per_cols) * col_stride

        num_rows, num_cols = self.patch_size
        labels = self.labels[start_row : (start_row + num_rows), start_col : (start_col + num_cols)]

        # WARNING : augmentation transform is NOT a contrastive transform
        if self.augment_transform is not None:
            patch = self.augment_transform(patch)

        labels = self.to_tensor_labels(labels)

        if self.index_tracking:
            return patch, labels, idx
        return patch.to(torch.complex64), labels


class PolSFDataManager:
    """
    Manager that handles both datasets and dataloaders creation.
    """

    def __init__(self, config, use_cuda=False):

        self.use_cuda = use_cuda
        self.contrastive = config["model"].get("contrastive", False)
        self.config = config["data"]

        # POLSF_ROOT wins over the config
        self.root_dir = os.environ.get("POLSF_ROOT", self.config["root_dir"])
        self.batch_size = self.config["batch_size"]
        self.num_workers = self.config["num_workers"]
        self.patch_size = tuple(self.config.get("patch_size", (128, 128)))
        self.patch_stride = tuple(self.config.get("patch_stride", self.patch_size))

    def _loader(self, dataset, shuffle):
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.use_cuda,
        )

    def get_dataloaders(self):
        if self.contrastive:
            return self._get_contrastive_dataloaders()
        return self._get_supervised_dataloaders()

    def _get_contrastive_dataloaders(self):
        """
        Pre-training sees every patch of the unlabelled regions.

        No split is held out: the checkpoint cannot be selected on a validation
        NT-Xent anyway, since that loss also drops when a stage collapses. The
        run is a fixed budget and the last epoch is what gets reused.
        """
        dataset = self._create_contrastive_dataset()
        logging.info(f"Contrastive pre-training on {len(dataset)} patches")

        # num_classes is a placeholder: the segmentation head is built either
        # way and cannot be sized to zero, it just stays unused here
        return self._loader(dataset, shuffle=True), None, None, tuple(dataset[0][0].shape), 1, []

    def _get_supervised_dataloaders(self):
        dataset = self._create_standard_dataset()
        train, valid, test = self._mosaic_split(dataset)

        logging.info(
            f"Mosaic split : {len(train)} train, {len(valid)} valid, {len(test)} test patches"
        )

        classes = list(dataset.classes)
        return (
            self._loader(train, shuffle=True),
            self._loader(valid, shuffle=False),
            self._loader(test, shuffle=False),
            tuple(dataset[0][0].shape),
            len(classes),
            classes,
        )

    def get_full_image_dataloader(self):

        dataset = EnhancedPolSFDataset(
            root=self.root_dir,
            contrastive_mode=False,
            transform=v2.Compose([PolSARtoTensor()]),
            augment_transform=LogAmplitude(),
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            index_tracking=True,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        alos_dataset = dataset.alos_dataset if hasattr(dataset, "alos_dataset") else dataset

        return (
            loader,
            alos_dataset.nsamples_per_cols,
            alos_dataset.nsamples_per_rows,
        )

    def _create_contrastive_dataset(self):

        crop_regions = [
            ((0, 0), (7888, 736)),
            ((0, 736), (2832, 8080)),
            ((7888, 0), (22608, 8080)),
            ((2832, 3520), (7888, 8080)),
        ]

        transform = self._get_transform()

        transform_pipeline = v2.Compose([PolSARtoTensor(), transform, LogAmplitude()])

        datasets = []
        for coords in crop_regions:
            datasets.append(
                EnhancedPolSFDataset(
                    root=self.root_dir,
                    contrastive_mode=True,
                    transform_contrastive=transform_pipeline,
                    patch_size=self.patch_size,
                    patch_stride=self.patch_stride,
                    crop_coordinates=coords,
                )
            )

        return torch.utils.data.ConcatDataset(datasets)

    def _create_standard_dataset(self):

        return EnhancedPolSFDataset(
            root=self.root_dir,
            contrastive_mode=False,
            transform=v2.Compose([PolSARtoTensor(), LogAmplitude()]),
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
        )

    def _mosaic_split(self, dataset):
        """
        In prior version of this project, we were selecting patches at random, 
        now the patch grid is splitted on a fixed lattice to ensure that :
        - two test patches are never adjacent
        - every interior test patch has the same neighbourhood; three train patches 
        and one valid patch.

        A patch at row i, column j goes to `(i + 2 * j) % 5`, which sends a
        fifth of the grid to test, a fifth to valid and the rest to train. 

        Being a lattice it needs no seed, and it keeps the class balance of the
        scene: the three subsets stay within a third of a point of each other
        on every class.

        NOTE : 

        It does not pretend to measure generalisation to an unseen area: each
        test patch is surrounded by training data 64 pixels away. It fixes that
        bias identically for everyone instead of removing it, so a difference in 
        score between two configs cannot come from a lucky draw.
        """
        columns = dataset.nsamples_per_cols
        buckets = {0: [], 1: [], 2: []}

        for idx in range(len(dataset)):
            row, column = divmod(idx, columns)
            cell = (row + 2 * column) % 5
            buckets[cell if cell < 2 else 2].append(idx)

        return tuple(
            torch.utils.data.Subset(dataset, buckets[cell]) for cell in (2, 1, 0)
        )  # train, valid, test

    def _get_transform(self):
        params = self.config.get("transform_contrastive") or {}
        return SARContrastiveAugmentations(**(params.get("params") or {}))
