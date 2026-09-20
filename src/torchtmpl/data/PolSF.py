import logging
import pathlib
import random
import os 
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

        self.config = config
        self.use_cuda = use_cuda

        # POLSF_ROOT wins over the config
        self.root_dir = os.environ.get("POLSF_ROOT", config["root_dir"])
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.valid_ratio = config["valid_ratio"]
        self.patch_size = tuple(config.get("patch_size", (128, 128)))
        self.patch_stride = tuple(config.get("patch_stride", self.patch_size))

    def get_dataloaders(self, contrastive=False):

        if contrastive:
            dataset = self._create_contrastive_dataset()
        else:
            dataset = self._create_standard_dataset()

        logging.info(f"Loaded {len(dataset)} samples")

        train_dataset, valid_dataset = self._split_dataset(dataset)

        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.use_cuda,
        }

        train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, **loader_kwargs)

        # contrastive mode concatenates crops of the unlabelled regions, so there
        # are no class names; num_classes stays at 1 because the segmentation head
        # is built either way and cannot be sized to zero
        classes = [] if contrastive else list(dataset.classes)
        num_classes = 1 if contrastive else len(classes)
        input_size = tuple(dataset[0][0].shape)

        return train_loader, valid_loader, input_size, num_classes, classes

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

        transform = self._get_transform(contrastive=True)

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

    def _split_dataset(self, dataset):
        """Splits a dataset in train and valid subsets"""
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        split = int(self.valid_ratio * len(dataset))

        return (
            torch.utils.data.Subset(dataset, indices[split:]),
            torch.utils.data.Subset(dataset, indices[:split]),
        )

    def _get_transform(self, contrastive=False):
        """Get transform from config (see transforms directory to see what options we got)"""

        if contrastive:
            transform_args = self.config.get("transform_contrastive", {})
            transform_params = transform_args.get("params", {})
            transform = SARContrastiveAugmentations(**transform_params)
        else:
            transform_args = self.config.get("transform_supervised", {})
            transform_params = transform_args.get("params", {})
            transform = lambda x: x  # noqa: E731

        return transform
