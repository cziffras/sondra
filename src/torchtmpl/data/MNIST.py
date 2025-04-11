import logging
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from torchvision.utils import make_grid
import random
import torch
import torchvision
import torchvision.transforms.v2 as v2_transforms

class MNISTSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img, _ = self.dataset[idx]
        mask = (img.real > 0).to(torch.int64).squeeze(0).float()
        return img, mask
    
    @property
    def classes(self):
        num_classes = [0, 1]
        return num_classes
    
def get_mnist_dataloaders(data_config, use_cuda, segmentation=True):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]

    cdtype = torch.complex64

    original_dataset = torchvision.datasets.MNIST(
        root=data_config.get("root_dir","./datasets"),
        train=True,
        download=True,
        transform=v2_transforms.Compose(
            [v2_transforms.PILToTensor(), v2_transforms.ToDtype(cdtype)]
        ),
    )
    
    if segmentation: 
        train_valid_dataset = MNISTSegmentationDataset(original_dataset)
    else:
        train_valid_dataset = original_dataset

    all_indices = list(range(len(train_valid_dataset)))
    random.shuffle(all_indices)
    split_idx = int(valid_ratio * len(train_valid_dataset))
    valid_indices, train_indices = all_indices[:split_idx], all_indices[split_idx:]

    # Train dataloader
    train_dataset = torch.utils.data.Subset(train_valid_dataset, train_indices)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Valid dataloader
    valid_dataset = torch.utils.data.Subset(train_valid_dataset, valid_indices)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )
    num_classes = len(train_valid_dataset.classes)
    input_size = tuple(train_dataset[0][0].shape)

    return train_loader, valid_loader, input_size, num_classes

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def test_mnist_dataloaders():
    data_config = {
        "root_dir": './data',
        "batch_size": 32,
        "num_workers": 0,
        "valid_ratio": 0.2,
    }
    use_cuda = torch.cuda.is_available()

    logging.info("using CUDA" if use_cuda else "NOT using CUDA ...")

    train_loader, valid_loader, input_size, num_classes = get_mnist_dataloaders(
        data_config, use_cuda
    )

    X, y = next(iter(train_loader))
    
    assert X.dtype == torch.complex64, "Input tensor is not complex !"

    show(make_grid(X.clone().real))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_MNIST_dataloaders()