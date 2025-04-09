import sys
import torch
from .unet import SegmentationUNet

def build_model(cfg, input_size, num_classes):
    module = sys.modules[__name__]
    model_class = getattr(module, cfg['class'])
    return model_class(cfg, input_size, num_classes)