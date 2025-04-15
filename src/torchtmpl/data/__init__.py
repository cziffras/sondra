import sys
import inspect
from .wrappers_inprogress import get_polsf_dataloaders # Switch import here to use rather olf or newer version
from .MNIST import get_mnist_dataloaders

non_seg_datadict = {"polsf"}


def get_dataloaders(data_config, use_cuda, contrastive=False):
    """
    Calls get_[dataset]_dataloaders method and only passes the required 
    args extracted from config. 
    """
    
    module = sys.modules[__name__]
    func_name = f"get_{data_config['dataset']}_dataloaders"
    func = getattr(module, func_name)
    
    segmentation = data_config.get("segmentation", True)
    
    all_args = {
        "data_config": data_config,
        "use_cuda": use_cuda,
        "segmentation": segmentation,
        "contrastive": contrastive
    }
    
    # if dataset in seg_datadict delete segmentation arg
    if data_config["dataset"] in non_seg_datadict:
        all_args.pop("segmentation")
 
    sig = inspect.signature(func)
    valid_args = {k: v for k, v in all_args.items() if k in sig.parameters}
    
    return func(**valid_args)