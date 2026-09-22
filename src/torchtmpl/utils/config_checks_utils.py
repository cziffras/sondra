def check_model_params_validity(config, use_cuda):

    from ..data import get_dataloaders
    from ..models import build_model

    data = get_dataloaders(config, use_cuda)

    first_batch_inputs, _ = next(iter(data.train))

    model = build_model(config["model"], data.input_size, data.num_classes)

    try:
        _ = model(first_batch_inputs)
    except Exception as err:
        raise ValueError(
            f"Invalid forward : check model's config. For more details : {err}"
        ) from err

    print("############## Config is correct ##############")


def count_parameters(model):
    # `numel()` counts a complex tensor as one entry per element
    return sum(
        p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad
    )
