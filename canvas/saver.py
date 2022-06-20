from torch import nn
from collections import OrderedDict


def get_state_dict(module: nn.Module, remove_placeholders: bool = False):
    items = module.state_dict().items()
    if remove_placeholders:
        items = filter(lambda item: 'canvas_placeholder_kernel' not in item[0], items)
    return OrderedDict({k: v.cpu() for k, v in items})


def restore_from_state_dict(module: nn.Module, state_dict: OrderedDict, strict: bool = False):
    missing, unexpected = module.load_state_dict(state_dict, strict=strict)
    for key in missing:
        assert 'canvas_placeholder_kernel' in key
    assert len(unexpected) == 0
