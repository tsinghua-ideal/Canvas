import importlib
import os
import re
import shutil
import sys

import cpp_canvas


cache_dir_name = 'canvas_torch_cached'
parent_dir = os.getcwd()
cache_dir = parent_dir + '/' + cache_dir_name
sys.path.append(parent_dir)

cached_torch_kernels = {}  # The names of kernels
cached_torch_modules = {}  # The module class of kernels


def load_from_cache_dir(name: str):
    global cached_torch_kernels, cached_torch_modules
    if name not in cached_torch_kernels:
        cached_torch_modules[name] = importlib.import_module('{}.{}'.format(cache_dir_name, name))
    else:
        importlib.reload(cached_torch_modules[name])
    assert name in cached_torch_modules
    cached_torch_kernels[name] = cached_torch_modules[name].__dict__[name]
    return cached_torch_kernels[name]


def load_from_code(code: str, overwrite: bool = True):
    # Get class name from the code
    obj = re.search(r'class (.*?)\(nn.Module\):', code)
    assert obj is not None, 'No class founded in code'
    name = obj.group(1)

    # Write into the cache directory
    global cached_torch_kernels
    if not overwrite and name in cached_torch_kernels:
        return cached_torch_kernels[name]
    path = cache_dir + '/' + name + '.py'
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(path) or overwrite:
        with open(path, 'w') as file:
            file.write(code)
            file.flush()
            file.close()

    # Load kernel and specs
    cls = load_from_cache_dir(name)
    return cls


def remove_cache_dir():
    # noinspection PyBroadException
    try:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    except Exception as _:
        pass


class KernelPack:
    def __init__(self, kernel_pack_impl: cpp_canvas.KernelPackImpl):
        self.module = load_from_code(kernel_pack_impl.torch_code)
        self.fills = kernel_pack_impl.fills
        self.graphviz = kernel_pack_impl.graphviz_code
