import atexit
import glob
import importlib
import json
import os
import re
import shutil
import sys
import time

import cpp_canvas


cache_dir_name = 'canvas_torch_cached'
parent_dir = os.getcwd()
cache_dir = parent_dir + '/' + cache_dir_name
sys.path.append(parent_dir)

cached_torch_kernels = {}  # The names of kernels.
cached_torch_modules = {}  # The module class of kernels.


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
    # Get class name from the code.
    obj = re.search(r'class (.*?)\(nn.Module\):', code)
    assert obj is not None, 'No class founded in code'
    name = obj.group(1)

    # Write into the cache directory.
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

    # Load kernel and specs.
    cls = load_from_cache_dir(name)
    return name, cls


def remove_cache_dir():
    # noinspection PyBroadException
    try:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    except Exception as _:
        pass


# Remove all caches before exit.
# atexit.register(remove_cache_dir)


class KernelPack:
    r"""Kernel replacement solution for `nn.Module`s.

        Attributes
        ----------
        torch_code: str
            The generated PyTorch code, implemented in a `torch.nn.Module`.
        module: torch.nn.Module
            The class type of the code, you can directly use this to make
            kernel instances.
        graphviz_code: str
            The generated GraphViz code, you may use some tool to visualize.
        detail: json
            The details of the kernel.
    """

    def __init__(self, timestamp: int = 0,
                 torch_code: str = '', graphviz_code: str = '', detail=None):
        self.timestamp = timestamp
        self.torch_code = torch_code.strip()
        self.name, self.module = load_from_code(torch_code)
        self.graphviz_code = graphviz_code
        self.detail = detail

    def save_torch_code(self, path: str):
        assert self.torch_code, 'No PyTorch code exists in the pack'
        with open(path, 'w') as file:
            file.write(self.torch_code)

    def save_graphviz_code(self, path: str):
        assert self.graphviz_code, 'No graphviz code exists in the pack'
        with open(path, 'w') as file:
            file.write(self.graphviz_code)

    @staticmethod
    def load_from_cpp(kernel_pack_impl: cpp_canvas.KernelPackImpl):
        if kernel_pack_impl.exception_info:
            raise RuntimeError(kernel_pack_impl.exception_info)
        return KernelPack(timestamp=time.time_ns(),
                          torch_code=kernel_pack_impl.torch_code,
                          graphviz_code=kernel_pack_impl.graphviz_code)

    @staticmethod
    def load_from_dir(path: str):
        assert os.path.exists(path) and os.path.isdir(path), f'{path} should be a directory'

        # Read all context from file.
        def read(suffix: str, required=False):
            files = glob.glob(os.path.join(path, f'*.{suffix}'))
            assert not (required and len(files) == 0), 'Error while loading files, no such file found'
            if len(files) == 0:
                return ''
            elif len(files) > 1:
                print(f'Multiple files found, loading {files[0]} ...')
            with open(files[0], 'r') as file:
                return file.read()

        j = read('json')
        detail = json.loads(j) if j else None
        return KernelPack(torch_code=read('py', True),
                          graphviz_code=read('dot'),
                          detail=detail)

    @staticmethod
    def load_from_file(path: str):
        assert os.path.exists(path) and os.path.isfile(path), f'{path} should be a file'
        with open(path, 'r') as file:
            return KernelPack(torch_code=file.read())

    @staticmethod
    def load(file_or_dir: str):
        assert os.path.exists(file_or_dir), f'{file_or_dir} does not exist'
        return KernelPack.load_from_dir(file_or_dir) if os.path.isdir(file_or_dir) \
            else KernelPack.load_from_file(file_or_dir)
