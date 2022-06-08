import cpp_canvas

from . import kernel_pack


def remove_cache():
    r"""Remove the cached kernel code directory.

        Example
        -------
        >>> canvas.remove_cache()
    """
    kernel_pack.remove_cache_dir()


def seed(value: int):
    r"""Set global seed for the C++ random engine.

        Parameters
        ----------
        value: int
            The seed to set for the C++ random engine.

        Example
        -------
        >>> canvas.seed(1998)

        """
    if not (type(value) == int and 0 <= value <= 0xffffffff):
        raise ValueError('The seed should be typed as an int, with a range'
                         'of C++ uint32_t, i.e. [0, 0xffffffff].')
    cpp_canvas.seed(value)
