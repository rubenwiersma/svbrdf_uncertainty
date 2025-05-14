from pathlib import Path

import numpy as np
from pyexr import read
from skimage.io import imread


def read_texture(path):
    """Read a texture from exr/png/jpg/etc.
    Uses pyexr for .exr files and skimage.io.imread for all other formats.
    """
    if isinstance(path, Path): path = str(path)
    if Path(path).match('*.exr'):
        return read(path)
    return imread(path).astype(np.float32) / 255