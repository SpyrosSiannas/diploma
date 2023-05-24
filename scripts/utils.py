import os
from typing import List

import numpy as np

PLY_HEADER = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
end_header
"""


def parse_ply_ascii_line(line: str) -> List[float]:
    """Parse single PLY line.

    Args:
    ----
    line (str): Line to parse

    Returns:
    -------
    List[float]: Array of point coordinates
    """
    try:
        line_values = [float(x) for x in line.rstrip().split(" ")]
    except Exception as e:
        print(f"Error parsing line: {line}, {e}")
        line_values = []

    return line_values


def read_ply_ascii_geo(filedir):
    """Read a point cloud from a PLY file.

    Args:
    ----
    filedir: (string or path-like object):

    Returns:
    -------
    np.array: The point cloud coordinates as ints
    """
    with open(filedir) as f:
        data = []
        # Get rid of the header
        for line in f:
            if line.startswith("end_header"):
                break
        for line in f:
            data.append(parse_ply_ascii_line(line))
        data = np.array(data)
        coords = data[:, 0:3].astype("int")

    return coords


def write_ply_ascii_geo(filedir, coords):
    """Write a point cloud to a PLY file.

    Args:
    ----
    filedir (string or path-like object): The filename in which to save the point cloud data.
    coords (numpy array): The coordinates to save
    """
    if os.path.exists(filedir):
        os.remove(filedir)
    with open(filedir, "a+") as f:
        f.write(PLY_HEADER.format(coords.shape[0]))
        coords = coords.astype("int")
        for p in coords:
            coords_str = [str(x) for x in p]
            f.writelines(" ".join(coords_str) + "\n")


def array2vector(array, step):
    """Ravel 2D array with multi-channel to one 1D vector by sum each channel with different step."""
    array, step = array.long().cpu(), step.long().cpu()
    vector = sum([array[:, i] * (step**i) for i in range(array.shape[-1])])

    return vector
