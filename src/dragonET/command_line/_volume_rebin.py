#
# volume_rebin.py
#
# Copyright (C) 2024 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
import time
from argparse import ArgumentParser
from typing import List

import mrcfile
import numpy as np

__all__ = ["volume_rebin"]


def get_description():
    """
    Get the program description

    """
    return "Rebin the volume"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the volume rebin parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "-i",
        type=str,
        default=None,
        dest="volume_in",
        required=True,
        help=("""
            The filename for the input projection images
            """),
    )
    parser.add_argument(
        "-o",
        type=str,
        default="rebinned.mrc",
        dest="volume_out",
        required=False,
        help=("""
            The filename for the output projection images
            """),
    )
    parser.add_argument(
        "-f",
        "--factor",
        type=float,
        default=1,
        dest="factor",
        help=("""
            The rebin factor (must be a power of 2).
            """),
    )

    return parser


def volume_rebin_impl(args):
    """
    Rebin the volume

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _volume_rebin(
        args.volume_in,
        args.volume_out,
        args.factor,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def volume_rebin(args: List[str] = None):
    """
    Rebin the volume

    """
    volume_rebin_impl(get_parser().parse_args(args=args))


def downsample_volume(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Rebin the volume

    """

    def is_power_of_2(n):
        return (n & (n - 1) == 0) and n != 0

    # Check rebin factor
    assert is_power_of_2(factor)

    # Downsample the volume
    shape = np.array(data.shape) // np.array([factor, factor, factor])
    print(
        "Rebinning volume by factor %d from (%d, %d, %d) -> (%d, %d, %d)"
        % (
            factor,
            data.shape[0],
            data.shape[1],
            data.shape[2],
            shape[0],
            shape[1],
            shape[2],
        )
    )
    shape = (
        shape[0],
        factor,
        shape[1],
        factor,
        shape[2],
        factor,
    )
    data = data.reshape(shape).sum(-1).sum(-2).sum(-3).astype("float32")

    # Return the sampled data
    return data


def upsample_volume(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Rebin the volume

    """

    def is_power_of_2(n):
        return (n & (n - 1) == 0) and n != 0

    # Check rebin factor
    assert is_power_of_2(factor)

    # Save the original data scaled down
    scaled_original = data / factor**3

    # Upsample the volume
    shape = np.array(data.shape) * np.array([factor, factor, factor])
    print(
        "Rebinning volume by factor %d from (%d, %d, %d) -> (%d, %d, %d)"
        % (
            factor,
            data.shape[0],
            data.shape[1],
            data.shape[2],
            shape[0],
            shape[1],
            shape[2],
        )
    )
    data = np.zeros_like(data, shape=shape)
    for k in range(factor):
        for j in range(factor):
            for i in range(factor):
                data[k::factor, j::factor, i::factor] = scaled_original

    # Return the sampled data
    return data


def rebin_volume(data: np.ndarray, factor: float) -> np.ndarray:
    """
    Rebin the volume

    """
    if factor > 1:
        data = downsample_volume(data, int(np.round(factor)))
    elif factor < 1:
        data = upsample_volume(data, int(np.round(1.0 / factor)))
    return data


def _volume_rebin(
    volume_in: str,
    volume_out: str,
    factor: float,
):
    """
    Rebin the volume

    """

    def read_volume(filename):
        print("Reading volume from %s" % filename)
        return mrcfile.mmap(filename).data

    def write_volume(volume, filename):
        print("Writing volume to %s" % filename)
        handle = mrcfile.new_mmap(
            filename,
            volume.shape,
            mrc_mode=mrcfile.utils.mode_from_dtype(volume.dtype),
            overwrite=True,
        )
        handle.data[:] = volume

    # Read the volume
    volume = read_volume(volume_in)

    # Rebin the volume
    volume = rebin_volume(volume, factor)

    # Write the volume
    write_volume(volume, volume_out)


if __name__ == "__main__":
    volume_rebin()
