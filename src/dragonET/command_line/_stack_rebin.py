#
# stack_rebin.py
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

__all__ = ["stack_rebin"]


def get_description():
    """
    Get the program description

    """
    return "Rebin the stack"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the stack rebin parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "-i",
        type=str,
        default=None,
        dest="projections_in",
        required=True,
        help=("""
            The filename for the input projection images
            """),
    )
    parser.add_argument(
        "-o",
        type=str,
        default="rebinned.mrc",
        dest="projections_out",
        required=False,
        help=("""
            The filename for the output projection images
            """),
    )
    parser.add_argument(
        "-f",
        "--factor",
        type=int,
        default=1,
        dest="factor",
        help=("""
            The rebin factor (must be a power of 2).
            """),
    )

    return parser


def stack_rebin_impl(args):
    """
    Rebin the stack

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _stack_rebin(
        args.projections_in,
        args.projections_out,
        args.factor,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def stack_rebin(args: List[str] = None):
    """
    Rebin the stack

    """
    stack_rebin_impl(get_parser().parse_args(args=args))


def rebin_stack(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Rebin the image stack

    """

    def is_power_of_2(n):
        return (n & (n - 1) == 0) and n != 0

    # Check rebin factor
    assert is_power_of_2(factor)

    # If factor is > 1 then rebin
    if factor > 1:
        shape = np.array(data.shape) // np.array([1, factor, factor])
        print(
            "Rebinning stack by factor %d from (%d, %d) -> (%d, %d)"
            % (factor, data.shape[1], data.shape[2], shape[1], shape[2])
        )
        shape = (
            shape[0],
            shape[1],
            factor,
            shape[2],
            factor,
        )
        data = data.reshape(shape).sum(-1).sum(2).astype("float32")
    return data


def _stack_rebin(
    projections_in: str,
    projections_out: str,
    factor: int,
):
    """
    Rebin the stack

    """

    def read_projections(filename):
        print("Reading projections from %s" % filename)
        return mrcfile.mmap(filename).data

    def write_projections(projections, filename):
        print("Writing projections to %s" % filename)
        handle = mrcfile.new(filename, overwrite=True)
        handle.set_data(projections)

    # Read the projections
    projections = read_projections(projections_in)

    # Rebin the stack
    projections = rebin_stack(projections, factor)

    # Write the projections
    write_projections(projections, projections_out)


if __name__ == "__main__":
    stack_rebin()
