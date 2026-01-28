#
# stack_edit.py
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

__all__ = ["stack_edit"]


def get_description():
    """
    Get the program description

    """
    return "Rebin the stack"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the stack edit parser

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
        default="edited.mrc",
        dest="projections_out",
        required=False,
        help=("""
            The filename for the output projection images
            """),
    )
    parser.add_argument(
        "--exclude",
        type=lambda x: [int(xx) for xx in x.split(",")],
        default=None,
        dest="exclude",
        help=("""
            The image indices (zero indexed) to exclude.
            """),
    )

    return parser


def stack_edit_impl(args):
    """
    Rebin the stack

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _stack_edit(
        args.projections_in,
        args.projections_out,
        args.exclude,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def stack_edit(args: List[str] = None):
    """
    Rebin the stack

    """
    stack_edit_impl(get_parser().parse_args(args=args))


def exclude_images(data: np.ndarray, exclude: list) -> np.ndarray:
    """
    Remove images from stack

    """

    # If the exclude list is not None then exclude frames
    if exclude is not None:
        select = np.ones(data.shape[0], dtype=bool)
        select[exclude] = False
        data = data[select, :, :]
    return data


def _stack_edit(
    projections_in: str,
    projections_out: str,
    exclude: list,
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
    projections = exclude_images(projections, exclude)

    # Write the projections
    write_projections(projections, projections_out)


if __name__ == "__main__":
    stack_edit()
