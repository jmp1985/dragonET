#
# stack_rot90.py
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

__all__ = ["stack_rot90"]


def get_description():
    """
    Get the program description

    """
    return "Rotate the stack"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the stack rot90 parser

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
        help=(
            """
            The filename for the input projection images
            """
        ),
    )
    parser.add_argument(
        "-o",
        type=str,
        default="rot90ned.mrc",
        dest="projections_out",
        required=False,
        help=(
            """
            The filename for the output projection images
            """
        ),
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=1,
        dest="number",
        help=(
            """
            The number of times to rotate by 90 degrees.
            """
        ),
    )

    return parser


def stack_rot90_impl(args):
    """
    Rotate the stack

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _stack_rot90(
        args.projections_in,
        args.projections_out,
        args.number,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def stack_rot90(args: List[str] = None):
    """
    Rotate the stack

    """
    stack_rot90_impl(get_parser().parse_args(args=args))


def _stack_rot90(
    projections_in: str,
    projections_out: str,
    number: int,
):
    """
    Rotate the stack

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

    # Rotate the stack
    if number != 0:
        projections = np.rot90(projections, number, axes=(1, 2))

    # Write the projections
    write_projections(projections, projections_out)


if __name__ == "__main__":
    stack_rot90()
