#
# rot90.py
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

__all__ = ["rot90"]


def get_description():
    """
    Get the program description

    """
    return "Rotate the projection images by 90 degrees"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the rot90 parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        dest="input",
        required=True,
        help=(
            """
            The input projection images.
            """
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        dest="output",
        required=True,
        help=(
            """
            The output projection images.
            """
        ),
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1,
        dest="num",
        help="The number of anticlockwise 90 degree rotations",
    )

    return parser


def rot90_impl(args):
    """
    Rotate the projection images

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _rot90(
        args.input,
        args.output,
        args.num,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def rot90(args: List[str] = None):
    """
    Rotate the projection images

    """
    rot90_impl(get_parser().parse_args(args=args))


def _rot90(
    input_filename: str,
    output_filename: str,
    num: int,
):
    """
    Rotate the projection images

    """

    def read_projections(filename):
        print("Reading projections from %s" % filename)
        return mrcfile.mmap(filename)

    def write_projections(filename, projections):
        print("Writing projections to %s" % filename)
        handle = mrcfile.new(filename, overwrite=True)
        handle.set_data(projections)

    # Load the projections data
    projections_file = read_projections(input_filename)

    # Rotate the projections
    projections = np.rot90(projections_file.data, num, axes=(1, 2))

    # Write the model
    write_projections(output_filename, projections)
