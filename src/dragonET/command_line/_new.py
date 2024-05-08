#
# new.py
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
import yaml

__all__ = ["new"]


def get_description():
    """
    Get the program description

    """
    return "Import experimental description"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the new parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "-p",
        "--projections",
        type=str,
        default=None,
        dest="projections",
        required=True,
        help=(
            """
            The projection images.
            """
        ),
    )
    parser.add_argument(
        "-a",
        "--angles",
        type=str,
        default=None,
        required=True,
        help=(
            """
            The angles in the rawtlt file.
            """
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="initial_model.yaml",
        dest="model",
        help=(
            """
            A YAML file describing the initial model.
            """
        ),
    )
    parser.add_argument(
        "-r",
        "--global_rotation",
        type=float,
        default=0,
        dest="global_rotation",
        help="The global in plane rotation (degrees)",
    )

    return parser


def new_impl(args):
    """
    Import the experimental description

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _new(
        args.projections,
        args.angles,
        args.model,
        args.global_rotation,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def new(args: List[str] = None):
    """
    Import the experimental description

    """
    new_impl(get_parser().parse_args(args=args))


def _new(
    projections_filename: str,
    angles_filename: str,
    model_filename: str,
    global_rotation=0,
):
    """
    Import the experimental description

    """

    def read_projections(filename):
        print("Reading projections from %s" % filename)
        return mrcfile.mmap(filename)

    def read_angles(filename):
        print("Reading angles from %s" % filename)
        return np.array(list(map(float, open(filename).readlines())))

    def write_model(filename, model):
        print("Writing model to %s" % filename)
        yaml.safe_dump(model, open(filename, "w"), default_flow_style=None)

    # Load the projections data
    projections_file = read_projections(projections_filename)

    # Read the angles
    angles = read_angles(angles_filename)

    # Check input
    assert projections_file.data.shape[0] == angles.size

    # Set the image size
    image_size = projections_file.data.shape[1:]

    # Construct the transform
    transform = np.zeros((angles.size, 5))
    transform[:, 0] = global_rotation
    transform[:, 2] = angles

    # Construct the model dictionary
    model = {
        "axis_origin": (0, 0, 0),
        "axis": (1, 0, 0),
        "image_size": image_size,
        "transform": transform.tolist(),
    }

    # Write the model
    write_model(model_filename, model)
