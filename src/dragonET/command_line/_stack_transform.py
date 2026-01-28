#
# stack_transform.py
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
import scipy
import yaml
from scipy.spatial.transform import Rotation

__all__ = ["stack_transform"]


def get_description():
    """
    Get the program description

    """
    return "Transform the stack"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the stack transform parser

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
        default="transformed.mrc",
        dest="projections_out",
        required=False,
        help=("""
            The filename for the output projection images
            """),
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        dest="model_in",
        required=True,
        help=("""
            The transform model.
            """),
    )

    return parser


def stack_transform_impl(args):
    """
    Transform the stack

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _stack_transform(
        args.projections_in,
        args.projections_out,
        args.model_in,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def stack_transform(args: List[str] = None):
    """
    Transform the stack

    """
    stack_transform_impl(get_parser().parse_args(args=args))


def transform_stack(images, matrix):
    images = images.copy()
    for i in range(images.shape[0]):
        Y, X = np.mgrid[0 : images.shape[1], 0 : images.shape[2]]
        X, Y = (
            matrix[i, 0, 0] * X + matrix[i, 0, 1] * Y + matrix[i, 0, 2],
            matrix[i, 1, 0] * X + matrix[i, 1, 1] * Y + matrix[i, 1, 2],
        )
        images[i] = scipy.ndimage.map_coordinates(images[i], (Y, X))
    return images


def _stack_transform(
    projections_in: str,
    projections_out: str,
    model_in: str,
):
    """
    Rebin the stack

    """

    def read_model(filename):
        print("Reading model from %s" % filename)
        return yaml.safe_load(open(filename))

    def read_projections(filename):
        print("Reading projections from %s" % filename)
        return mrcfile.mmap(filename).data

    def write_projections(projections, filename):
        print("Writing projections to %s" % filename)
        handle = mrcfile.new(filename, overwrite=True)
        handle.set_data(projections)

    def get_matrix(P, image_size):
        # Get the origin translation
        oy, ox = np.array(image_size) / 2

        # Get the components
        dx = P[:, 0] * image_size[1]
        dy = P[:, 1] * image_size[0]
        a = np.radians(P[:, 2])
        b = np.radians(P[:, 3])
        c = np.radians(P[:, 4])

        # Only use in plane rotation and translation
        matrix = Rotation.from_euler("z", a).as_matrix()
        matrix[:, 0, 2] = dx
        matrix[:, 1, 2] = dy

        # Translate from centre of the image
        T = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]])

        # First subtract the centre of image, then apply rotation, then
        # translate back to centre
        matrix = T @ matrix @ np.linalg.inv(T)

        # Return the matrix
        return matrix

    # Read the model
    model = read_model(model_in)

    # Read the projections
    projections = read_projections(projections_in)

    # Get the matrix from the model
    matrix = get_matrix(np.array(model["transform"]), projections.shape[1:])

    # Rebin the stack
    projections = transform_stack(projections, matrix)

    # Write the projections
    write_projections(projections, projections_out)


if __name__ == "__main__":
    stack_transform()
