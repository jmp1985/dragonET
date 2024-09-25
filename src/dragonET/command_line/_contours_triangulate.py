#
# contours_triangulate.py
#
# Copyright (C) 2024 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
import time
from argparse import ArgumentParser
from typing import List

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

__all__ = ["contours_triangulate"]


def get_description():
    """
    Get the program description

    """
    return "Refine a model to align the projection images"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the contours_triangulate parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "--contours_in",
        type=str,
        default=None,
        dest="contours_in",
        required=True,
        help=(
            """
            A YAML file containing contour information.
            """
        ),
    )
    parser.add_argument(
        "--model_in",
        type=str,
        default=None,
        dest="model_in",
        required=True,
        help=(
            """
            A file describing the initial model. This file can either be a
            .rawtlt file or a YAML file.
            """
        ),
    )
    parser.add_argument(
        "--points_out",
        type=str,
        default="triangulated.npz",
        dest="points_out",
        help=(
            """
            A YAML file describing the refined model.
            """
        ),
    )

    return parser


def contours_triangulate_impl(args):
    """
    Triangulate the contours

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _contours_triangulate(
        args.model_in,
        args.contours_in,
        args.points_out,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def contours_triangulate(args: List[str] = None):
    """
    Triangulate the contours

    """
    contours_triangulate_impl(get_parser().parse_args(args=args))


def triangulate(dx, dy, a, b, c, data, mask):
    """
    Triangulate the points

    """

    # Create the observation matrix
    W = np.concatenate([data[:, :, 0], data[:, :, 1]], axis=0)
    M = np.concatenate([mask, mask], axis=0)

    # Get number of points
    num_points = W.shape[1]

    # Get the rotation matrices
    Rabc = Rotation.from_euler("yxz", np.stack([c, b, a], axis=1)).as_matrix()
    R = np.concatenate([Rabc[:, 0, :], Rabc[:, 1, :]], axis=0)

    # The translation
    t = np.concatenate([dx, dy])

    # Subtract centroid
    W = W - t[:, None]

    # Compute the 3D spot positions
    S = np.zeros((3, num_points))
    for j in range(num_points):
        Mj = M[:, j]
        W0 = W[Mj, j]
        Rj = R[Mj, :]
        S[:, j] = np.linalg.inv(Rj.T @ Rj) @ Rj.T @ W0
    return S


def _contours_triangulate(
    model_in: str,
    contours_in: str,
    points_out: str,
):
    """
    Triangulate the contours

    """

    def read_points(filename) -> tuple:
        print("Reading points from %s" % filename)
        handle = np.load(filename)
        return handle["data"], handle["mask"], handle["octave"]

    def read_model(filename) -> dict:
        print("Reading model from %s" % filename)
        return yaml.safe_load(open(filename, "r"))

    def write_points(filename, points):
        print("Writing contours to %s" % filename)
        np.savez(open(filename, "wb"), points=points)

    # Read the model
    model = read_model(model_in)

    # Get the parameters
    P = np.array(model["transform"])

    # The image size
    image_size = model["image_size"]

    # Read the points
    data, mask, octave = read_points(contours_in)

    # Get the parameters
    dx = P[:, 0] + 0.5
    dy = P[:, 1] + 0.5
    a = np.radians(P[:, 2])
    b = np.radians(P[:, 3])
    c = np.radians(P[:, 4])

    # Triangulate the points
    points = triangulate(dx, dy, a, b, c, data, mask)

    # Write the contours
    write_points(points_out, points)


if __name__ == "__main__":
    contours_triangulate()
