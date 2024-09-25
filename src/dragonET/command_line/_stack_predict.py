#
# stack_predict.py
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
from scipy.spatial.transform import Rotation

import dragonET

__all__ = ["stack_predict"]


def get_description():
    """
    Get the program description

    """
    return "Predict the stack images"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the stack predict parser

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
        default="predicted.mrc",
        dest="projections_out",
        required=False,
        help=(
            """
            The filename for the output projection images
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
        "-s",
        "--subset_size",
        type=int,
        default=1,
        dest="subset_size",
        help=(
            """
            The size of the subset to use to predict adjacent images.
            """
        ),
    )

    return parser


def stack_predict_impl(args):
    """
    Predict the stack images

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _stack_predict(
        args.projections_in,
        args.projections_out,
        args.model_in,
        args.subset_size,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def stack_predict(args: List[str] = None):
    """
    Predict the stack images

    """
    stack_predict_impl(get_parser().parse_args(args=args))


def get_matrix_from_parameters(P):
    """
    Get the matrices from the parameters

    """

    # Create the rotation matrix for each image
    a = np.radians(P[:, 2])  # Yaw
    b = np.radians(P[:, 3])  # Pitch
    c = np.radians(P[:, 4])  # Roll
    Rabc = Rotation.from_euler("yxz", np.stack([c, b, a]).T).as_matrix()

    # Construct the matrix from the parameters
    R = np.full((P.shape[0], 4, 4), np.eye(4))
    R[:, :3, :3] = Rabc
    R[:, 0, 3] = P[:, 0]  # Shift X
    R[:, 1, 3] = P[:, 1]  # Shift Y
    return R


def get_parameters_from_matrix(R):
    """
    Get the parameters from the matrix

    """
    euler = Rotation.from_matrix(R[:, :3, :3]).as_euler("yxz")
    P = np.zeros((R.shape[0], 5))
    P[:, 0] = R[:, 0, 3]
    P[:, 1] = R[:, 1, 3]
    P[:, 2] = np.degrees(euler[:, 2])
    P[:, 3] = np.degrees(euler[:, 1])
    P[:, 4] = np.degrees(euler[:, 0])
    return P


def predict_image(data: np.ndarray, P_data: np.ndarray, P_image: np.ndarray):
    """
    Predict the image

    """
    # Check the data and parameter shapes
    assert data.shape[0] == P_data.shape[0]

    # Get matrices for the image
    R_data = get_matrix_from_parameters(P_data)
    R_image = get_matrix_from_parameters(P_image[None, :])

    # Rotate all the input images w.r.t the output image
    R_data = R_data @ np.linalg.inv(R_image)
    R_image = R_image @ np.linalg.inv(R_image)

    # Get the updated parameter arrays
    P_data = get_parameters_from_matrix(R_data)
    P_image = get_parameters_from_matrix(R_image)

    # Compute the maximum angular difference between images
    R_diff = np.dot(R_data[:, :3, :3], R_image[:3, :3].T)
    cos_theta = (np.trace(R_diff, axis1=1, axis2=2) - 1) / 2
    diff_angle = np.abs(np.arccos(cos_theta))

    # Compute the height of the volume to reconstruct
    height = int(np.ceil(np.max(data.shape[1:]) * np.max(np.abs(np.sin(diff_angle)))))

    # Init the volume
    shape = (data.shape[1], height, data.shape[2])
    volume = np.zeros(shape, dtype="float32")

    # Prepare to reconstruct
    data = np.swapaxes(data, 0, 1).copy()

    # Reconstruct the volume from the input images
    volume = dragonET.command_line._reconstruct.recon(
        data, P_data, volume, 1, np.array((0, 1, 0)), np.array((0, 0, 0)), 1, "gpu"
    )

    # Return the predicted image by projecting along the axis
    return np.sum(volume, axis=1)


def predict_stack(data: np.ndarray, P: np.ndarray, subset_size: int) -> np.ndarray:
    """
    Predict the stack images

    """
    # Check the subset size is atleast 1
    assert subset_size >= 1

    # Initialise the result. We have 2 predictions per image
    result = np.zeros(
        (2, data.shape[0], data.shape[1], data.shape[2]), dtype=data.dtype
    )

    # Do the prediction for each image
    for j in range(data.shape[0]):
        for k, (i0, i1) in enumerate(
            [(j - subset_size, j), (j + 1, j + 1 + subset_size)]
        ):
            if i0 >= 0 and i1 <= data.shape[0]:
                print("Predicting image %d from images %d to %d" % (j, i0, i1))
                result[k, j] = predict_image(data[i0:i1], P[i0:i1], P[j])

    # Return the result
    return result


def _stack_predict(
    projections_in: str,
    projections_out: str,
    model_in: str,
    subset_size: int,
):
    """
    Predict the stack images

    """

    def read_projections(filename):
        print("Reading projections from %s" % filename)
        return mrcfile.mmap(filename).data

    def read_model(filename):
        print("Reading model from %s" % filename)
        return yaml.safe_load(open(filename))

    def write_projections(projections, filename):
        print("Writing projections to %s" % filename)
        handle = mrcfile.new(filename, overwrite=True)
        handle.set_data(projections)

    # Read the projections
    projections = read_projections(projections_in)

    # Read the model
    P = np.array(read_model(model_in)["transform"])

    # Predict the stack images
    projections = predict_stack(projections, P, subset_size)

    # Write the projections
    write_projections(projections, projections_out)


if __name__ == "__main__":
    stack_predict()
