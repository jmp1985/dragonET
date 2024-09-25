#
# contours_extend.py
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
import scipy.ndimage
import scipy.optimize
import scipy.signal
import yaml

# from matplotlib import pylab
from scipy.spatial.transform import Rotation

import dragonET

__all__ = ["contours_extend"]


def get_description():
    """
    Get the program description

    """
    return "Refine a model to align the projection images"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the contours_extend parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "-p",
        type=str,
        default=None,
        dest="projections_in",
        required=True,
        help=(
            """
            The filename for the projection images
            """
        ),
    )
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
        "--contours_out",
        type=str,
        default="extended.npz",
        dest="contours_out",
        help=(
            """
            A YAML file describing the refined model.
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


def contours_extend_impl(args):
    """
    Extend the contours

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _contours_extend(
        args.projections_in,
        args.model_in,
        args.contours_in,
        args.contours_out,
        args.subset_size,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def contours_extend(args: List[str] = None):
    """
    Extend the contours

    """
    contours_extend_impl(get_parser().parse_args(args=args))


def compute_derivatives(predicted, image):
    """
    Compute the image derivatives

    """
    stencil = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
    Ix = scipy.ndimage.convolve(predicted, stencil[None, :], mode="nearest")
    Iy = scipy.ndimage.convolve(predicted, stencil[:, None], mode="nearest")
    It = image - predicted
    return Ix, Iy, It


def compute_optical_flow(Ix, Iy, It):
    # Compute the weights
    # Wx = scipy.signal.windows.gaussian(Ix.shape[0], Ix.shape[0] / (2*3))
    # Wy = scipy.signal.windows.gaussian(Iy.shape[0], Iy.shape[0] / (2*3))
    # W = np.diag((Wx[None, :] * Wy[:, None]).flatten())

    # Compute the optical flow
    A = np.stack([Ix.flatten(), Iy.flatten()]).T
    B = -It.flatten()
    # V = np.linalg.pinv(A.T @ W @ A) @ (A.T @ W @ B)
    V = np.linalg.pinv(A.T @ A) @ (A.T @ B)
    return V


def extend_contours_for_image(
    stack, image, P_stack, P_image, points, data, mask, octave, max_threshold=0.8
):
    """
    Try to extend the contours onto the image

    """

    # Initialise the data and mask
    data_image = np.zeros((1, data.shape[1], data.shape[2]))
    mask_image = np.zeros((1, mask.shape[1]))

    # Predict the image from the images
    predicted = dragonET.command_line._stack_predict.predict_image(
        stack, P_stack, P_image
    )

    # Compute the derivatives
    Ix, Iy, It = compute_derivatives(predicted, image)

    # Select features that are visible in the stack
    select = np.count_nonzero(mask, axis=0) > 0

    # Predict the feature on the image
    dx = P_image[0] + stack.shape[2] // 2
    dy = P_image[1] + stack.shape[1] // 2
    a = np.radians(P_image[2])
    b = np.radians(P_image[3])
    c = np.radians(P_image[4])

    # Get the rotation matrices
    Rabc = Rotation.from_euler("yxz", np.stack([c, b, a])).as_matrix()
    R = np.stack([Rabc[0, :], Rabc[1, :]])

    # The translation
    t = np.array([dx, dy])

    # Predict the points
    W = R @ points[:, select] + t[:, None]
    X = W[0, :]
    Y = W[1, :]

    # Compute the sizes
    size = 16 * 2 ** octave[select]

    # Loop through selected features
    cc = np.zeros(X.shape[0])
    Vx = np.zeros(X.shape[0])
    Vy = np.zeros(X.shape[0])
    for index in range(X.shape[0]):
        # Predict the x, y location of the feature
        xc = X[index]
        yc = Y[index]

        # Compute ROI
        i0 = int(max(0, np.floor(xc - size[index] // 2)))
        i1 = int(min(stack.shape[2], np.ceil(i0 + size[index])))
        j0 = int(max(0, np.floor(yc - size[index] // 2)))
        j1 = int(min(stack.shape[1], np.ceil(j0 + size[index])))
        if i0 >= i1 or j0 >= j1:
            continue

        # Compute the optical flow to register the feature
        V = compute_optical_flow(Ix[j0:j1, i0:i1], Iy[j0:j1, i0:i1], It[j0:j1, i0:i1])

        # Compute the CC of the registered feature
        p = predicted[j0:j1, i0:i1]
        o = image[j0:j1, i0:i1]
        o = scipy.ndimage.shift(o, V, order=1, prefilter=False, mode="nearest")
        cc[index] = np.corrcoef(p.flatten(), o.flatten())[0, 1]

        # Update the position
        Vx[index] = -V[0]
        Vy[index] = -V[1]

    # Set the updated points
    data_image[0, select, 0] = X + Vx
    data_image[0, select, 1] = Y + Vy

    # Select only those points whose cc is greater than a given value
    Q1, Q3 = np.quantile(cc, [0.25, 0.75])
    IQR = Q3 - Q1
    threshold = min(max_threshold, (Q1 - 1.5 * IQR))
    mask_image[0, select] = cc > threshold
    print(
        "Tracked %d / %d features with cc > %.2f and average shift of (%.1f, %.1f)"
        % (
            np.count_nonzero(mask_image),
            np.count_nonzero(select),
            threshold,
            np.mean(Vx),
            np.mean(Vy),
        )
    )

    # Return the data and mask
    return data_image, mask_image


def extend_contours_internal(projections, P, data, mask, octave, subset_size: int):
    """
    Try to extend the contours

    """

    # Check the subset size is atleast 1
    assert subset_size >= 1

    # Get the parameters
    dx = P[:, 0] + projections.shape[2] // 2
    dy = P[:, 1] + projections.shape[1] // 2
    a = np.radians(P[:, 2])
    b = np.radians(P[:, 3])
    c = np.radians(P[:, 4])

    # Triangulate the 3D points
    points = dragonET.command_line._contours_triangulate.triangulate(
        dx, dy, a, b, c, data, mask
    )

    # Copy input
    data = data.copy()
    mask = mask.copy()

    # Extend the contours for each image from adjacent images
    for j in range(data.shape[0]):
        for k, (i0, i1) in enumerate(
            [(j - subset_size, j), (j + 1, j + 1 + subset_size)]
        ):
            if i0 >= 0 and i1 <= data.shape[0]:
                print(
                    "Extending contours onto image %d from images %d to %d"
                    % (j, i0, i1)
                )
                data[j], mask[j] = extend_contours_for_image(
                    projections[i0:i1],
                    projections[j],
                    P[i0:i1],
                    P[j],
                    points,
                    data[i0:i1],
                    mask[i0:i1],
                    octave,
                )

    # Return the data and mask
    return data, mask


def _contours_extend(
    projections_in: str,
    model_in: str,
    contours_in: str,
    contours_out: str,
    subset_size: int,
):
    """
    Extend the contours

    """

    def read_projections(filename):
        print("Reading projections from %s" % filename)
        return mrcfile.mmap(filename).data

    def read_points(filename) -> tuple:
        print("Reading points from %s" % filename)
        handle = np.load(filename)
        return handle["data"], handle["mask"], handle["octave"]

    def read_model(filename) -> dict:
        print("Reading model from %s" % filename)
        return yaml.safe_load(open(filename, "r"))

    def write_points(filename, data, mask, octave):
        print("Writing contours to %s" % filename)
        np.savez(open(filename, "wb"), data=data, mask=mask, octave=octave)

    # Read the projections
    projections = read_projections(projections_in)

    # Read the model
    model = read_model(model_in)

    # Get the parameters
    P = np.array(model["transform"])

    # Read the points
    data, mask, octave = read_points(contours_in)

    # Try to extend the contours
    data, mask = extend_contours_internal(
        projections, P, data, mask, octave, subset_size
    )

    # Write the contours
    write_points(contours_out, data, mask, octave)


if __name__ == "__main__":
    contours_extend()
