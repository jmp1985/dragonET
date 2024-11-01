#
# contours_refine.py
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
from skimage.measure import ransac
from skimage.transform import EuclideanTransform

import dragonET.command_line

__all__ = ["contours_refine"]


def get_description():
    """
    Get the program description

    """
    return "Refine the contours to match features better across images"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the contours_refine parser

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
        "--model_out",
        type=str,
        default="refined_model.yaml",
        dest="model_out",
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
        default="refined.npz",
        dest="contours_out",
        help=(
            """
            A YAML file describing the refined model.
            """
        ),
    )

    return parser


def contours_refine_impl(args):
    """
    Extend the contours

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _contours_refine(
        args.projections_in,
        args.model_in,
        args.model_out,
        args.contours_in,
        args.contours_out,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def contours_refine(args: List[str] = None):
    """
    Extend the contours

    """
    contours_refine_impl(get_parser().parse_args(args=args))


def _refine_model(P, data, mask, image_size):
    """
    Refine the geometric model

    """

    # Get the parameters
    dx = (P[:, 0] + 0.5) * image_size[1]  # Scale by image size
    dy = (P[:, 1] + 0.5) * image_size[0]  # Scale by image size
    a = np.radians(P[:, 2])
    b = np.radians(P[:, 3])
    c = np.radians(P[:, 4])

    idx = np.argmin(np.abs(c))

    active = np.zeros(P.shape, dtype=bool).T
    active[0, :] = 1  # dx
    active[1, :] = 1  # dy
    active[2, :] = 1  # a
    active[2, idx] = 0  # a

    select = np.count_nonzero(mask, axis=0) >= 3
    mask = mask[:, select]
    data = data[:, select]

    dx, dy, a, b, c, rmsd = dragonET.command_line._refine.refine_model(
        dx, dy, a, b, c, data, mask, active=active
    )

    active[3, :] = 1  # b
    active[3, idx] = 0  # b

    dx, dy, a, b, c, rmsd = dragonET.command_line._refine.refine_model(
        dx, dy, a, b, c, data, mask, active=active
    )

    dx = (dx / image_size[1]) - 0.5
    dy = (dy / image_size[0]) - 0.5
    P = np.stack([dx, dy, np.degrees(a), np.degrees(b), np.degrees(c)], axis=1)
    return P


def _predict_image(projections, P, P_image):
    def _get_matrix_from_parameters(P):
        return dragonET.command_line._stack_predict.get_matrix_from_parameters(P)

    def _get_parameters_from_matrix(R):
        return dragonET.command_line._stack_predict.get_parameters_from_matrix(R)

    def _get_matrix(P, image_size):
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

    def _transform_stack(projections, P, P_image):
        # Get matrices for the image
        R_data = _get_matrix_from_parameters(P)
        R_image = _get_matrix_from_parameters(P_image[None, :])

        # Rotate all the input images w.r.t the output image
        R_data = R_data @ np.linalg.inv(R_image)

        # Get the updated parameter arrays
        P = _get_parameters_from_matrix(R_data)

        # Get transform matrix
        matrix = _get_matrix(P, projections.shape[1:])

        # Transform image
        return np.mean(
            dragonET.command_line._stack_transform.transform_stack(projections, matrix),
            axis=0,
        )

    I = _transform_stack(projections, P, P_image)
    M = _transform_stack(np.ones_like(projections), P, P_image)
    M = np.isclose(M, 1, atol=0.1)
    return I, M


def _predict_coordinates(data, mask, P, P_image, image_size):
    def _get_matrix_from_parameters(P):
        return dragonET.command_line._stack_predict.get_matrix_from_parameters(P)

    def _get_parameters_from_matrix(R):
        return dragonET.command_line._stack_predict.get_parameters_from_matrix(R)

    def _get_matrix(P, image_size):
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

    def _transform_coordinates(data, mask, P, P_image, image_size):
        # Get matrices for the image
        R_data = _get_matrix_from_parameters(P[None, :])
        R_image = _get_matrix_from_parameters(P_image[None, :])

        # Rotate all the input images w.r.t the output image
        R_data = R_data @ np.linalg.inv(R_image)

        # Get the updated parameter arrays
        P = _get_parameters_from_matrix(R_data)

        # Get transform matrix
        matrix = _get_matrix(P, image_size)[0]
        matrix = np.linalg.inv(matrix)

        X = data[mask, 0]
        Y = data[mask, 1]
        X, Y = (
            matrix[0, 0] * X + matrix[0, 1] * Y + matrix[0, 2],
            matrix[1, 0] * X + matrix[1, 1] * Y + matrix[1, 2],
        )

        data_return = data.copy()
        data_return[mask, 0] = X
        data_return[mask, 1] = Y
        return data_return

    return _transform_coordinates(data, mask, P, P_image, image_size)


def compute_derivatives(predicted, image):
    """
    Compute the image derivatives

    """
    stencil = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
    Ix = scipy.ndimage.convolve(predicted, stencil[None, :], mode="nearest")
    Iy = scipy.ndimage.convolve(predicted, stencil[:, None], mode="nearest")
    It = image - predicted
    return Ix, Iy, It


def compute_optical_flow(Ix, Iy, It, Im):
    # Compute the weights (2.3 approximates hamming)
    Wx = scipy.signal.windows.gaussian(Ix.shape[1], (Ix.shape[1] / 2) / (2.3))
    Wy = scipy.signal.windows.gaussian(Iy.shape[0], (Iy.shape[0] / 2) / (2.3))
    W = Wx[None, :] * Wy[:, None]
    W = (Im * np.sqrt(W)).flatten()

    # Compute the optical flow
    A = np.stack([Ix.flatten(), Iy.flatten()]).T * W[:, None]
    B = -It.flatten() * W
    V = np.linalg.pinv(A.T @ A) @ (A.T @ B)
    return V


def _propagate(predicted, observed, image_mask, octave, data_initial):
    image_size = np.array(predicted.shape)

    # Compute the derivatives
    Ix, Iy, It = compute_derivatives(predicted, observed)
    Im = image_mask

    # Initialise the observed positions
    data_observed = np.zeros_like(data_initial)
    mask_observed = np.zeros_like(data_initial, dtype=bool)

    # Compute the sizes
    size = 16 * 2**octave

    # Get the X and Y
    X = data_initial[:, 0]
    Y = data_initial[:, 1]

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
        i1 = int(min(image_size[1], np.ceil(i0 + size[index])))
        j0 = int(max(0, np.floor(yc - size[index] // 2)))
        j1 = int(min(image_size[0], np.ceil(j0 + size[index])))
        if i0 >= i1 or j0 >= j1:
            continue

        # Get the predicted sub set of the image
        predicted_j = predicted[j0:j1, i0:i1]
        Ixj = Ix[j0:j1, i0:i1]
        Iyj = Iy[j0:j1, i0:i1]
        Itj = It[j0:j1, i0:i1]
        Imj = Im[j0:j1, i0:i1]
        Yj, Xj = np.mgrid[j0:j1, i0:i1]

        V = np.zeros(2)
        for it in range(10):
            # Compute the optical flow to register the feature
            dV = -compute_optical_flow(Ixj, Iyj, Itj, Imj)

            V += dV

            shifted_j = scipy.ndimage.map_coordinates(
                observed,
                np.stack([Yj + V[1], Xj + V[0]]),
                prefilter=False,
                mode="nearest",
                order=1,
            )

            Itj = shifted_j - predicted_j

            if np.sqrt(np.sum(dV**2)) < 1e-3:
                break

        # Compute the CC of the registered feature
        p = predicted_j[Imj].flatten()
        o = shifted_j[Imj].flatten()
        if p.size > 0 and o.size > 0 and np.std(p) > 0 and np.std(o) > 0:
            c = np.corrcoef(p, o)[0, 1]
            assert not np.isnan(c)
            cc[index] = c
        else:
            cc[index] = 0

        # if True:#p.size > 0 and cc[index] < 0.5:
        #     print(V)
        #     fig, ax = pylab.subplots(ncols=4)
        #     ax[0].imshow(Imj)
        #     ax[1].imshow(predicted_j)
        #     ax[2].imshow(shifted_j)
        #     ax[3].imshow(predicted_j - shifted_j)
        #     pylab.show()

        # Update the position
        Vx[index] = V[0]
        Vy[index] = V[1]

    # Set the updated points
    data_observed[:, 0] = X + Vx
    data_observed[:, 1] = Y + Vy

    # Select only those points whose cc is greater than a given value
    Q1, Q3 = np.quantile(cc, [0.25, 0.75])
    IQR = Q3 - Q1
    threshold_limits = (0.8, 0.9)
    threshold = np.clip(Q1 - 1.5 * IQR, *threshold_limits)
    mask_observed = cc > 0.5  # threshold
    print(
        "Tracked %d / %d features with cc > %.2f and average shift of (%.1f, %.1f)"
        % (
            np.count_nonzero(mask_observed),
            mask_observed.size,
            threshold,
            np.mean(Vx),
            np.mean(Vy),
        )
    )

    # Return the data and mask
    return data_observed, mask_observed


def _validate(data_predicted, data_observed, mask, P, image_size):
    positions_dst = data_observed[mask] / image_size[::-1]
    positions_src = data_predicted[mask] / image_size[::-1]
    min_samples = 4

    # Compute the Euclidean transform
    transform, inliers = ransac(
        (positions_src, positions_dst),
        EuclideanTransform,
        min_samples=min_samples,
        residual_threshold=2 / 512,  # 0.01,
        max_trials=1000,
    )
    print(
        "Selecting %d/%d points as inliers with rotation of %.2f (deg) and x/y translation of (%.2f, %.2f)"
        % (
            np.count_nonzero(inliers),
            inliers.size,
            np.degrees(transform.rotation),
            transform.translation[0] * image_size[1],
            transform.translation[1] * image_size[0],
        )
    )

    indices = np.where(mask)[0][inliers]
    mask = mask.copy()
    mask[indices] = 1

    # R = Rotation.from_euler("yxz", [0, 0, transform.rotation]).as_matrix()
    # t = transform.translation
    # M = np.eye(4)
    # M[:3, :3] = R
    # M[:2, 3] = t

    # R0 = get_matrix_from_parameters(P[None, :])
    # M = M @ R0
    # P = get_parameters_from_matrix(M)[0]
    return mask


def _refine_contours(projections, P, data, mask, octave):
    # Get the image size
    image_size = np.array(projections.shape[1:])

    # Only select points with enough correspondences
    select = np.count_nonzero(mask, axis=0) >= 2
    data = data[:, select]
    mask = mask[:, select]
    octave = octave[select]

    # The order of images (skip zero)
    order = np.argsort(np.abs(P[:, 4]))[1:]

    # Select each image sequentially
    for index in order:
        # Select the frames to predict from
        if index > 0 and np.abs(P[index - 1, 4]) < np.abs(P[index, 4]):
            reference = index - 1
        elif index < P.shape[0] - 1 and np.abs(P[index + 1, 4]) < np.abs(P[index, 4]):
            reference = index + 1

        # Print some information
        print(
            "Aligning image %d (%.1f deg) to image %d (%.1f deg)"
            % (index, P[index, 4], reference, P[reference, 4])
        )

        # Get the current image
        observed = projections[index]

        # Predict the image
        predicted, predicted_image_mask = _predict_image(
            projections[reference][None, :, :], P[reference][None, :], P[index]
        )

        # Predict the coordinate
        data_predicted = _predict_coordinates(
            data[reference], mask[reference], P[reference], P[index], image_size
        )

        # Select only those features recorded on this image and the reference
        point_selection = mask[index, :] & mask[reference, :]

        # Propagate contours from model to image
        data_observed, mask_observed = _propagate(
            predicted,
            observed,
            predicted_image_mask,
            octave[point_selection],
            data_predicted[point_selection],
        )

        # Perform geometric validation of points
        mask_observed = _validate(
            data_predicted[point_selection],
            data_observed,
            mask_observed,
            P[index],
            image_size,
        )

        # Set the mask and data
        data[index, point_selection] = data_observed
        mask[index, point_selection] = mask_observed

    return data, mask, octave


def process(projections, P, data, mask, octave):
    """
    Try to extend the contours

    """

    # Get the image size
    image_size = np.array(projections.shape[1:])

    # Scale the data to pixel coordinates
    data = data * image_size[None, None, ::-1]  # Scale by image size

    # Refine the geometric model
    P = _refine_model(P, data, mask, image_size)

    # Refine the contours
    data, mask, octave = _refine_contours(projections, P, data, mask, octave)

    # Normalise coords
    data = data / image_size[None, None, ::-1]  # Scale by image size

    # Return the contours and model
    return P, data, mask, octave


def _contours_refine(
    projections_in: str,
    model_in: str,
    model_out: str,
    contours_in: str,
    contours_out: str,
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

    def write_model(model, filename):
        print("Writing model to %s" % filename)
        yaml.safe_dump(model, open(filename, "w"), default_flow_style=None)

    # Read the projections
    projections = read_projections(projections_in)

    # Read the model
    model = read_model(model_in)

    # Get the parameters
    P = np.array(model["transform"])

    # Read the points
    data, mask, octave = read_points(contours_in)
    assert data.shape[0] == mask.shape[0]
    assert data.shape[1] == mask.shape[1]
    assert data.shape[1] == octave.shape[0]

    # Try to extend the contours
    for it in range(10):
        # Only select points with enough correspondences
        select = np.count_nonzero(mask, axis=0) >= 2
        data = data[:, select]
        mask = mask[:, select]
        octave = octave[select]

        # Do the processing
        P, data, mask, octave = process(projections, P, data, mask, octave)

    # Write the contours
    write_points(contours_out, data, mask, octave)

    # Save the refined model
    model["transform"] = P.tolist()
    write_model(model, model_out)


if __name__ == "__main__":
    contours_refine()
