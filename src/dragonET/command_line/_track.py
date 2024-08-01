#
# track.py
#
# Copyright (C) 2024 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
import time
from argparse import ArgumentParser
from collections import defaultdict
from typing import List

import mrcfile
import numpy as np
import yaml
from skimage.feature import SIFT, match_descriptors  # , plot_matches
from skimage.measure import ransac
from skimage.transform import EuclideanTransform

np.random.seed(0)
__all__ = ["track"]


def get_description():
    """
    Get the program description

    """
    return "Do a rough alignment of the projection images"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the track parser

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
        "--model_in",
        type=str,
        default=None,
        dest="model_in",
        required=True,
        help=(
            """
            A file describing the initial model.
            """
        ),
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="tracked_model.yaml",
        dest="model_out",
        help=(
            """
            A file describing the output model.
            """
        ),
    )
    parser.add_argument(
        "--contours",
        type=str,
        default="contours.npz",
        dest="contours_out",
        help=(
            """
            A binary file describing the contours.
            """
        ),
    )

    return parser


def track_impl(args):
    """
    Do a rough alignment of the projection images

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _track(
        args.projections_in,
        args.model_in,
        args.model_out,
        args.contours_out,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def track(args: List[str] = None):
    """
    Do a rough alignment of the projection images

    """
    track_impl(get_parser().parse_args(args=args))


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
        data = data.reshape(shape).sum(-1).sum(2)
    return data


def extract_features(projections, rebin_factor):
    # Get the rebin factor octave
    rebin_factor_octave = np.log2(rebin_factor).astype(int)

    # Initialise the SIFT algorithm
    descriptor_extractor = SIFT(
        upsampling=1, c_dog=0.009, n_scales=8, n_hist=16, n_ori=16
    )

    # Initialise the feature lookup
    features = []

    # For each image run the SIFT feature extractor
    for i in range(projections.shape[0]):
        # Extract the features
        descriptor_extractor.detect_and_extract(projections[i])

        # Add to the feature lookup. Here we save the positions in (x, y) order
        # rather than the (y, x) order that comes out of the descriptor. This
        # is because we want to do some matrix calculations and its easier to
        # keep track of if everything is consistent.
        features.append(
            {
                "keypoints": descriptor_extractor.positions[:, ::-1],  # * rebin_factor,
                "descriptors": descriptor_extractor.descriptors,
                "octaves": descriptor_extractor.octaves,  # + rebin_factor_octave,
                "orientations": descriptor_extractor.orientations,
            }
        )

        print(
            "Extracted %d features from image %d / %d"
            % (len(descriptor_extractor.positions), i + 1, projections.shape[0])
        )

    return features


def find_matching_features(features, min_samples=5):
    # Initialise the transformation matrix for each image
    matrix = np.full((len(features), 3, 3), np.eye(3))

    # Initialise the list of matches
    match_list = {}

    # Match features across adjacent images
    for i, j in enumerate(range(1, len(features))):
        # Match the features
        matches = match_descriptors(
            features[i]["descriptors"],
            features[j]["descriptors"],
            max_ratio=0.95,
            cross_check=True,
        )

        # Get the octaves
        octaves_i, octaves_j = (
            features[i]["octaves"][matches[:, 0]],
            features[j]["octaves"][matches[:, 1]],
        )

        # Select only those points which have matching octaves
        matches = matches[octaves_i == octaves_j, :]

        # Get the orientations
        orientations_i, orientations_j = (
            features[i]["orientations"][matches[:, 0]],
            features[j]["orientations"][matches[:, 1]],
        )

        # Compute the angular difference and select those with an angular
        # difference less than the number of divisions used to find the
        # orientation
        zn = np.exp(1j * (orientations_i - orientations_j))
        angle_diff = np.abs(np.angle(zn) - np.angle(np.median(zn)))
        select_orientation = angle_diff < (2 * np.pi / 16)

        # Select only those with matching orientations
        matches = matches[select_orientation, :]

        # Get the positions
        positions_i, positions_j = (
            features[i]["keypoints"][matches[:, 0]],
            features[j]["keypoints"][matches[:, 1]],
        )

        # Only bother if we have enough samples
        assert len(positions_i) >= min_samples

        # Compute the Euclidean transform
        transform, inliers = ransac(
            (positions_i, positions_j),
            EuclideanTransform,
            min_samples=min_samples,
            residual_threshold=5,
            max_trials=1000,
        )

        # Add the list of matches to a dictionary.
        for index in np.where(inliers)[0]:
            key_i = (i, matches[index, 0])
            key_j = (j, matches[index, 1])
            match_list[key_i] = key_j

        # Update the matrix
        matrix[j] = transform.params @ matrix[j - 1]

        print(
            "Matching images (%d, %d): fitted %d points out of %d matches from (%d, %d) features"
            % (
                i + 1,
                j + 1,
                np.count_nonzero(inliers),
                matches.shape[0],
                len(features[i]["keypoints"]),
                len(features[j]["keypoints"]),
            )
        )

    return matrix, match_list


def construct_data_matrix(features, match_list):
    # Check if the point is already part of an existing contour
    def find(key_i):
        for key, value in contours.items():
            if key_i in value:
                return key
        return None

    # Construct the contours
    contours = defaultdict(set)
    for key_i, key_j in match_list.items():
        key = find(key_i)
        if key is not None:
            contours[key].add(key_j)
        else:
            contours[key_i].add(key_j)

    # Compute the number of frames and points
    num_frames = len(features)
    num_points = len(contours)

    # Construct the data matrix and mask
    data = np.zeros((num_frames, num_points, 2))
    mask = np.zeros((num_frames, num_points), dtype=bool)
    octave = np.zeros(num_points, dtype=int)
    for index, key in enumerate(contours):
        octave[index] = features[key[0]]["octaves"][key[1]]
        for frame, feature_index in [key] + list(contours[key]):
            mask[frame, index] = True
            data[frame, index] = features[frame]["keypoints"][feature_index]
            assert octave[index] == features[frame]["octaves"][feature_index]

    # Select only those points which have 3 or more observations which is a
    # requirement to ensure that the point can be triangulated in 3D.
    select = np.count_nonzero(mask, axis=0) >= 3
    data = data[:, select]
    mask = mask[:, select]

    print(
        "Selected %d / %d points with >= 3 observations"
        % (np.count_nonzero(select), select.size)
    )

    # Return the data matrix and mask
    return data, mask, octave


def recentre_points(data, mask, matrix, origin=(0, 0)):
    # Compute the initial translation which brings the points on each image
    # closest to the origin
    Ox, Oy = origin
    A = []
    B = []
    for j in range(data.shape[0]):
        for i in range(data.shape[1]):
            if mask[j, i]:
                M00, M01, M02 = matrix[j, 0, :]
                M10, M11, M12 = matrix[j, 1, :]
                Rx = data[j, i, 0] - M00 * Ox - M01 * Oy - M02
                Ry = data[j, i, 1] - M10 * Ox - M11 * Oy - M12
                B.append(Rx * M00 + Ry * M01)
                B.append(Rx * M10 + Ry * M11)
                A.append((M00**2 + M10 * M01, M01 * M00 + M11 * M01))
                A.append((M00 * M10 + M10 * M11, M01 * M10 + M11**2))
    A = np.array(A)
    B = np.array(B)
    t0 = np.linalg.inv(A.T @ A) @ A.T @ B

    # Construct the translation matrix
    translation = np.array([[1, 0, t0[0]], [0, 1, t0[1]], [0, 0, 1]])

    # Apply the translation
    matrix = matrix @ translation

    # Return the matrix
    return matrix


def construct_model(matrix, P0):
    # Get the translations
    dx = matrix[:, 0, 2]
    dy = matrix[:, 1, 2]

    # Get the rotation component
    a = np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0])
    b = P0[:, 1]
    c = P0[:, 2]

    # Return the model
    return np.stack([a, b, c, dy, dx], axis=1)


def track_stack(
    projections,
    P,
    rebin_factor=8,
    min_samples=5,
):
    """
    Do the alignment

    """

    def rescale(a):
        s1 = 1 / (np.max(a) - np.min(a))
        s0 = -s1 * a.min()
        return s0 + s1 * a

    # Rebin the projections to a smaller size. We do this for two reasons.
    # First of all, the SIFT algorithm picks out too much noise if we do at at
    # full size. Experience shows that rebinning 8 times works better.
    # Secondly, it is much faster for the initial rounds of alignment to rebin
    # in this way. We also rescale the data so that the pixel values are
    # floating point values between 0 and 1 because the SIFT algorithm expects
    # the data like this and will not work well otherwise.
    rebinned_projections = rescale(rebin_stack(projections, rebin_factor))

    # Extract the image features
    features = extract_features(rebinned_projections, rebin_factor)

    # Find matching features and compute initial transform between images
    matrix, match_list = find_matching_features(features, min_samples)

    # Construct data matrix. This is a FxPx2 matrix containing the coordinates
    # of all points on all frames. The mask is a FxP matrix showing whether the
    # point has been measured on that frame. This is a more convenient
    # representation for performing the refinement.
    data, mask, octave = construct_data_matrix(features, match_list)

    # Recentre the points around the origin. This calculates the optimal matrix
    # that puts the points around the origin on each image.
    matrix = recentre_points(data, mask, matrix, origin=(0, 0))

    # Construct the model parameters
    P = construct_model(matrix, P)

    # Return contours and the model parameters
    return data, mask, octave, P


def _track(
    projections_in: str,
    model_in: str,
    model_out: str,
    contours_out: str,
):
    """
    Do the alignment

    """

    def read_projections(filename):
        print("Reading projections from %s" % filename)
        return mrcfile.mmap(filename).data

    def read_model(filename) -> dict:
        print("Reading model from %s" % filename)
        return yaml.safe_load(open(filename, "r"))

    def write_model(model, filename):
        print("Writing model to %s" % filename)
        yaml.safe_dump(model, open(filename, "w"), default_flow_style=None)

    def write_contours(filename, data, mask, octave):
        print("Writing contours to %s" % filename)
        np.savez(open(filename, "wb"), data=data, mask=mask, octave=octave)

    # Read the projections
    projections = read_projections(projections_in)

    # Read the model
    model = read_model(model_in)

    # Read the initial model and convert degrees to radians for use here
    P = np.array(model["transform"], dtype=float)

    # Extract some contours
    data, mask, octave, P = track_stack(projections, P)

    # Update the model and convert back to degrees
    model["transform"] = P.tolist()

    # Write the model to file
    write_model(model, model_out)
    write_contours(contours_out, data, mask, octave)


if __name__ == "__main__":
    track()