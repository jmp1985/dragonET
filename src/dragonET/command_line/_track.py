#
# track.py
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
from skimage.feature import SIFT, match_descriptors  # , plot_matches

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
        "--contours",
        type=str,
        default="contours.yaml",
        dest="contours_out",
        help=(
            """
            A YAML file describing the contours.
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
        args.contours_out,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def track(args: List[str] = None):
    """
    Do a rough alignment of the projection images

    """
    track_impl(get_parser().parse_args(args=args))


def rebin(data, shape):
    """
    Rebin a multidimensional array

    Args:
        data (array): The input array
        shape (tuple): The new shape

    """
    shape = (
        shape[0],
        data.shape[0] // shape[0],
        shape[1],
        data.shape[1] // shape[1],
    )
    output = data.reshape(shape).sum(-1).sum(1)
    return output


def track_stack(projections, P):
    """
    Do the alignment

    """

    def reject_outliers(data, m=2):
        # Calculate median
        median = np.median(data, axis=0)

        # Calculate median absolute deviation (MAD)
        mad = np.median(np.abs(data - median[None, :]), axis=0)

        # Define upper and lower bounds for outliers
        upper_bound = median + m * mad
        lower_bound = median - m * mad

        # Return data without outliers
        select = (
            (lower_bound[0] <= data[:, 0])
            & (data[:, 0] <= upper_bound[0])
            & (lower_bound[1] <= data[:, 1])
            & (data[:, 1] <= upper_bound[1])
        )
        return select

    contours = []
    points = {}
    points_number = 0
    for i in range(0, projections.shape[0] - 1):
        i1 = i
        i2 = i1 + 1
        N = 8

        img1 = projections[i1, :, :]  # Query image
        img2 = projections[i2, :, :]  # Train image

        img1 = rebin(img1, np.array(img1.shape) // N)
        img2 = rebin(img2, np.array(img2.shape) // N)

        img1 = img1.astype("float32")
        img2 = img2.astype("float32")

        descriptor_extractor = SIFT(upsampling=1.0, c_dog=500)

        descriptor_extractor.detect_and_extract(img1)
        keypoints1 = descriptor_extractor.keypoints
        descriptors1 = descriptor_extractor.descriptors

        descriptor_extractor.detect_and_extract(img2)
        keypoints2 = descriptor_extractor.keypoints
        descriptors2 = descriptor_extractor.descriptors

        shift0 = (P[i1, 3:5] - P[i2, 3:5]) / N

        descriptors1 = np.concatenate([keypoints1, descriptors1], axis=1)
        descriptors2 = np.concatenate(
            [(keypoints2 - np.array(shift0)[None, :]), descriptors2], axis=1
        )

        matches12 = match_descriptors(
            descriptors1, descriptors2, max_ratio=0.6, cross_check=True
        )

        shifts = []
        for i, j in matches12:
            pi = keypoints1[i]
            pj = keypoints2[j]
            shift = pj - pi
            shifts.append(shift)
        shifts = np.array(shifts)

        select = reject_outliers(shifts)

        for index in np.where(select)[0]:
            index1, index2 = matches12[index]
            if (i1, index1) in points:
                number = points[(i1, index1)]
            else:
                number = points_number
                points_number += 1
            points[(i1, index1)] = number
            points[(i2, index2)] = number
            contours.append(
                (
                    number,
                    int(i1),
                    float(keypoints1[index1, 0] * N),
                    float(keypoints1[index1, 1] * N),
                )
            )
            contours.append(
                (
                    number,
                    int(i2),
                    float(keypoints2[index2, 0] * N),
                    float(keypoints2[index2, 1] * N),
                )
            )

        shifts = shifts[select, :]
        shift = np.mean(shifts, axis=0)
        print(i1, np.sqrt(np.sum((shift - shift0) ** 2)))

        # fig, ax = pylab.subplots()
        # ax.hist(shifts[:,0])
        # ax.hist(shifts[:,1])
        # pylab.show()

        # fig, ax = pylab.subplots(figsize=(11, 8))

        # pylab.gray()

        # plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12)
        # ax.axis('off')
        # ax.set_title("Original Image vs. Flipped Image\n" "(all keypoints and matches)")

        # pylab.tight_layout()
        # pylab.show()

    contours = sorted(contours)

    return contours


def _track(
    projections_in: str,
    model_in: str,
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

    def write_contours(filename, contours):
        print("Writing contours to %s" % filename)
        yaml.safe_dump(contours, open(filename, "w"), default_flow_style=None)

    # Read the projections
    projections = read_projections(projections_in)

    # Read the model
    model = read_model(model_in)

    # Read the initial model and convert degrees to radians for use here
    P = np.array(model["transform"], dtype=float)

    # Extract some contours
    contours = track_stack(projections, P)

    # # Update the model and convert back to degrees
    # model["transform"] = P.tolist()

    # # Write the model to file
    # write_model(model, model_out)
    write_contours(contours_out, {"contours": contours})


if __name__ == "__main__":
    track()
