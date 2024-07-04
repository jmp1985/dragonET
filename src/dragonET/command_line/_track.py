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

# import scipy.ndimage
import skimage.filters
import yaml

# from matplotlib import pylab
from skimage.feature import SIFT, match_descriptors  # , plot_matches
from skimage.measure import ransac

# from skimage.morphology import disk
# from skimage.restoration import (
#     denoise_bilateral,
#     denoise_nl_means,
#     denoise_tv_chambolle,
#     estimate_sigma,
# )
from skimage.transform import EuclideanTransform  # , warp

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


def track_stack(
    projections,
    P,
    rebin_factor=8,
    min_samples=5,
    match_adjacent_images=3,
    track_adjacent_images=3,
):
    """
    Do the alignment

    """

    def is_power_of_2(n):
        return (n & (n - 1) == 0) and n != 0

    def rescale(a):
        s1 = 1 / (np.max(a) - np.min(a))
        s0 = -s1 * a.min()
        return s0 + s1 * a

    # Initialise the SIFT algorithm
    descriptor_extractor = SIFT(
        upsampling=1, c_dog=0.009, n_scales=8, n_hist=16, n_ori=16
    )

    # Check rebin factor
    assert is_power_of_2(rebin_factor)

    # Set the rebin shape
    rebin_shape = np.array(projections.shape[1:]) // rebin_factor

    # Get the rebin factor octave
    rebin_factor_octave = np.log2(rebin_factor).astype(int)

    # Rebin the stack
    stack = np.zeros((projections.shape[0], rebin_shape[0], rebin_shape[1]))
    for i in range(projections.shape[0]):
        print("Rebinning image %d" % (i + 1))
        stack[i, :, :] = rescale(rebin(projections[i, :, :], rebin_shape).astype(float))

    # Initialise the list of features
    features = []

    # For each image run the SIFT feature extractor
    for i in range(projections.shape[0]):
        # Rebin the image and then rescale between 0 and 1
        image = rescale(rebin(projections[i, :, :], rebin_shape).astype(float))

        # Extract the features
        descriptor_extractor.detect_and_extract(image)

        # Add to the feature lookup
        features.append(
            {
                "keypoints": descriptor_extractor.positions * rebin_factor,
                "descriptors": descriptor_extractor.descriptors,
                "octaves": descriptor_extractor.octaves + rebin_factor_octave,
                "orientations": descriptor_extractor.orientations,
            }
        )

        print(
            "Extracted %d features from image %d / %d"
            % (len(descriptor_extractor.positions), i + 1, projections.shape[0])
        )

    # Initialise the list of matches
    match_list = defaultdict(set)

    # Match features across adjacent images
    for i in range(len(features) - 1):
        for j in range(i + 1, min(i + match_adjacent_images + 1, len(features))):
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
            if len(positions_i) >= min_samples:
                # Compute the Euclidean transform
                transform, inliers = ransac(
                    (positions_j, positions_i),
                    EuclideanTransform,
                    min_samples=min_samples,
                    residual_threshold=3 * rebin_factor,
                    max_trials=1000,
                )

                # Only proceed if we have some inliers
                if inliers is not None and np.count_nonzero(inliers) >= 5:
                    # Add the list of matches to a dictionary. We choose the key to
                    # be lower match index. But we check to see if that key is
                    # already a value for a previous key. This is so that we can
                    # check the matches for consistency later
                    for index in np.where(inliers)[0]:
                        key_i = (i, matches[index, 0])
                        key_j = (j, matches[index, 1])
                        match_list[key_i].add(key_j)
                        match_list[key_j].add(key_i)

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

    # For each contour predict the position for the next N frames
    for key in sorted(match_list.keys()):
        contour = sorted([key] + list(match_list[key]))
        image = list([c[0] for c in contour])
        zmin = max(0, min(image) - track_adjacent_images)
        zmax = min(projections.shape[0], max(image) + track_adjacent_images + 1)
        for z in range(zmin, zmax):
            if z not in image:
                ind = np.argmin(np.abs(np.array(image) - z))
                k = contour[ind]

                position0 = features[k[0]]["keypoints"][k[1]]
                descriptor = features[k[0]]["descriptors"][k[1]]
                octave = features[k[0]]["octaves"][k[1]]
                orientation = features[k[0]]["orientations"][k[1]]

                position = position0 - (P[z, 3:] - P[k[0], 3:])
                if np.all(position > 0) and np.all(
                    position < np.array(projections.shape[1:])
                ):
                    feature_index = features[z]["keypoints"].shape[0]
                    features[z]["keypoints"] = np.append(
                        features[z]["keypoints"], position[None, :], axis=0
                    )
                    features[z]["descriptors"] = np.append(
                        features[z]["descriptors"], descriptor[None, :], axis=0
                    )
                    features[z]["octaves"] = np.append(features[z]["octaves"], octave)
                    features[z]["orientations"] = np.append(
                        features[z]["orientations"], orientation
                    )
                    match_list[key].add((z, feature_index))

    def region(x):
        class Slicer:
            def __getitem__(self, item):
                yslice, xslice = item
                ysize = yslice.stop - yslice.start
                xsize = xslice.stop - xslice.start
                yslice2 = slice(max(0, yslice.start), min(x.shape[0], yslice.stop))
                xslice2 = slice(max(0, xslice.start), min(x.shape[1], xslice.stop))
                yslice3 = slice(
                    yslice2.start - yslice.start, ysize - yslice.stop + yslice2.stop
                )
                xslice3 = slice(
                    xslice2.start - xslice.start, xsize - xslice.stop + xslice2.stop
                )
                data = np.zeros((ysize, xsize))
                mask = np.zeros((ysize, xsize), dtype=int)
                data[yslice3, xslice3] = x[yslice2, xslice2]
                mask[yslice3, xslice3] = 1
                return data, mask

        return Slicer()

    for key in sorted(match_list.keys()):
        contour = sorted([key] + list(match_list[key]))
        if len(contour) < 3:
            print("Skipping (%d/%d)" % (key))
        else:
            print("Fitting %d templates for (%d, %d)" % (len(contour), key[0], key[1]))

            # Create the template stack and mask
            octave = features[key[0]]["octaves"][key[1]]
            size = 16 * (2**octave)
            template_data = np.zeros((len(contour), size, size))
            template_mask = np.ones((size, size), dtype=int)

            # Get the image from the contour
            image = np.array([c[0] for c in contour])

            # Get the order relative to the key image
            order = np.argsort(np.abs(image - key[0]))

            # Loop through the images in order and register each template
            for i in order:
                # Get the image and feature index
                image_index, feature_index = contour[i]

                # Get the feature position and extent
                yc, xc = features[image_index]["keypoints"][feature_index]
                y0 = np.floor(yc - size // 2).astype(int)
                x0 = np.floor(xc - size // 2).astype(int)
                y1 = y0 + size
                x1 = x0 + size

                # If this is the key image just grab the template, otherwise do
                # the registration
                if image_index == key[0]:
                    template, mask = region(projections[image_index])[y0:y1, x0:x1]
                    # pylab.imshow(template)
                    # pylab.show()
                else:
                    # Get the search template
                    ys0 = y0 - size // 2  # 4*2**octave
                    xs0 = x0 - size // 2  # 4*2**octave
                    ys1 = y1 + size // 2  # 4*2**octave
                    xs1 = x1 + size // 2  # 4*2**octave
                    search_image, mask = region(projections[image_index])[
                        ys0:ys1, xs0:xs1
                    ]
                    # search_mask = mask & template_mask

                    # Get the template
                    template = np.sum(template_data, axis=0)
                    # fig, ax = pylab.subplots(ncols=3)
                    # ax[0].imshow(search_image)
                    # ax[1].imshow(template)

                    # Do the registration
                    result = skimage.feature.match_template(search_image, template)
                    dy, dx = np.floor(
                        np.unravel_index(np.argmax(result), result.shape)
                        - np.array(result.shape) / 2
                    ).astype(int)
                    y0, y1 = y0 + dy, y1 + dy
                    x0, x1 = x0 + dx, x1 + dx
                    yc, xc = yc + dy, xc + dx

                    # Set the new position
                    features[image_index]["keypoints"][feature_index] = (yc, xc)

                    # Get the template and mask
                    template, mask = region(projections[image_index])[y0:y1, x0:x1]

                    # print(image_index)
                    # ax[2].imshow(template)
                    # pylab.show()

                # Set the template data and mask
                template_data[i, :, :] = template
                template_mask = template_mask & mask

            # pylab.imshow(np.sum(template_data, axis=0))
            # pylab.show()

    # Construct the contours
    contours = []
    contour_index = 0
    for key in sorted(match_list.keys()):
        contour = sorted([key] + list(match_list[key]))
        if len(contour) >= 3:
            for image_index, feature_index in contour:
                position = features[image_index]["keypoints"][feature_index]
                octave = features[image_index]["octaves"][feature_index]
                contours.append(
                    (
                        int(contour_index),
                        int(image_index),
                        float(position[0]),
                        float(position[1]),
                        int(octave),
                    )
                )
            contour_index += 1

    # Make contour array
    contours = np.array(
        [tuple(x) for x in contours],
        dtype=[
            ("index", "int"),
            ("z", "int"),
            ("y", "float"),
            ("x", "float"),
            ("octave", "int"),
        ],
    )

    # P0 = P
    # P = P0.copy()
    # contours = []
    # points = {}
    # points_number = 0
    # for i in range(0, projections.shape[0] - 1):
    #     i1 = i
    #     for j in range(i1 + 1, min(projections.shape[0], i1 + 2)):
    #         i2 = j
    #         N = 8

    #         img1 = projections[i1, :, :]  # Query image
    #         img2 = projections[i2, :, :]  # Train image

    #         img1 = img1.astype("float32")
    #         img2 = img2.astype("float32")

    #         img1 = rescale(img1)
    #         img2 = rescale(img2)

    #         # img1 = denoise_tv_chambolle(img1, weight=10.0)
    #         # img2 = denoise_tv_chambolle(img2, weight=10.0)
    #         # img1 = denoise_bilateral(img1)
    #         # img2 = denoise_bilateral(img2)
    #         # sigma_est = np.mean(estimate_sigma(img1))
    #         # img1 = denoise_nl_means(img1,  h=1.15 * sigma_est, sigma=sigma_est, fast_mode=True)
    #         # img2 = denoise_nl_means(img2,  h=1.15 * sigma_est, sigma=sigma_est, fast_mode=True)

    #         img1 = rebin(img1, np.array(img1.shape) // N)
    #         img2 = rebin(img2, np.array(img2.shape) // N)

    #         # img1 = skimage.filters.gabor(img1, frequency=0.6)
    #         # img2 = skimage.filters.gabor(img2, frequency=0.6)
    #         # pylab.imshow(img1)
    #         # pylab.show()
    #         # img1 = skimage.filters.median(img1, disk(5))
    #         # img2 = skimage.filters.median(img2, disk(5))
    #         # img1 = scipy.ndimage.uniform_filter(img1, size=N)
    #         # img2 = scipy.ndimage.uniform_filter(img2, size=N)

    #         # img1 = scipy.ndimage.zoom(img1, 1/N)
    #         # img2 = scipy.ndimage.zoom(img2, 1/N)
    #         # print(img1.shape)

    #         # img1 = np.log(img1)
    #         # img2 = np.log(img2)

    #         img1 = rescale(img1)
    #         img2 = rescale(img2)

    #         # img2 = skimage.exposure.match_histograms(img2, img1)
    #         # img1 = skimage.exposure.adjust_log(img1)
    #         # img2 = skimage.exposure.adjust_log(img2)
    #         # img1 = rescale(img1)
    #         # img2 = rescale(img2)

    #         # img1 = skimage.exposure.equalize_adapthist(img1, 256)
    #         # img2 = skimage.exposure.equalize_adapthist(img2, 256)

    #         descriptor_extractor = SIFT(
    #             upsampling=1, c_dog=0.009, n_scales=8, n_hist=16, n_ori=16
    #         )  # , c_dog=500)
    #         # descriptor_extractor = SIFT(upsampling=1, n_scales=3, n_octaves=8, c_dog=0.005)#, c_dog=500)

    #         descriptor_extractor.detect_and_extract(img1)
    #         keypoints1 = descriptor_extractor.keypoints
    #         descriptors1 = descriptor_extractor.descriptors

    #         descriptor_extractor.detect_and_extract(img2)
    #         keypoints2 = descriptor_extractor.keypoints
    #         descriptors2 = descriptor_extractor.descriptors

    #         shift0 = (P0[i2, 3:5] - P0[i1, 3:5]) / N

    #         # descriptors1 = np.concatenate([keypoints1, descriptors1], axis=1)
    #         # descriptors2 = np.concatenate(
    #         #     [(keypoints2 - np.array(shift0)[None, :]), descriptors2], axis=1
    #         # )

    #         matches12 = match_descriptors(
    #             descriptors1, descriptors2, max_ratio=0.95, cross_check=True
    #         )

    #         matches1, matches2 = (
    #             keypoints1[matches12[:, 0]],
    #             keypoints2[matches12[:, 1]],
    #         )
    #         transform, inliers = ransac(
    #             (matches2, matches1),
    #             EuclideanTransform,
    #             min_samples=5,
    #             residual_threshold=3,
    #             max_trials=1000,
    #         )
    #         assert np.sum(inliers) >= 5
    #         # print(
    #         #     np.sum(inliers),
    #         #     inliers.size,
    #         #     np.degrees(transform.rotation),
    #         #     transform.translation - shift0,
    #         # )

    #         # transform_robust = transform(rotation = transform_robust.rotation) + transform(translation = -flip(transform_robust.translation))

    #         # fig, ax = pylab.subplots(figsize=(11, 8))

    #         # pylab.gray()

    #         # plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12[inliers])
    #         # ax.axis('off')
    #         # ax.set_title("Original Image vs. Flipped Image\n" "(all keypoints and matches)")

    #         # pylab.tight_layout()
    #         # pylab.show()

    #         # shifts = []
    #         # for i, j in matches12:
    #         #     pi = keypoints1[i]
    #         #     pj = keypoints2[j]
    #         #     shift = pj - pi
    #         #     shifts.append(shift)
    #         # shifts = np.array(shifts)

    #         # select = reject_outliers(shifts)

    #         for index in np.where(inliers)[0]:
    #             index1, index2 = matches12[index]
    #             if (i1, index1) in points:
    #                 number = points[(i1, index1)]
    #             else:
    #                 number = points_number
    #                 points_number += 1
    #             if (i1, index1) not in points:
    #                 contours.append(
    #                     (
    #                         number,
    #                         int(i1),
    #                         float(keypoints1[index1, 0] * N),
    #                         float(keypoints1[index1, 1] * N),
    #                     )
    #                 )
    #             contours.append(
    #                 (
    #                     number,
    #                     int(i2),
    #                     float(keypoints2[index2, 0] * N),
    #                     float(keypoints2[index2, 1] * N),
    #                 )
    #             )
    #             points[(i1, index1)] = number
    #             points[(i2, index2)] = number

    #         # transform = EuclideanTransform(rotation = transform.rotation, translation=-transform.translation[::-1])
    #         # img2 = warp(img2, transform)
    #         # pylab.imshow(img1 - img2)
    #         # pylab.show()

    #         # shifts = shifts[select, :]
    #         # shift = np.mean(shifts, axis=0)
    #         # print(i1, np.sqrt(np.sum((shift - shift0) ** 2)))
    #         print(
    #             i1,
    #             np.sum(inliers),
    #             inliers.size,
    #             keypoints1.shape[0],
    #             keypoints2.shape[0],
    #             np.degrees(transform.rotation),
    #             transform.translation * N - shift0 * N,
    #         )
    #         P[i2, 0] = P[i1, 0] + np.degrees(transform.rotation)
    #         P[i2, 3] = P[i1, 3] + transform.translation[0] * N
    #         P[i2, 4] = P[i1, 4] + transform.translation[1] * N

    #         # fig, ax = pylab.subplots()
    #         # ax.hist(shifts[:,0])
    #         # ax.hist(shifts[:,1])
    #         # pylab.show()

    # contours = sorted(contours)

    return contours


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

    def write_contours(filename, contours):
        print("Writing contours to %s" % filename)
        yaml.safe_dump(contours, open(filename, "w"), default_flow_style=None)

    # Read the projections
    projections = read_projections(projections_in)[:10]

    # Read the model
    model = read_model(model_in)

    # Read the initial model and convert degrees to radians for use here
    P = np.array(model["transform"], dtype=float)

    # Extract some contours
    contours = track_stack(projections, P)

    # Update the model and convert back to degrees
    model["transform"] = P.tolist()

    # Write the model to file
    write_model(model, model_out)
    write_contours(contours_out, {"contours": contours.tolist()})


if __name__ == "__main__":
    track()
