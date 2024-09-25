#
# pick.py
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

__all__ = ["pick"]


def get_description():
    """
    Get the program description

    """
    return "Manually pick fiduccials"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the pick parser

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
        "-o",
        "--contours_out",
        type=str,
        default="contours.yaml",
        dest="contours_out",
        help=(
            """
            A YAML file describing the picked point coordinates.
            """
        ),
    )
    parser.add_argument(
        "-i",
        "--contours_in",
        type=str,
        default=None,
        dest="contours_in",
        help=(
            """
            A YAML file describing the picked point coordinates.
            """
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        dest="model",
        help=(
            """
            A YAML file describing the geometry model.
            """
        ),
    )

    return parser


def pick_impl(args):
    """
    Pick the fiduccials manually

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _pick(
        args.projections,
        args.contours_out,
        args.contours_in,
        args.model,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def pick(args: List[str] = None):
    """
    Pick the fiduccials manually

    """
    pick_impl(get_parser().parse_args(args=args))


def _pick(
    projections_filename: str,
    contours_out_filename: str,
    contours_in_filename: str = None,
    model_in_filename: str = None,
):
    """
    Pick the fiduccials manually

    """
    import napari

    def read_projections(filename):
        print("Reading projections from %s" % filename)
        return mrcfile.mmap(filename)

    def read_contours(filename):
        if filename:
            print("Reading contours from %s" % filename)
            contours = yaml.safe_load(open(filename))["contours"]
        else:
            contours = None
        return contours

    def read_model(filename):
        if filename:
            print("Reading model from %s" % filename)
            model = yaml.safe_load(open(filename))
        else:
            model = None
        return model

    def write_contours(filename, contours):
        print("Writing contours to %s" % filename)
        yaml.safe_dump(contours, open(filename, "w"), default_flow_style=None)

    def make_transform(model):
        # Initialise the transforms
        transform = np.zeros((projections_file.data.shape[0], 4, 4))
        for i in range(transform.shape[0]):
            transform[i, :, :] = np.diag([1, 1, 1, 1])

        # Get the transform for each image
        if model and "transform" in model:
            P = np.array(model["transform"])
            assert P.shape[0] == transform.shape[0]
            for i in range(P.shape[0]):
                theta = np.radians(P[i, 2])

                O = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, projections_file.data.shape[1] / 2],
                        [0, 0, 1, projections_file.data.shape[2] / 2],
                        [0, 0, 0, 1],
                    ]
                )

                T = np.array(
                    [[1, 0, 0, 0], [0, 1, 0, P[i, 1]], [0, 0, 1, P[i, 0]], [0, 0, 0, 1]]
                )

                R = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1],
                    ]
                )

                transform[i] = O @ R @ np.linalg.inv(O) @ np.linalg.inv(T)

        # Return the transform
        return transform

    def transform_layer(event):
        # Get the index of the current slice
        # If the current transform is close to the expected transform do nothing
        # Otherwise transform the image
        index = event.source._slice_indices[0]
        if not np.all(np.isclose(event.source.affine, transform[index, :, :])):
            event.source.affine = transform[index]

    def set_contours(viewer, transform, contours):
        # If there are contours then add them as layers
        if contours:
            points = np.array(contours)[:, 1:4]
            points = [
                (transform[int(p[0])] @ np.array(p + [1]))[0:3] for p in points.tolist()
            ]
            points = [(int(p[0]), p[1], p[2]) for p in points]
            viewer.add_points(points, name="All points", face_color="blue", size=50)
            # contours = sorted(contours, key=lambda x: x[0])
            # for index, group in itertools.groupby(contours, lambda x: x[0]):
            #     print("Adding contour for point %d" % index)
            #     points = [x[1:] for x in group]
            #     points = [
            #         (transform[int(p[0])] @ np.array(p + [1]))[0:3] for p in points
            #     ]
            #     points = [(int(p[0]), p[1], p[2]) for p in points]
            #     name = "Points [%d]" % index if index > 0 else "Points"
            #     viewer.add_points(points, name=name, face_color="blue", size=50)

    def get_contours(viewer, transform):
        # Loop through the point sets and get the lists of contours
        # Make sure to transform the points using the inverse transform
        contours = []
        index = 0
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Points):
                points2 = layer.data
                if points2.shape[0] == 0:
                    print("- Warning: skipping contour with no points")
                else:
                    print("Contour %d has %d points" % (index, points2.shape[0]))
                    points2 = np.array(sorted(points2.tolist(), key=lambda x: x[0]))
                    if points2.shape[0] == 1:
                        print("- Warning: contour %d has only 1 point" % index)
                    if len(points2[:, 0]) != len(set(points2[:, 0])):
                        print(
                            "- Warning: Contour %d has more than 1 point per image"
                            % index
                        )
                    for p in points2.tolist():
                        q = (
                            np.linalg.inv(transform[int(p[0])])
                            @ np.array([0, p[1], p[2], 1])
                        ).tolist()
                        contours.append((index, int(p[0]), q[1], q[2]))
                    index += 1
        return contours

    # Initialise the viewer
    viewer = napari.Viewer()

    # Load the projections data
    projections_file = read_projections(projections_filename)

    # Read the contours
    contours = read_contours(contours_in_filename)

    # Read the model
    model = read_model(model_in_filename)

    # Create the transform array
    transform = make_transform(model)

    # Add the image layer
    viewer.add_image(projections_file.data, name="Projections")

    # Connect the transform image event
    viewer.layers[0].events.set_data.connect(transform_layer)
    viewer.layers[0].affine = transform[viewer.layers[0]._slice_indices[0]]

    # Add the contours to the viewer
    set_contours(viewer, transform, contours)

    # Start Napari
    napari.run()

    # Get the points layers
    contours = get_contours(viewer, transform)

    # Write the contours
    write_contours(contours_out_filename, {"contours": contours})
