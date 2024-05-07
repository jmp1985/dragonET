#
# refine.py
#
# Copyright (C) 2024 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
import itertools
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

    # Initialise the viewer
    viewer = napari.Viewer()

    # Load the projections data
    projections_file = read_projections(projections_filename)

    # Read the contours
    contours = read_contours(contours_in_filename)

    # Read the model
    model = read_model(model_in_filename)

    P = np.array(model["transform"])
    translate = np.zeros((P.shape[0], 3))
    translate[:, 1] = P[:, 3]
    translate[:, 2] = P[:, 4]
    translatr = translate.tolist()

    # Add the image layer
    viewer.add_image(projections_file.data, name="Projections", translate=translate)

    # Add the contours to the viewer
    contours = sorted(contours, key=lambda x: x[0])
    for index, group in itertools.groupby(contours, lambda x: x[0]):
        points = [x[1:] for x in group]
        name = "Points [%d]" % index if index > 0 else "Points"
        viewer.add_points(points, name=name)

    # Start Napari
    napari.run()

    # Get the points layers
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
                        "- Warning: Contour %d has more than 1 point per image" % index
                    )
                for p in points2.tolist():
                    contours.append((index, p[0], p[1], p[2]))
                index += 1

    # Write the contours
    write_contours(contours_out_filename, {"contours": contours})
