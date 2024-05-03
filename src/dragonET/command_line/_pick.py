#
# refine.py
#
# Copyright (C) 2024 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
import time
from argparse import ArgumentParser
from typing import List

import mrcfile
import napari
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
        "-c",
        "--contours",
        type=str,
        default="contours.yaml",
        dest="contours",
        help=(
            """
            A YAML file describing the picked point coordinates.
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
        args.contours,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def pick(args: List[str] = None):
    """
    Pick the fiduccials manually

    """
    pick_impl(get_parser().parse_args(args=args))


def _pick(projections_filename: str, contours_filename: str):
    """
    Pick the fiduccials manually

    """

    def write_contours(filename, contours):
        print("Writing contours to %s" % filename)
        yaml.safe_dump(contours, open(filename, "w"), default_flow_style=None)

    # Initialise the viewer
    viewer = napari.Viewer()

    # Load the projections data
    projections_file = mrcfile.mmap(projections_filename)

    # Add the image layer
    viewer.add_image(projections_file.data, name="Projections")

    # Start Napari
    napari.run()

    # Get the points layers
    contours = []
    for index, layer in enumerate(viewer.layers):
        if isinstance(layer, napari.layers.Points):
            points = layer.data
            z = points[:, 0]
            print("Contour %d has %d points" % (index, points.shape[0]))
            if points.shape[0] == 1:
                print("- Warning: contour %d has only 1 point" % index)
            if len(z) != len(set(z)):
                print("- Warning: Contour %d has more than 1 point per image" % index)
            for p in points.tolist():
                contours.append((index, p[0], p[1], p[2]))

    # Write the contours
    write_contours(contours_filename, {"contours": contours})
