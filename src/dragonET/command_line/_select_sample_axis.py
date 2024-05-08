#
# select_sample_axis.py
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

__all__ = ["select_sample_axis"]


def get_description():
    """
    Get the program description

    """
    return "Select the sample axis"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "-v",
        "--volume",
        type=str,
        default=None,
        dest="volume",
        required=True,
        help=(
            """
            The volume.
            """
        ),
    )
    parser.add_argument(
        "-i",
        "--model_in",
        type=str,
        default=None,
        dest="model_in",
        required=True,
        help=(
            """
            A YAML file describing the geometry model.
            """
        ),
    )
    parser.add_argument(
        "-o",
        "--model_out",
        type=str,
        default="aligned_model.yaml",
        dest="model_out",
        help=(
            """
            A YAML file describing the geometry model.
            """
        ),
    )

    return parser


def select_sample_axis_impl(args):
    """
    Select the sample axis

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _select_sample_axis(
        args.volume,
        args.model_in,
        args.model_out,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def select_sample_axis(args: List[str] = None):
    """
    Select the sample axis

    """
    select_sample_axis_impl(get_parser().parse_args(args=args))


def _select_sample_axis(
    volume_filename: str,
    model_in_filename: str = None,
    model_out_filename: str = None,
):
    """
    Select sample axis

    """
    import napari

    def read_volume(filename):
        print("Reading volume from %s" % filename)
        return mrcfile.mmap(filename)

    def read_model(filename):
        print("Reading model from %s" % filename)
        return yaml.safe_load(open(filename))

    def write_model(filename, model):
        print("Writing model to %s" % filename)
        yaml.safe_dump(model, open(filename, "w"), default_flow_style=None)

    def normalise(v):
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return v

    def get_points(layers):
        p1 = np.array((0, 0, 0))
        p2 = np.array((1, 0, 0))
        for layer in layers:
            if isinstance(layer, napari.layers.Points):
                points = layer.data
                if points.shape[0] == 0:
                    print("- Warning: no points in layer")
                elif points.shape[0] == 1:
                    print("- Warning: only 1 point in layer")
                else:
                    p1 = points[0, :]
                    p2 = points[1, :]
                    print("Point 1: %f, %f, %f" % tuple(p1))
                    print("Point 2: %f, %f, %f" % tuple(p2))
                    p1 = p1 - np.array(volume_file.data.shape) / 2
                    p2 = p2 - np.array(volume_file.data.shape) / 2
                    break
        return [p1, p2]

    def compute_axis_and_origin(points):
        p1, p2 = points
        axis = normalise(p2 - p1)
        t = -p1[0] / axis[0]
        axis_origin = t * axis + p1
        return axis, axis_origin

    # Initialise the viewer
    viewer = napari.Viewer()

    # Load the projections data
    volume_file = read_volume(volume_filename)

    # Read the model
    model = read_model(model_in_filename)

    # Add the image layer
    viewer.add_image(volume_file.data, name="Projections")

    # Start Napari
    napari.run()

    # Compute origin and direction
    axis, axis_origin = compute_axis_and_origin(get_points(viewer.layers))
    print("Axis: %f, %f, %f" % tuple(axis))
    print("Axis origin: %f, %f, %f" % tuple(axis_origin))

    # Set the axis
    model["axis"] = axis.tolist()
    model["axis_origin"] = axis_origin.tolist()

    # Write the model
    write_model(model_out_filename, model)
