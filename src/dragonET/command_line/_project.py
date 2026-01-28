#
# project.py
#
# Copyright (C) 2024 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
import time
from argparse import ArgumentParser
from typing import List

import astra
import mrcfile
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

__all__ = ["project"]


def get_description() -> str:
    """
    Get the program description

    """
    return "Do the projection"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the project parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        dest="model",
        required=True,
        help=("""
            A file describing the initial model. This file can either be a
            .rawtlt file or a YAML file.
            """),
    )
    parser.add_argument(
        "-v",
        "--volume",
        type=str,
        default=None,
        dest="volume",
        required=True,
        help=("""
            The volume to project from.
            """),
    )
    parser.add_argument(
        "-p",
        "--projections",
        type=str,
        default="projections.mrc",
        dest="projections",
        help=("""
            The projection images.
            """),
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=1,
        dest="pixel_size",
        help="The pixel size relative to the voxel size",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["gpu", "gpu_and_host", "host"],
        default="gpu",
        dest="device",
        help="The device settings to use",
    )

    return parser


def project_impl(args):
    """
    Reconstruct the volume

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _project(
        args.volume,
        args.model,
        args.projections,
        args.pixel_size,
        args.device,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def project(args: List[str] = None):
    """
    Reconstruct the volume

    """
    project_impl(get_parser().parse_args(args=args))


def _prepare_astra_geometry(
    P: np.ndarray,
    pixel_size: float = 1,
    image_size: tuple = (0, 0),
    axis=(0, 0, 1),
    axis_origin=(0, 0, 0),
) -> np.ndarray:
    """
    Prepare the geometry vectors

    Params:
        P: The array of parameters
        pixel_size: The pixel size relative to the voxel size
        axis: The sample axis to align
        axis_origin: The sample axis origin to align

    Returns:
        The array of geometry vectors

    """

    def matrix_to_rotate_a_onto_b(a, b):
        # Compute the unit vectors
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

        # Compute the matrix
        # cos(t), -sin(t), 0
        # sin(t), cos(t), 0
        # 0, 0, 1
        d_ab = np.dot(a, b)
        c_ab = np.linalg.norm(np.cross(a, b))
        G = np.array([[d_ab, -c_ab, 0], [c_ab, d_ab, 0], [0, 0, 1]])

        # Compute the rotation matrix U such that Ua = b
        u = a
        v = b - np.dot(a, b) * a
        len_v = np.linalg.norm(v)
        if len_v > 0:
            v = v / len_v
            w = np.cross(b, a)
            F = np.linalg.inv(np.stack([u, v, w]).T)
            U = np.linalg.inv(F) @ G @ F
        else:
            U = np.diag((1.0, 1.0, 1.0))

        # Return the rotation matrix
        return U

    def prepare_sample_alignment_rotation_and_translation(axis, axis_origin):
        U = matrix_to_rotate_a_onto_b(axis, np.array((0, 0, 1)))
        return U, -axis_origin

    print("Preparing geometry with pixel size %f" % pixel_size)
    assert all(np.array(image_size) > 0)

    # Prepare the transform to align the sample. The origin is wrt the centre
    # of the volume and the direction is a unit vector. Here we compute the
    # rotation matrix and translation that put a given line along the centre of
    # the reconstruction volume
    Rs, Ts = prepare_sample_alignment_rotation_and_translation(axis, axis_origin)

    # The transformation
    shiftx = P[:, 0] * image_size[1]  # Shift X
    shifty = P[:, 1] * image_size[0]  # Shift Y
    a = np.radians(P[:, 2])  # Yaw
    b = np.radians(P[:, 3])  # Pitch
    c = np.radians(P[:, 4])  # Roll

    # Create the rotation matrix for each image
    Ra = Rotation.from_euler("z", a).as_matrix()
    Rb = Rotation.from_euler("x", b).as_matrix()
    Rc = Rotation.from_euler("y", c).as_matrix()

    # Need to invert the rotation matrix for astra convention
    R = np.linalg.inv(Ra @ Rb @ Rc @ Rs.T)

    # Create the translation vector for each image
    t = np.stack([-shiftx, -shifty, np.zeros(shiftx.size)], axis=1)

    # Initialise the per-image geometry vectors
    vectors = np.zeros((P.shape[0], 12))

    # Ray direction vector
    vectors[:, 0:3] = R @ (0, 0, -1)

    # Detector centre
    vectors[:, 3:6] = (np.einsum("...ij,...j", R, t) + Ts) * pixel_size

    # Vector from detector pixel (0,0) to (0,1)
    vectors[:, 6:9] = R @ (pixel_size, 0, 0)

    # Vector from detector pixel (0,0) to (1,0)
    vectors[:, 9:12] = R @ (0, pixel_size, 0)

    # Return the vectors
    return vectors


def _project_with_astra(
    volume: np.ndarray,
    vectors: np.ndarray,
    shape: tuple,
    device: str = "gpu",
) -> np.ndarray:
    """
    Do the projection with astra

    Params:
        volume: The volume
        vectors: The array of geometry vectors
        shape: The projections shape
        device: The device to do the projection on

    Returns:
        The projected volume

    """
    # Create the projections array
    projections = np.zeros((shape[0], vectors.shape[0], shape[1]), dtype=np.float32)

    # Create the volume geometry
    vol_geom = astra.create_vol_geom(
        volume.shape[1],  # Num rows in reconstruction (axis 1)
        volume.shape[2],  # Num cols in reconstruction (axis 2)
        volume.shape[0],  # Num slices in reconstruction (axis 0)
    )

    # Create the projection geometry
    proj_geom = astra.create_proj_geom(
        "parallel3d_vec",
        shape[0],  # Num rows in projections (axis 0)
        shape[1],  # Num cols in projections (axis 2)
        vectors,  # Geometry vectors
    )

    # Check the device input
    if device not in ["gpu", "gpu_and_host", "host"]:
        raise RuntimeError("Device must be 'gpu' or 'host', got %s" % device)

    # Create the projector object
    if device in ["gpu", "gpu_and_host"]:
        projector_id = astra.create_projector("cuda3d", proj_geom, vol_geom)
    elif device in ["host"]:
        raise RuntimeError("Not implemented")

    # Do the projection
    W = astra.OpTomo(projector_id)
    W.FP(volume, out=projections)

    # Put the projections in sinogram order
    projections = np.swapaxes(projections, 0, 1)

    # Return the projected volume
    return projections


def project_internal(volume, P, pixel_size, axis, axis_origin, mode):
    """
    Project the image

    """
    # Get the image size
    image_size = (volume.shape[0], volume.shape[2])

    # Prepare the geometry vector description
    vectors = _prepare_astra_geometry(P, pixel_size, image_size, axis, axis_origin)

    # Do the projection with astra
    return _project_with_astra(volume, vectors, image_size, mode)


def _project(
    volume_filename: str,
    model_filename: str,
    projections_filename: str,
    pixel_size: float = 1,
    device: str = "gpu",
):
    """
    Do the projection

    Params:
        volume_filename: The filename of the reconstructed volume
        model_filename: The filename of the geometry model
        projections_filename: The filename of the projections
        pixel_size: The pixel size on the images relative to the voxel size
        device: The device to do the projection on

    """

    def read_model(filename):
        print("Reading model from %s" % filename)
        return yaml.safe_load(open(filename))

    def read_volume(filename):
        print("Reading volume from %s" % filename)
        return mrcfile.mmap(filename).data

    def write_projections(filename, projections):
        print("Writing projections to %s" % filename)
        outfile = mrcfile.new(filename, overwrite=True)
        outfile.set_data(projections)

    def normalise(v):
        return v / np.linalg.norm(v)

    # Read the model
    model = read_model(model_filename)

    # Read the volume data
    volume = read_volume(volume_filename)

    # Get the transform from the model
    P = np.array(model["transform"], dtype=float)

    # Get the vector to align to
    axis = np.array(normalise(model.get("axis", (1, 0, 0)))[::-1])
    axis_origin = np.array(model.get("axis_origin", (0, 0, 0))[::-1])

    # Do the projection
    projections = project_internal(volume, P, pixel_size, axis, axis_origin, device)

    # Create a new file with the projected images
    write_projections(projections_filename, projections)


if __name__ == "__main__":
    project()
