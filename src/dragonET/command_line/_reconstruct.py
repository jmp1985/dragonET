#
# reconstruct.py
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

__all__ = ["reconstruct"]


def get_description() -> str:
    """
    Get the program description

    """
    return "Do the reconstruction"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the reconstruct parser

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
        "-m",
        "--model",
        type=str,
        default=None,
        dest="model",
        required=True,
        help=(
            """
            A file describing the initial model. This file can either be a
            .rawtlt file or a YAML file.
            """
        ),
    )
    parser.add_argument(
        "-v",
        "--volume",
        type=str,
        default="volume.mrc",
        dest="volume",
        help=(
            """
            The reconstructed volume.
            """
        ),
    )
    parser.add_argument(
        "-i",
        "--initial_volume",
        type=str,
        default=None,
        dest="initial_volume",
        help=(
            """
            The initial volume.
            """
        ),
    )
    parser.add_argument(
        "--volume_shape",
        type=lambda x: tuple(map(int, x.split(","))),
        default=None,
        dest="volume_shape",
        help="The shape of the volume",
    )
    parser.add_argument(
        "--pixel_size",
        type=float,
        default=1,
        dest="pixel_size",
        help="The pixel size relative to the voxel size",
    )
    parser.add_argument(
        "-n",
        "--num_iterations",
        type=int,
        default=1,
        dest="num_iterations",
        help="The number of iterations.",
    )

    return parser


def reconstruct_impl(args):
    """
    Reconstruct the volume

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _reconstruct(
        args.projections,
        args.model,
        args.volume,
        args.initial_volume,
        args.volume_shape,
        args.pixel_size,
        args.num_iterations,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def reconstruct(args: List[str] = None):
    """
    Reconstruct the volume

    """
    reconstruct_impl(get_parser().parse_args(args=args))


def _prepare_astra_geometry(P: np.ndarray, pixel_size: float = 1) -> np.ndarray:
    """
    Prepare the geometry vectors

    Params:
        P: The array od parameters
        pixel_size: The pixel size relative to the voxel size

    Returns:
        The array of geometry vectors

    """
    print("Preparing geometry with pixel size %f" % pixel_size)

    # The transformation
    a = np.radians(P[:, 0])  # Yaw
    b = np.radians(P[:, 1])  # Pitch
    c = np.radians(P[:, 2])  # Roll
    shifty = P[:, 3]  # Shift Y
    shiftx = P[:, 4]  # Shift X

    # Create the rotation matrix for each image
    Ra = Rotation.from_rotvec(np.array((0, 1, 0))[None, :] * a[:, None]).as_matrix()
    Rb = Rotation.from_rotvec(np.array((1, 0, 0))[None, :] * b[:, None]).as_matrix()
    Rc = Rotation.from_rotvec(np.array((0, 0, 1))[None, :] * c[:, None]).as_matrix()

    # Need to invert the rotation matrix for astra convention
    R = np.linalg.inv(Ra @ Rb @ Rc)

    # Create the translation vector for each image
    t = np.stack([-shiftx, np.zeros(shiftx.size), -shifty], axis=1)

    # Initialise the per-image geometry vectors
    vectors = np.zeros((P.shape[0], 12))

    # Ray direction vector
    vectors[:, 0:3] = R @ (0, -1, 0)

    # Detector centre
    vectors[:, 3:6] = np.einsum("...ij,...j", R, t * pixel_size)

    # Vector from detector pixel (0,0) to (0,1)
    vectors[:, 6:9] = R @ (pixel_size, 0, 0)

    # Vector from detector pixel (0,0) to (1,0)
    vectors[:, 9:12] = R @ (0, 0, pixel_size)

    # Return the vectors
    return vectors


def _reconstruct_with_astra(
    projections: np.ndarray,
    vectors: np.ndarray,
    volume: np.ndarray = None,
    num_iterations: int = 1,
) -> np.ndarray:
    """
    Do the reconstruction with astra

    Params:
        projections: The array of projections in sinogram order
        vectors: The array of geometry vectors
        volume: The initial volume
        num_iterations: The number of iterations

    Returns:
        The reconstructed volume

    """

    # If volume is not set then initialise to zero
    if volume is None:
        volume = np.zeros(
            (
                projections.shape[0] // 2,
                projections.shape[2] // 2,
                projections.shape[2] // 2,
            ),
            dtype="float32",
        )

    # Create the volume geometry
    vol_geom = astra.create_vol_geom(
        volume.shape[1],  # Num rows in reconstruction (axis 1)
        volume.shape[2],  # Num cols in reconstruction (axis 2)
        volume.shape[0],  # Num slices in reconstruction (axis 0)
    )

    # Create the projection geometry
    proj_geom = astra.create_proj_geom(
        "parallel3d_vec",
        projections.shape[0],  # Num rows in projections (axis 0)
        projections.shape[2],  # Num cols in projections (axis 2)
        vectors,  # Geometry vectors
    )

    # Create the projection and reconstruction data
    projections_id = astra.data3d.create("-sino", proj_geom, projections)
    volume_id = astra.data3d.create("-vol", vol_geom, volume)

    # Do the reconstruction
    print("Reconstructing with %d iterations" % num_iterations)
    alg_cfg = astra.astra_dict("CGLS3D_CUDA")
    alg_cfg["ProjectionDataId"] = projections_id
    alg_cfg["ReconstructionDataId"] = volume_id
    algorithm_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(algorithm_id, iterations=num_iterations)

    # Get the reconstructed volume
    volume = astra.data3d.get(volume_id)

    # Cleanup the astra objects
    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete(volume_id)
    astra.data3d.delete(projections_id)

    # Return the reconstructed volume
    return volume


def _reconstruct(
    projections_filename: str,
    model_filename: str,
    volume_filename: str,
    initial_volume_filename: str = None,
    volume_shape: tuple = None,
    pixel_size: float = 1,
    num_iterations: int = 1,
):
    """
    Do the reconstruction

    Params:
        projections_filename: The filename of the projections
        model_filename: The filename of the geometry model
        volume_filename: The filename of the reconstructed volume
        initial_volume_filename: The filename of the initial volume
        num_iterations: The number of iterations to use

    """

    def read_model(filename):
        print("Reading model from %s" % filename)
        return yaml.safe_load(open(filename))

    def read_projections(filename):
        print("Reading projections from %s" % filename)
        return mrcfile.open(filename)

    def read_volume(filename, shape):
        if filename:
            print("Reading initial volume from %s" % filename)
            volume_file = mrcfile.open(filename)
            volume = volume_file.data.copy()
        elif shape:
            print("Initialising volume with shape: (%d, %d, %d)" % shape)
            volume = np.zeros(shape, dtype="float32")
        else:
            volume = None
        return volume

    def write_volume(filename, volume):
        print("Writing volume to %s" % filename)
        outfile = mrcfile.new(filename, overwrite=True)
        outfile.set_data(volume)

    # Read the model
    model = read_model(model_filename)

    # Read the projections data
    projections_file = read_projections(projections_filename)

    # Read in volume
    volume = read_volume(initial_volume_filename, volume_shape)

    # Get the transform from the model
    P = np.array(model["transform"], dtype=float)

    # Check the input is consistent
    assert P.shape[0] == projections_file.data.shape[0]

    # Put the projections in sinogram order
    projections = np.swapaxes(projections_file.data, 0, 1)

    # Prepare the geometry vector description
    vectors = _prepare_astra_geometry(P, pixel_size=pixel_size)

    # Do the reconstruction with astra
    volume = _reconstruct_with_astra(projections, vectors, volume, num_iterations)

    # Create a new file with the reconstructed volume
    write_volume(volume_filename, volume)


if __name__ == "__main__":
    reconstruct()
