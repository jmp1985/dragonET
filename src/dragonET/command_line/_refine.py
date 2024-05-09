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

import numpy as np
import scipy.optimize
import yaml
from matplotlib import pylab
from scipy.spatial.transform import Rotation

__all__ = ["refine"]


def get_description():
    """
    Get the program description

    """
    return "Refine a model to align the projection images"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the refine parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "--points_in",
        type=str,
        default=None,
        dest="points_in",
        required=True,
        help=(
            """
            A YAML file containing contour information.
            """
        ),
    )
    parser.add_argument(
        "--points_out",
        type=str,
        default="refined_points.yaml",
        dest="points_out",
        help=(
            """
            A YAML file describing the refined point coordinates.
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
            A YAML file describing the refined model.
            """
        ),
    )
    parser.add_argument(
        "--refine_pitch",
        type=bool,
        default=False,
        dest="refine_pitch",
        help="Refine pitch (True/False)",
    )
    parser.add_argument(
        "--refine_roll",
        type=bool,
        default=False,
        dest="refine_roll",
        help="Refine roll (True/False)",
    )

    return parser


def refine_impl(args):
    """
    Refine the model of the sample and align the images

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _refine(
        args.model_in,
        args.model_out,
        args.points_in,
        args.points_out,
        args.refine_pitch,
        args.refine_roll,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def refine(args: List[str] = None):
    """
    Refine the model of the sample and align the images

    """
    refine_impl(get_parser().parse_args(args=args))


def flatten_parameters(P, X):
    return np.concatenate([P.flatten(), X.flatten()])


def unflatten_parameters(params, P_shape, X_shape):
    P_size = np.prod(P_shape)
    P = params[:P_size].reshape(P_shape)
    X = params[P_size:].reshape(X_shape)
    return P, X


def make_bounds(
    P, X, reference_image=None, refine_yaw=False, refine_pitch=False, refine_roll=False
):
    if refine_yaw:
        yaw = 180
    else:
        yaw = 1e-7

    if refine_pitch:
        pitch = 90
    else:
        pitch = 1e-7

    if refine_roll:
        roll = 0.5
    else:
        roll = 1e-7

    # P lower bounds
    P_lower = np.zeros_like(P)
    P_lower[:, 0] = P[:, 0] - np.radians(yaw)  # Yaw
    P_lower[:, 1] = P[:, 1] - np.radians(pitch)  # Pitch
    P_lower[:, 2] = P[:, 2] - np.radians(roll)  # Roll
    P_lower[:, 3] = -4096  # Y
    P_lower[:, 4] = -4096  # X

    # P upper bounds
    P_upper = np.zeros_like(P)
    P_upper[:, 0] = P[:, 0] + np.radians(yaw)  # Yaw
    P_upper[:, 1] = P[:, 1] + np.radians(pitch)  # Pitch
    P_upper[:, 2] = P[:, 2] + np.radians(roll)  # Roll
    P_upper[:, 3] = 4096  # Y
    P_upper[:, 4] = 4096  # X

    # Set the reference image parameters
    if reference_image:
        P_lower[reference_image, 1] = P[reference_image, 1] - 1e-7
        P_lower[reference_image, 2] = P[reference_image, 2] - 1e-7
        # P_lower[reference_image, 3] = P[reference_image, 3] - 1e-7
        # P_lower[reference_image, 4] = P[reference_image, 4] - 1e-7
        P_upper[reference_image, 1] = P[reference_image, 1] + 1e-7
        P_upper[reference_image, 2] = P[reference_image, 2] + 1e-7
        # P_upper[reference_image, 3] = P[reference_image, 3] + 1e-7
        # P_upper[reference_image, 4] = P[reference_image, 4] + 1e-7

    # X lower bounds
    X_lower = np.zeros_like(X)
    X_lower[:, :] = -4096

    # X upper bounds
    X_upper = np.zeros_like(X)
    X_upper[:, :] = 4096

    # Lower and upper bounds
    lower = np.concatenate([P_lower.flatten(), X_lower.flatten()])
    upper = np.concatenate([P_upper.flatten(), X_upper.flatten()])

    # Return bounds
    return lower, upper


# class Manager(object):

#     def __init__(self, a=None, b=None, c=None, dy=None, dx=None):
#         self.a = a
#         self.b = b
#         self.c = c
#         self.dy = dy
#         self.dx = dx

#     def unflatten_parameters(self, params):
#         P = np.zeros((num_images, 5))
#         X = np.zeros((num_points, 3))
#         if self.a:
#             P[:,0] = self.a
#         else:
#             P[:,0] = params[:,num_images]

#         if self.b:
#             P[:,1] = self.b
#         else:
#             P[:,1] = params[:,num_images]

#         if self.c:
#             P[:,2] = self.c
#         if self.dy:
#             P[:,3] = self.dy
#         if self.dx:
#             P[:,4] = self.dx

#         return P, X


# def residuals(
#     params,
#     P_shape=None,
#     X_shape=None,
#     origin=None,
#     contours=None,
#     manager=None):
#     P, X = manager.unflatten_parameters(params)


def residuals(
    params,
    P_shape=None,
    X_shape=None,
    image_size=None,
    contours=None,
):
    """
    Coordinate system Z, Y, X in volume space

    """

    # Unflatten the parameters
    P, X = unflatten_parameters(params, P_shape, X_shape)

    # The centre of the image
    centre = np.array([image_size[1] / 2, 0, image_size[0] / 2])

    # The transformation
    a = P[:, 0]  # Yaw
    b = P[:, 1]  # Pitch
    c = P[:, 2]  # Roll
    shifty = P[:, 3]  # Shift Y
    shiftx = P[:, 4]  # Shift X

    # Create the translation vector for each image
    T = np.stack([shiftx, np.zeros(shiftx.size), shifty], axis=1)

    # Create the rotation matrix for each image
    Ra = Rotation.from_rotvec(np.array((0, 1, 0))[None, :] * a[:, None]).as_matrix()
    Rb = Rotation.from_rotvec(np.array((1, 0, 0))[None, :] * b[:, None]).as_matrix()
    Rc = Rotation.from_rotvec(np.array((0, 0, 1))[None, :] * c[:, None]).as_matrix()

    # The full rotation matrix
    R = Ra @ Rb @ Rc

    # Get index, z, y and x from the contours
    index = contours["index"]  # Point index
    z = contours["z"]  # Image number
    y_obs = contours["y"]  # Observed y coordinate on image
    x_obs = contours["x"]  # Observed x coordinate on image

    # Compute the residuals
    p = np.einsum("...ij,...j", R[z], X[index]) + T[z] + centre
    r = np.zeros((contours.shape[0], 2))
    r[:, 0] = p[:, 2] - y_obs  # (p . (0, 0, 1)) - y_obs
    r[:, 1] = p[:, 0] - x_obs  # (p . (1, 0, 0)) - x_obs

    print(np.sqrt(np.mean(r**2)))
    return r.flatten()


def jacobian(
    params,
    P_shape=None,
    X_shape=None,
    image_size=None,
    contours=None,
):
    # Unflatten the parameters
    P, X = unflatten_parameters(params, P_shape, X_shape)

    # The transformation
    a = P[:, 0]  # Yaw
    b = P[:, 1]  # Pitch
    c = P[:, 2]  # Roll
    shifty = P[:, 3]  # Shift Y
    shiftx = P[:, 4]  # Shift X

    # Create the rotation matrix for each image
    Ra = Rotation.from_rotvec(np.array((0, 1, 0))[None, :] * a[:, None]).as_matrix()
    Rb = Rotation.from_rotvec(np.array((1, 0, 0))[None, :] * b[:, None]).as_matrix()
    Rc = Rotation.from_rotvec(np.array((0, 0, 1))[None, :] * c[:, None]).as_matrix()

    # The full rotation matrix
    R = Ra @ Rb @ Rc

    fix_a = False
    fix_b = False
    fix_c = False

    # Get index, z, y and x from the contours
    index = contours["index"]  # Point index
    z = contours["z"]  # Image number
    y_obs = contours["y"]  # Observed y coordinate on image
    x_obs = contours["x"]  # Observed x coordinate on image

    # Derivatives of p wrt to point parameters
    dp_dX0 = R[z, :, 0]  # R[z] @ dp0_dX0 = R[z] @ (1, 0, 0)
    dp_dX1 = R[z, :, 1]  # R[z] @ dp0_dX1 = R[z] @ (0, 1, 0)
    dp_dX2 = R[z, :, 2]  # R[z] @ dp0_dX2 = R[z] @ (0, 0, 1)

    # Initialise the elements of the Jacobian
    JP = np.zeros((contours.shape[0], 2, P.shape[0], P.shape[1]))
    JX = np.zeros((contours.shape[0], 2, X.shape[0], X.shape[1]))

    # The array of indices
    i = np.arange(z.size)

    if not fix_a:
        # Compute cos and sin of yaw
        cosa = np.cos(a)
        sina = np.sin(a)

        # Derivative wrt yaw
        # [[-sina, 0,  cosa],
        #  [    0, 0,     0],
        #  [-cosa, 0, -sina]]
        dRa_da = np.zeros((P.shape[0], 3, 3))
        dRa_da[:, 0, 0] = -sina
        dRa_da[:, 0, 2] = cosa
        dRa_da[:, 2, 0] = -cosa
        dRa_da[:, 2, 2] = -sina

        # Derivatives of full rotation matrix wrt yaw
        dR_da = dRa_da @ Rb @ Rc

        # Derivatives of p wrt to yaw
        dp_da = np.einsum("...ij,...j", dR_da[z], X[index])

        # Derivatives wrt yaw
        JP[i, 0, z, 0] = dp_da[i, 2]  # dy_prd_da = dp_da . (0, 0, 1)
        JP[i, 1, z, 0] = dp_da[i, 0]  # dx_prd_da = dp_da . (1, 0, 0)

    if not fix_b:
        # Cos and sin of pitch
        cosb = np.cos(b)
        sinb = np.sin(b)

        # Derivative wrt pitch
        # [[0,     0,     0],
        #  [0, -sinb, -cosb],
        #  [0,  cosb, -sinb]]
        dRb_db = np.zeros((P.shape[0], 3, 3))
        dRb_db[:, 1, 1] = -sinb
        dRb_db[:, 1, 2] = -cosb
        dRb_db[:, 2, 1] = cosb
        dRb_db[:, 2, 2] = -sinb

        # Derivatives of full rotation matrix wrt pitch
        dR_db = Ra @ dRb_db @ Rc

        # Derivatives of p wrt to pitch
        dp_db = np.einsum("...ij,...j", dR_db[z], X[index])

        # Derivatives wrt pitch
        JP[i, 0, z, 1] = dp_db[i, 2]  # dy_prd_db = dp_db . (0, 0, 1)
        JP[i, 1, z, 1] = dp_db[i, 0]  # dx_prd_db = dp_db . (1, 0, 0)

    if not fix_c:
        # Cos and sin of yaw, pitch and roll
        cosc = np.cos(c)
        sinc = np.sin(c)

        # Derivative wrt roll
        # [[-sinc, -cosc, 0],
        #  [ cosc, -sinc, 0],
        #  [    0,     0, 0]]
        dRc_dc = np.zeros((P.shape[0], 3, 3))
        dRc_dc[:, 0, 0] = -sinc
        dRc_dc[:, 0, 1] = -cosc
        dRc_dc[:, 1, 0] = cosc
        dRc_dc[:, 1, 1] = -sinc

        # Derivatives of full rotation matrix wrt yaw, pitch and roll
        dR_dc = Ra @ Rb @ dRc_dc

        # Derivatives of p wrt to rotations
        dp_dc = np.einsum("...ij,...j", dR_dc[z], X[index])

        # Derivatives wrt roll
        JP[i, 0, z, 2] = dp_dc[i, 2]  # dy_prd_dc = dp_dc . (0, 0, 1)
        JP[i, 1, z, 2] = dp_dc[i, 0]  # dx_prd_dc = dp_dc . (1, 0, 0)

    # Derivatives wrt dy and dx
    JP[i, 0, z, 3] = 1  # dy_prd_dy = dp_dy . (0, 0, 1) = (0, 0, 1) . (0, 0, 1)
    JP[i, 1, z, 3] = 0  # dx_prd_dy = dp_dy . (1, 0, 0) = (0, 0, 1) . (1, 0, 0)
    JP[i, 0, z, 4] = 0  # dy_prd_dx = dp_dx . (0, 0, 1) = (1, 0, 0) . (0, 0, 1)
    JP[i, 1, z, 4] = 1  # dx_prd_dx = dp_dx . (1, 0, 0) = (1, 0, 0) . (1, 0, 0)

    # Derivatives wrt point parameters
    JX[i, 0, index, 0] = dp_dX0[i, 2]  # dy_prd_dX0 = dp_dX0 . (0, 0, 1)
    JX[i, 0, index, 1] = dp_dX1[i, 2]  # dy_prd_dX1 = dp_dX1 . (0, 0, 1)
    JX[i, 0, index, 2] = dp_dX2[i, 2]  # dy_prd_dX2 = dp_dX2 . (0, 0, 1)
    JX[i, 1, index, 0] = dp_dX0[i, 0]  # dx_prd_dX0 = dp_dX0 . (1, 0, 0)
    JX[i, 1, index, 1] = dp_dX1[i, 0]  # dx_prd_dX1 = dp_dX1 . (1, 0, 0)
    JX[i, 1, index, 2] = dp_dX2[i, 0]  # dx_prd_dX2 = dp_dX2 . (1, 0, 0)

    # Return the Jacobian
    JP = JP.reshape((contours.shape[0] * 2, P.size))
    JX = JX.reshape((contours.shape[0] * 2, X.size))
    J = np.concatenate([JP, JX], axis=1)
    return J


def estimate(
    P,
    X,
    contours,
    image_size,
    reference_image=None,
    refine_yaw=False,
    refine_pitch=False,
    refine_roll=False,
):
    # Flatten the parameters
    params = flatten_parameters(P, X)

    # Make the parameter bounds
    lower, upper = make_bounds(
        P, X, reference_image, refine_yaw, refine_pitch, refine_roll
    )

    # Make the kwargs to pass to the residuals function
    kwargs = {
        "P_shape": P.shape,
        "X_shape": X.shape,
        "image_size": image_size,
        "contours": contours,
    }

    # Do the optimisation
    result = scipy.optimize.least_squares(
        residuals, params, jac=jacobian, bounds=(lower, upper), kwargs=kwargs
    )

    rmsd = np.sqrt(result.cost / len(contours))

    # Unflatten the parameters
    P, X = unflatten_parameters(result.x, P.shape, X.shape)

    return P, X, rmsd


def read_points(filename) -> dict:
    return yaml.safe_load(open(filename, "r"))


def read_model(filename) -> dict:
    return yaml.safe_load(open(filename, "r"))


def write_points(points, filename):
    yaml.safe_dump(points, open(filename, "w"), default_flow_style=None)


def write_model(model, filename):
    yaml.safe_dump(model, open(filename, "w"), default_flow_style=None)


def _refine(
    model_in: str,
    model_out: str,
    points_in: str,
    points_out: str = None,
    refine_pitch: bool = False,
    refine_roll: bool = False,
):
    # Read the model
    model = read_model(model_in)

    # Read the points
    points = read_points(points_in)

    # Read the contours
    contours = np.array(
        [tuple(x) for x in points["contours"]],
        dtype=[("index", "int"), ("z", "int"), ("y", "float"), ("x", "float")],
    )

    # Read the initial model and convert degrees to radians for use here
    P = np.array(model["transform"], dtype=float)
    P[:, 0] = np.radians(P[:, 0])
    P[:, 1] = np.radians(P[:, 1])
    P[:, 2] = np.radians(P[:, 2])

    # Get the initial angles
    angles = P[:, 2]  # Tilt angles are the roll angles

    # The image size
    image_size = model["image_size"]

    # Set the reference image
    reference_image = np.argmin(angles**2)

    # The number of images and points
    num_images = P.shape[0]
    num_points = len(set(contours["index"]))
    assert num_images == len(set(contours["z"]))

    # If the coordinates are specified then use these as a starting point
    if "coordinates" in points:
        X = np.array(points["coordinates"], dtype=float)
        assert X.shape[0] == num_points
    else:
        # Initialise the point parameters
        X = np.zeros((num_points, 3))

    # print("with yaw")
    # P, X, rmsd = estimate(
    #     P,
    #     X,
    #     contours,
    #     image_size,
    #     reference_image,
    #     refine_yaw=False,
    #     refine_pitch=False,
    #     refine_roll=False,
    # )

    print("with yaw")
    P, X, rmsd = estimate(
        P,
        X,
        contours,
        image_size,
        reference_image,
        refine_yaw=True,
        refine_pitch=False,
        refine_roll=False,
    )

    if refine_pitch:
        print("With pitch")
        P, X, rmsd = estimate(
            P,
            X,
            contours,
            image_size,
            reference_image,
            refine_yaw=True,
            refine_pitch=True,
            refine_roll=False,
        )

    if refine_roll:
        print("With roll")
        P, X, rmsd = estimate(
            P,
            X,
            contours,
            image_size,
            reference_image,
            refine_yaw=True,
            refine_pitch=True,
            refine_roll=True,
        )

    print(P[reference_image])

    # Plot the parameters
    pylab.plot(np.degrees(P[:, 0]), label="Yaw")
    pylab.plot(np.degrees(P[:, 1]), label="Pitch")
    pylab.plot(np.degrees(P[:, 2]), label="Roll")
    pylab.legend()
    pylab.show()

    # Update the model and convert back to degrees
    P[:, 0] = np.degrees(P[:, 0])
    P[:, 1] = np.degrees(P[:, 1])
    P[:, 2] = np.degrees(P[:, 2])
    model["transform"] = P.tolist()

    # Update the points
    points["coordinates"] = X.tolist()

    # Save the refined model
    write_model(model, model_out)

    # Save the refined points
    if points_out:
        write_points(points, points_out)


if __name__ == "__main__":
    refine()
