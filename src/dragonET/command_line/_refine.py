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
        default="refined.csv",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        dest="batch_size",
        help="Set the batch size to determine initial model.",
    )
    parser.add_argument(
        "--nbatch_cycles",
        type=int,
        default=2,
        dest="nbatch_cycles",
        help="Set the number of batch cycles to perform before global optimization",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1,
        dest="kappa",
        help="The regularisation for the pitch and roll.",
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
        args.batch_size,
        args.nbatch_cycles,
        args.kappa,
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


def make_bounds(P, X, reference_image=None, refine_pitch=False, refine_roll=False):
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
    P_lower[:, 0] = np.radians(-90)  # Yaw
    P_lower[:, 1] = np.radians(-pitch)  # Pitch
    P_lower[:, 2] = P[:, 2] - np.radians(roll)  # Roll
    P_lower[:, 3] = -512 * 8  # Y
    P_lower[:, 4] = -512 * 8  # X

    # P upper bounds
    P_upper = np.zeros_like(P)
    P_upper[:, 0] = np.radians(90)  # Yaw
    P_upper[:, 1] = np.radians(pitch)  # Pitch
    P_upper[:, 2] = P[:, 2] + np.radians(roll)  # Roll
    P_upper[:, 3] = 512 * 8  # Y
    P_upper[:, 4] = 512 * 8  # X

    # Set the reference image parameters
    # if reference_image:
    # P_lower[reference_image, 1] = -1e-10
    # P_lower[reference_image, 2] = P[reference_image, 2] - 1e-7
    # P_upper[reference_image, 1] = 1e-10
    # P_upper[reference_image, 2] = P[reference_image, 2] + 1e-7

    # X lower bounds
    X_lower = np.zeros_like(X)
    X_lower[:, :] = -512 * 8

    # X upper bounds
    X_upper = np.zeros_like(X)
    X_upper[:, :] = 512 * 8

    # Lower and upper bounds
    lower = np.concatenate([P_lower.flatten(), X_lower.flatten()])
    upper = np.concatenate([P_upper.flatten(), X_upper.flatten()])

    # Return bounds
    return lower, upper


def residuals(
    params,
    P_shape=None,
    X_shape=None,
    origin=None,
    theta=None,
    kappa=None,
    contours=None,
):
    # Unflatten the parameters
    P, X = unflatten_parameters(params, P_shape, X_shape)

    # Construct the matrices
    R = np.zeros((P.shape[0], 3, 3))
    T = np.zeros((P.shape[0], 3))
    for i in range(P.shape[0]):
        # Get the parameters
        yaw, pitch, roll, shifty, shiftx = P[i]

        # Cos and sin of yaw, pitch and roll
        cosa = np.cos(yaw)
        sina = np.sin(yaw)
        cosb = np.cos(pitch)
        sinb = np.sin(pitch)
        cosc = np.cos(roll)
        sinc = np.sin(roll)

        # Yaw rotation matrix
        Ra = np.array([[1, 0, 0], [0, cosa, -sina], [0, sina, cosa]])

        # Pitch rotation matrix
        Rb = np.array([[cosb, -sinb, 0], [sinb, cosb, 0], [0, 0, 1]])

        # Roll rotation matrix
        Rc = np.array([[cosc, 0, sinc], [0, 1, 0], [-sinc, 0, cosc]])

        # The full rotation matrix
        R[i] = Ra @ Rb @ Rc

        # The translation
        T[i] = np.array([0, shifty, shiftx])

    # Compute the residuals
    r = np.zeros((contours.shape[0], 2))
    for j in range(r.shape[0]):
        index, z, y_obs, x_obs = contours[j]
        p = (R[z] @ (X[index] - origin)) + origin + T[z]
        r[j][0] = p[1] - y_obs  # (p . (0, 1, 0)) - y_obs
        r[j][1] = p[2] - x_obs  # (p . (0, 0, 1)) - x_obs

    # Compute the constraints
    cb = (kappa * 180 / np.pi) * P[:, 1]  # Prefer values close to zero for pitch
    cc = (kappa * 180 / np.pi) * (
        P[:, 2] - theta
    )  # Prefer values close to tilts for roll

    print(np.mean(r**2))
    return np.concatenate([r.flatten(), cb, cc])


def jacobian(
    params,
    P_shape=None,
    X_shape=None,
    origin=None,
    theta=None,
    kappa=None,
    contours=None,
):
    # Unflatten the parameters
    P, X = unflatten_parameters(params, P_shape, X_shape)

    # Construct the matrices for each image
    R = np.zeros((P.shape[0], 3, 3))
    T = np.zeros((P.shape[0], 3))
    dR_da = np.zeros((P.shape[0], 3, 3))
    dR_db = np.zeros((P.shape[0], 3, 3))
    dR_dc = np.zeros((P.shape[0], 3, 3))
    for i in range(P.shape[0]):
        # Get the parameters
        yaw, pitch, roll, shifty, shiftx = P[i]

        # Cos and sin of yaw, pitch and roll
        cosa = np.cos(yaw)
        sina = np.sin(yaw)
        cosb = np.cos(pitch)
        sinb = np.sin(pitch)
        cosc = np.cos(roll)
        sinc = np.sin(roll)

        # Yaw rotation matrix
        Ra = np.array([[1, 0, 0], [0, cosa, -sina], [0, sina, cosa]])

        # Pitch rotation matrix
        Rb = np.array([[cosb, -sinb, 0], [sinb, cosb, 0], [0, 0, 1]])

        # Roll rotation matrix
        Rc = np.array([[cosc, 0, sinc], [0, 1, 0], [-sinc, 0, cosc]])

        # The full rotation matrix
        R[i] = Ra @ Rb @ Rc

        # Derivative wrt yaw
        dRa_da = np.array([[0, 0, 0], [0, -sina, -cosa], [0, cosa, -sina]])

        # Derivative wrt pitch
        dRb_db = np.array([[-sinb, -cosb, 0], [cosb, -sinb, 0], [0, 0, 0]])

        # Derivative wrt roll
        dRc_dc = np.array([[-sinc, 0, cosc], [0, 0, 0], [-cosc, 0, -sinc]])

        # Derivatives of full rotation matrix wrt yaw, pitch and roll
        dR_da[i] = dRa_da @ Rb @ Rc
        dR_db[i] = Ra @ dRb_db @ Rc
        dR_dc[i] = Ra @ Rb @ dRc_dc

    # Compute the elements of the Jacobian
    JP = np.zeros((contours.shape[0], 2, P.shape[0], P.shape[1]))
    JX = np.zeros((contours.shape[0], 2, X.shape[0], X.shape[1]))
    for j in range(contours.shape[0]):
        # Get the observations
        index, z, y_obs, x_obs = contours[j]

        # Derivatives of p wrt to rotation matrix
        dp_da = dR_da[z] @ (X[index] - origin)
        dp_db = dR_db[z] @ (X[index] - origin)
        dp_dc = dR_dc[z] @ (X[index] - origin)

        # Derivatives of p wrt to point parameters
        dp_dX0 = R[z][:, 0]  # R[z] @ dp0_dX0 = R[z] @ (1, 0, 0)
        dp_dX1 = R[z][:, 1]  # R[z] @ dp0_dX1 = R[z] @ (0, 1, 0)
        dp_dX2 = R[z][:, 2]  # R[z] @ dp0_dX2 = R[z] @ (0, 0, 1)

        dp_dO0 = -R[z][:, 0] + np.array((1, 0, 0))  # -R[z] @ dO_dO0 + dO_dO0
        dp_dO1 = -R[z][:, 1] + np.array((0, 1, 0))  # -R[z] @ dO_dO1 + dO_dO1
        dp_dO2 = -R[z][:, 2] + np.array((0, 0, 1))  # -R[z] @ dO_dO2 + dO_dO2

        # Derivatives wrt image parameters
        JP[j, 0, z, 0] = dp_da[1]  # dy_prd_da = dp_da . (0, 1, 0)
        JP[j, 0, z, 1] = dp_db[1]  # dy_prd_db = dp_db . (0, 1, 0)
        JP[j, 0, z, 2] = dp_dc[1]  # dy_prd_dc = dp_dc . (0, 1, 0)
        JP[j, 0, z, 3] = 1  # dy_prd_dy = dp_dy . (0, 1, 0) = (0, 1, 0) . (0, 1, 0)
        JP[j, 0, z, 4] = 0  # dy_prd_dx = dp_dx . (0, 1, 0) = (0, 0, 1) . (0, 1, 0)
        JP[j, 1, z, 0] = dp_da[2]  # dx_prd_da = dp_da . (0, 0, 1)
        JP[j, 1, z, 1] = dp_db[2]  # dx_prd_db = dp_db . (0, 0, 1)
        JP[j, 1, z, 2] = dp_dc[2]  # dx_prd_dc = dp_dc . (0, 0, 1)
        JP[j, 1, z, 3] = 0  # dx_prd_dy = dp_dy . (0, 0, 1) = (0, 1, 0) . (0, 0, 1)
        JP[j, 1, z, 4] = 1  # dx_prd_dx = dp_dx . (0, 0, 1) = (0, 0, 1) . (0, 0, 1)

        # Derivatives wrt point parameters
        JX[j, 0, index, 0] = dp_dX0[1]  # dy_prd_dX0 = dp_dX0 . (0, 1, 0)
        JX[j, 0, index, 1] = dp_dX1[1]  # dy_prd_dX1 = dp_dX1 . (0, 1, 0)
        JX[j, 0, index, 2] = dp_dX2[1]  # dy_prd_dX2 = dp_dX2 . (0, 1, 0)
        JX[j, 1, index, 0] = dp_dX0[2]  # dx_prd_dX0 = dp_dX0 . (0, 0, 1)
        JX[j, 1, index, 1] = dp_dX1[2]  # dx_prd_dX1 = dp_dX1 . (0, 0, 1)
        JX[j, 1, index, 2] = dp_dX2[2]  # dx_prd_dX2 = dp_dX2 . (0, 0, 1)

    # Compute elements of the Jacobian for constraints
    JCP = np.zeros((P.shape[0] * 2, P.shape[0], P.shape[1]))
    JCX = np.zeros((P.shape[0] * 2, X.shape[0], X.shape[1]))
    i = np.arange(P.shape[0])
    JCP[i, i, 1] = kappa * 180 / np.pi  # dcb_db
    JCP[i + P.shape[0], i, 2] = kappa * 180 / np.pi  # dcc_dc
    JCP = JCP.reshape((P.shape[0] * 2, P.size))
    JCX = JCX.reshape((P.shape[0] * 2, X.size))
    JC = np.concatenate([JCP, JCX], axis=1)

    # Return the Jacobian
    JP = JP.reshape((contours.shape[0] * 2, P.size))
    JX = JX.reshape((contours.shape[0] * 2, X.size))
    J = np.concatenate([JP, JX], axis=1)
    return np.concatenate([J, JC], axis=0)


def estimate(
    P,
    X,
    contours,
    angles,
    origin,
    kappa,
    reference_image=None,
    refine_pitch=False,
    refine_roll=False,
):
    # Flatten the parameters
    params = flatten_parameters(P, X)

    # Make the parameter bounds
    lower, upper = make_bounds(P, X, reference_image, refine_pitch, refine_roll)

    # Make the kwargs to pass to the residuals function
    kwargs = {
        "P_shape": P.shape,
        "X_shape": X.shape,
        "origin": origin,
        "theta": angles,
        "kappa": kappa,
        "contours": contours,
    }

    # Do the optimisation
    result = scipy.optimize.least_squares(
        residuals, params, jac=jacobian, bounds=(lower, upper), kwargs=kwargs
    )

    # Unflatten the parameters
    P, X = unflatten_parameters(result.x, P.shape, X.shape)

    return P, X


def read_points(filename) -> dict:
    return yaml.safe_load(open(filename, "r"))


def read_model(filename) -> dict:
    return yaml.safe_load(open(filename, "r"))
    # model = np.loadtxt(filename, delimiter=",")
    #     assertif len(model.shape) == 1:
    #         P = np.zeros((model.shape[0], 5))
    #         P[:, 2] = np.radians(model)  # Roll angles are the known tilt angles
    #     else:
    #         P = model
    #         # P[:, 0] = np.radians(P[:,0])
    #         # P[:, 1] = np.radians(P[:,1])
    #         # P[:, 2] = np.radians(P[:,2])
    # return P


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
    batch_size: int = 10,
    nbatch_cycles: int = 2,
    kappa: float = 10,
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

    # The origin in sample space
    origin = np.array(model["axis_origin"], dtype=float)

    # Set the reference image
    reference_image = np.argmin(angles**2)

    # The number of images and points
    num_images = P.shape[0]
    num_points = len(set(contours["index"]))
    assert num_images == len(set(contours["z"]))

    # If the coordinates are specified then use these directly in global
    # refinement, otherwise do a batch refinement to get initial point
    # parameters
    if "coordinates" in points:
        X = np.array(points["coordinates"], dtype=float)
        assert X.shape[0] == num_points
    else:
        # Initialise the point parameters
        X = np.zeros((num_points, 3))
        X[:, 0] = origin[0]
        X[:, 1] = origin[1]
        X[:, 2] = origin[2]

        # Perform a number of initial batch cycles prior to global optimisation
        for cycle in range(nbatch_cycles):
            # Loop through all batches of images
            for batch0 in np.arange(0, num_images, batch_size // 2):
                batch1 = batch0 + batch_size
                select = (contours["z"] >= batch0) & (contours["z"] < batch1)
                contours2 = contours[select]
                contours2["z"] = contours2["z"] - contours2["z"].min()
                angles2 = angles[batch0:batch1]
                P2 = P[batch0:batch1]
                P2, X = estimate(P2, X, contours2, angles2, origin, kappa)
                P[batch0:batch1] = P2

    print("All")
    P, X = estimate(
        P,
        X,
        contours,
        angles,
        origin,
        kappa,
        reference_image,
        refine_pitch=True,
    )

    if refine_pitch:
        print("With pitch")
        P, X = estimate(
            P, X, contours, angles, origin, kappa, reference_image, refine_pitch=True
        )

    if refine_roll:
        print("With roll")
        P, X = estimate(
            P,
            X,
            contours,
            angles,
            origin,
            kappa,
            reference_image,
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
