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


def make_bounds(
    P, X, reference_image=None, refine_yaw=False, refine_pitch=False, refine_roll=False
):
    if refine_yaw:
        yaw = 30
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


class TargetBase:
    """
    Base class for refinement target

    """

    cols: list = []

    def __init__(
        self, P: np.ndarray, X: np.ndarray, image_size: tuple, contours: np.ndarray
    ):
        """
        Initialise the target

        """
        self.P = P
        self.X = X
        self.image_size = image_size
        self.contours = contours

    def flatten_parameters(self, P: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Flatten the parameters to refine

        """
        return np.concatenate([P[:, self.cols].flatten(), X.flatten()])

    def unflatten_parameters(self, params: np.ndarray) -> tuple:
        """
        Unflatten the parameters to refine

        """

        # The number of image parameters to select
        size = self.P.shape[0] * len(self.cols)

        # Get the column parameters
        P_cols, params = params[:size], params[size:]
        P_cols = P_cols.reshape((-1, len(self.cols)))

        # Get the point parameters
        X = params.reshape(self.X.shape)

        # Construct the image parameters
        P = self.P.copy()
        P[:, self.cols] = P_cols[:, :]

        # Return the image and point parameters
        return P, X

    def residuals(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the least squares residuals

        """

        # Unflatten the parameters
        P, X = self.unflatten_parameters(params)

        # Stored variables
        image_size = self.image_size
        contours = self.contours

        # The transformation
        a = P[:, 0]  # Yaw
        b = P[:, 1]  # Pitch
        c = P[:, 2]  # Roll
        dy = P[:, 3]  # Shift Y
        dx = P[:, 4]  # Shift X

        # The centre of the image
        centre = np.array([image_size[1] / 2, 0, image_size[0] / 2])

        # Get index, z, y and x from the contours
        index = contours["index"]  # Point index
        z = contours["z"]  # Image number
        y_obs = contours["y"]  # Observed y coordinate on image
        x_obs = contours["x"]  # Observed x coordinate on image

        # Do the predictions
        y_prd, x_prd = self.predict(a, b, c, dy, dx, X, centre, index, z)

        # Compute the residuals
        r = np.zeros((contours.shape[0], 2))
        r[:, 0] = y_prd - y_obs
        r[:, 1] = x_prd - x_obs

        # Return the residuals
        print(np.sqrt(np.mean(r**2)))
        return r.flatten()

    @staticmethod
    def predict(
        a: np.array,
        b: np.ndarray,
        c: np.ndarray,
        dy: np.ndarray,
        dx: np.ndarray,
        X: np.ndarray,
        centre: np.ndarray,
        index: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the positions of the points on the images

        """

        # Create the translation vector for each image
        T = np.stack([dx, np.zeros(dx.size), dy], axis=1)

        # Create the rotation matrix for each image
        Ra, Rb, Rc = TargetBase.compute_rotation_matrices(a, b, c)

        # The full rotation matrix
        R = Ra @ Rb @ Rc

        # Compute the predicted positions
        p = np.einsum("...ij,...j", R[z], X[index]) + T[z] + centre

        # Return y_prd and x_prd
        y_prd = p[:, 2]  # (p . (0, 0, 1))
        x_prd = p[:, 0]  # (p . (1, 0, 0))
        return y_prd, x_prd

    @staticmethod
    def compute_rotation_matrices(
        a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
        """
        Compute the rotation matrices

        """
        # Create the rotation matrix for each image
        Ra = Rotation.from_rotvec(np.array((0, 1, 0))[None, :] * a[:, None]).as_matrix()
        Rb = Rotation.from_rotvec(np.array((1, 0, 0))[None, :] * b[:, None]).as_matrix()
        Rc = Rotation.from_rotvec(np.array((0, 0, 1))[None, :] * c[:, None]).as_matrix()

        # Return the rotation matrices
        return Ra, Rb, Rc

    @staticmethod
    def d_da(
        a: np.ndarray,
        Rb: np.ndarray,
        Rc: np.ndarray,
        X: np.ndarray,
        index: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the derivatives wrt yaw

        """
        # Compute cos and sin of yaw
        cosa = np.cos(a)
        sina = np.sin(a)

        # Derivative wrt yaw
        # [[-sina, 0,  cosa],
        #  [    0, 0,     0],
        #  [-cosa, 0, -sina]]
        dRa_da = np.zeros((a.shape[0], 3, 3))
        dRa_da[:, 0, 0] = -sina
        dRa_da[:, 0, 2] = cosa
        dRa_da[:, 2, 0] = -cosa
        dRa_da[:, 2, 2] = -sina

        # Derivatives of full rotation matrix wrt yaw
        dR_da = dRa_da @ Rb @ Rc

        # Derivatives of p wrt to yaw
        dp_da = np.einsum("...ij,...j", dR_da[z], X[index])

        # Derivatives wrt yaw
        dy_prd_da = dp_da[:, 2]  # dy_prd_da = dp_da . (0, 0, 1)
        dx_prd_da = dp_da[:, 0]  # dx_prd_da = dp_da . (1, 0, 0)

        # Return the derivatives
        return dy_prd_da, dx_prd_da

    @staticmethod
    def d_db(
        Ra: np.ndarray,
        b: np.ndarray,
        Rc: np.ndarray,
        X: np.ndarray,
        index: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the derivatives wrt pitch

        """
        # Cos and sin of pitch
        cosb = np.cos(b)
        sinb = np.sin(b)

        # Derivative wrt pitch
        # [[0,     0,     0],
        #  [0, -sinb, -cosb],
        #  [0,  cosb, -sinb]]
        dRb_db = np.zeros((b.shape[0], 3, 3))
        dRb_db[:, 1, 1] = -sinb
        dRb_db[:, 1, 2] = -cosb
        dRb_db[:, 2, 1] = cosb
        dRb_db[:, 2, 2] = -sinb

        # Derivatives of full rotation matrix wrt pitch
        dR_db = Ra @ dRb_db @ Rc

        # Derivatives of p wrt to pitch
        dp_db = np.einsum("...ij,...j", dR_db[z], X[index])

        # Derivatives wrt pitch
        dy_prd_db = dp_db[:, 2]  # dy_prd_db = dp_db . (0, 0, 1)
        dx_prd_db = dp_db[:, 0]  # dx_prd_db = dp_db . (1, 0, 0)

        # Return the derivatives
        return dy_prd_db, dx_prd_db

    @staticmethod
    def d_dc(
        Ra: np.ndarray,
        Rb: np.ndarray,
        c: np.ndarray,
        X: np.ndarray,
        index: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the derivatives wrt roll

        """
        # Cos and sin of roll
        cosc = np.cos(c)
        sinc = np.sin(c)

        # Derivative wrt roll
        # [[-sinc, -cosc, 0],
        #  [ cosc, -sinc, 0],
        #  [    0,     0, 0]]
        dRc_dc = np.zeros((c.shape[0], 3, 3))
        dRc_dc[:, 0, 0] = -sinc
        dRc_dc[:, 0, 1] = -cosc
        dRc_dc[:, 1, 0] = cosc
        dRc_dc[:, 1, 1] = -sinc

        # Derivatives of full rotation matrix wrt yaw, pitch and roll
        dR_dc = Ra @ Rb @ dRc_dc

        # Derivatives of p wrt to rotations
        dp_dc = np.einsum("...ij,...j", dR_dc[z], X[index])

        # Derivatives wrt roll
        dy_prd_dc = dp_dc[:, 2]  # dy_prd_dc = dp_dc . (0, 0, 1)
        dx_prd_dc = dp_dc[:, 0]  # dx_prd_dc = dp_dc . (1, 0, 0)

        # Return the derivatives
        return dy_prd_dc, dx_prd_dc

    @staticmethod
    def d_dX(R: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute the derivatives wrt X

        """
        # Derivatives of p wrt to point parameters
        dp_dX0 = R[z, :, 0]  # R[z] @ dp0_dX0 = R[z] @ (1, 0, 0)
        dp_dX1 = R[z, :, 1]  # R[z] @ dp0_dX1 = R[z] @ (0, 1, 0)
        dp_dX2 = R[z, :, 2]  # R[z] @ dp0_dX2 = R[z] @ (0, 0, 1)

        # Derivatives wrt point parameters
        dy_prd_dX0 = dp_dX0[:, 2]  # dy_prd_dX0 = dp_dX0 . (0, 0, 1)
        dy_prd_dX1 = dp_dX1[:, 2]  # dy_prd_dX1 = dp_dX1 . (0, 0, 1)
        dy_prd_dX2 = dp_dX2[:, 2]  # dy_prd_dX2 = dp_dX2 . (0, 0, 1)
        dx_prd_dX0 = dp_dX0[:, 0]  # dx_prd_dX0 = dp_dX0 . (1, 0, 0)
        dx_prd_dX1 = dp_dX1[:, 0]  # dx_prd_dX1 = dp_dX1 . (1, 0, 0)
        dx_prd_dX2 = dp_dX2[:, 0]  # dx_prd_dX2 = dp_dX2 . (1, 0, 0)

        # Return the derivatives
        return (dy_prd_dX0, dy_prd_dX1, dy_prd_dX2), (
            dx_prd_dX0,
            dx_prd_dX1,
            dx_prd_dX2,
        )


class TargetDyDx(TargetBase):
    """
    Target class for Dy and Dx refinement

    """

    # Select the columns for refinement
    cols = [3, 4]

    def jacobian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the jacobian

        """
        # Unflatten the parameters
        P, X = self.unflatten_parameters(params)

        # The transformation
        a = P[:, 0]  # Yaw
        b = P[:, 1]  # Pitch
        c = P[:, 2]  # Roll

        # Create the rotation matrix for each image
        Ra, Rb, Rc = self.compute_rotation_matrices(a, b, c)

        # The full rotation matrix
        R = Ra @ Rb @ Rc

        # Get index, z, y and x from the contours
        index = self.contours["index"]  # Point index
        z = self.contours["z"]  # Image number

        # Initialise the elements of the Jacobian
        JP = np.zeros((self.contours.shape[0], 2, P.shape[0], 2))
        JX = np.zeros((self.contours.shape[0], 2, X.shape[0], X.shape[1]))

        # The array of indices
        i = np.arange(z.size)

        # Derivatives wrt dy and dx
        JP[i, 0, z, 0] = 1  # dy_prd_dy = dp_dy . (0, 0, 1) = (0, 0, 1) . (0, 0, 1)
        JP[i, 1, z, 0] = 0  # dx_prd_dy = dp_dy . (1, 0, 0) = (0, 0, 1) . (1, 0, 0)
        JP[i, 0, z, 1] = 0  # dy_prd_dx = dp_dx . (0, 0, 1) = (1, 0, 0) . (0, 0, 1)
        JP[i, 1, z, 1] = 1  # dx_prd_dx = dp_dx . (1, 0, 0) = (1, 0, 0) . (1, 0, 0)

        # Derivatives wrt point parameters
        d_dX = self.d_dX(R, z)
        JX[i, 0, index, 0] = d_dX[0][0]  # dy_prd_dX0
        JX[i, 0, index, 1] = d_dX[0][1]  # dy_prd_dX1
        JX[i, 0, index, 2] = d_dX[0][2]  # dy_prd_dX2
        JX[i, 1, index, 0] = d_dX[1][0]  # dx_prd_dX0
        JX[i, 1, index, 1] = d_dX[1][1]  # dx_prd_dX1
        JX[i, 1, index, 2] = d_dX[1][2]  # dx_prd_dX2

        # Return the Jacobian
        JP = JP.reshape((self.contours.shape[0] * 2, P.shape[0] * 2))
        JX = JX.reshape((self.contours.shape[0] * 2, X.size))
        J = np.concatenate([JP, JX], axis=1)
        return J


class TargetADyDx(TargetBase):
    """
    Target class for A, Dy and Dx refinement

    """

    # Select the columns for refinement
    cols = [0, 3, 4]

    def jacobian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the jacobian

        """
        # Unflatten the parameters
        P, X = self.unflatten_parameters(params)

        # The transformation
        a = P[:, 0]  # Yaw
        b = P[:, 1]  # Pitch
        c = P[:, 2]  # Roll

        # Create the rotation matrix for each image
        Ra, Rb, Rc = self.compute_rotation_matrices(a, b, c)

        # The full rotation matrix
        R = Ra @ Rb @ Rc

        # Get index, z, y and x from the contours
        index = self.contours["index"]  # Point index
        z = self.contours["z"]  # Image number

        # Initialise the elements of the Jacobian
        JP = np.zeros((self.contours.shape[0], 2, P.shape[0], len(self.cols)))
        JX = np.zeros((self.contours.shape[0], 2, X.shape[0], X.shape[1]))

        # The array of indices
        i = np.arange(z.size)

        # Derivatives wrt yaw
        d_da = self.d_da(a, Rb, Rc, X, index, z)
        JP[i, 0, z, 0] = d_da[0]  # dy_prd_da
        JP[i, 1, z, 0] = d_da[1]  # dx_prd_da

        # Derivatives wrt dy and dx
        JP[i, 0, z, 1] = 1  # dy_prd_dy = dp_dy . (0, 0, 1) = (0, 0, 1) . (0, 0, 1)
        JP[i, 1, z, 1] = 0  # dx_prd_dy = dp_dy . (1, 0, 0) = (0, 0, 1) . (1, 0, 0)
        JP[i, 0, z, 2] = 0  # dy_prd_dx = dp_dx . (0, 0, 1) = (1, 0, 0) . (0, 0, 1)
        JP[i, 1, z, 2] = 1  # dx_prd_dx = dp_dx . (1, 0, 0) = (1, 0, 0) . (1, 0, 0)

        # Derivatives wrt point parameters
        d_dX = self.d_dX(R, z)
        JX[i, 0, index, 0] = d_dX[0][0]  # dy_prd_dX0
        JX[i, 0, index, 1] = d_dX[0][1]  # dy_prd_dX1
        JX[i, 0, index, 2] = d_dX[0][2]  # dy_prd_dX2
        JX[i, 1, index, 0] = d_dX[1][0]  # dx_prd_dX0
        JX[i, 1, index, 1] = d_dX[1][1]  # dx_prd_dX1
        JX[i, 1, index, 2] = d_dX[1][2]  # dx_prd_dX2

        # Return the Jacobian
        JP = JP.reshape((self.contours.shape[0] * 2, P.shape[0] * len(self.cols)))
        JX = JX.reshape((self.contours.shape[0] * 2, X.size))
        J = np.concatenate([JP, JX], axis=1)
        return J


class TargetABDyDx(TargetBase):
    """
    Target class for A, B, Dy and Dx refinement

    """

    # Select the columns for refinement
    cols = [0, 1, 3, 4]

    def jacobian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the jacobian

        """
        # Unflatten the parameters
        P, X = self.unflatten_parameters(params)

        # The transformation
        a = P[:, 0]  # Yaw
        b = P[:, 1]  # Pitch
        c = P[:, 2]  # Roll

        # Create the rotation matrix for each image
        Ra, Rb, Rc = self.compute_rotation_matrices(a, b, c)

        # The full rotation matrix
        R = Ra @ Rb @ Rc

        # Get index, z, y and x from the contours
        index = self.contours["index"]  # Point index
        z = self.contours["z"]  # Image number

        # Initialise the elements of the Jacobian
        JP = np.zeros((self.contours.shape[0], 2, P.shape[0], len(self.cols)))
        JX = np.zeros((self.contours.shape[0], 2, X.shape[0], X.shape[1]))

        # The array of indices
        i = np.arange(z.size)

        # Derivatives wrt yaw
        d_da = self.d_da(a, Rb, Rc, X, index, z)
        JP[i, 0, z, 0] = d_da[0]  # dy_prd_da
        JP[i, 1, z, 0] = d_da[1]  # dx_prd_da

        # Derivatives wrt pitch
        d_db = self.d_db(Ra, b, Rc, X, index, z)
        JP[i, 0, z, 1] = d_db[0]  # dy_prd_db
        JP[i, 1, z, 1] = d_db[1]  # dx_prd_db

        # Derivatives wrt dy and dx
        JP[i, 0, z, 2] = 1  # dy_prd_dy = dp_dy . (0, 0, 1) = (0, 0, 1) . (0, 0, 1)
        JP[i, 1, z, 2] = 0  # dx_prd_dy = dp_dy . (1, 0, 0) = (0, 0, 1) . (1, 0, 0)
        JP[i, 0, z, 3] = 0  # dy_prd_dx = dp_dx . (0, 0, 1) = (1, 0, 0) . (0, 0, 1)
        JP[i, 1, z, 3] = 1  # dx_prd_dx = dp_dx . (1, 0, 0) = (1, 0, 0) . (1, 0, 0)

        # Derivatives wrt point parameters
        d_dX = self.d_dX(R, z)
        JX[i, 0, index, 0] = d_dX[0][0]  # dy_prd_dX0
        JX[i, 0, index, 1] = d_dX[0][1]  # dy_prd_dX1
        JX[i, 0, index, 2] = d_dX[0][2]  # dy_prd_dX2
        JX[i, 1, index, 0] = d_dX[1][0]  # dx_prd_dX0
        JX[i, 1, index, 1] = d_dX[1][1]  # dx_prd_dX1
        JX[i, 1, index, 2] = d_dX[1][2]  # dx_prd_dX2

        # Return the Jacobian
        JP = JP.reshape((self.contours.shape[0] * 2, P.shape[0] * len(self.cols)))
        JX = JX.reshape((self.contours.shape[0] * 2, X.size))
        J = np.concatenate([JP, JX], axis=1)
        return J


class TargetABCDyDx(TargetBase):
    """
    Target class for A, B, C, Dy and Dx refinement

    """

    # Select the columns for refinement
    cols = [0, 1, 2, 3, 4]

    def jacobian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the jacobian

        """
        # Unflatten the parameters
        P, X = self.unflatten_parameters(params)

        # The transformation
        a = P[:, 0]  # Yaw
        b = P[:, 1]  # Pitch
        c = P[:, 2]  # Roll

        # Create the rotation matrix for each image
        Ra, Rb, Rc = self.compute_rotation_matrices(a, b, c)

        # The full rotation matrix
        R = Ra @ Rb @ Rc

        # Get index, z, y and x from the contours
        index = self.contours["index"]  # Point index
        z = self.contours["z"]  # Image number

        # Initialise the elements of the Jacobian
        JP = np.zeros((self.contours.shape[0], 2, P.shape[0], P.shape[1]))
        JX = np.zeros((self.contours.shape[0], 2, X.shape[0], X.shape[1]))

        # The array of indices
        i = np.arange(z.size)

        # Derivatives wrt yaw
        d_da = self.d_da(a, Rb, Rc, X, index, z)
        JP[i, 0, z, 0] = d_da[0]  # dy_prd_da
        JP[i, 1, z, 0] = d_da[1]  # dx_prd_da

        # Derivatives wrt pitch
        d_db = self.d_db(Ra, b, Rc, X, index, z)
        JP[i, 0, z, 1] = d_db[0]  # dy_prd_db
        JP[i, 1, z, 1] = d_db[1]  # dx_prd_db

        # Derivatives wrt roll
        d_dc = self.d_dc(Ra, Rb, c, X, index, z)
        JP[i, 0, z, 2] = d_dc[0]  # dy_prd_dc
        JP[i, 1, z, 2] = d_dc[1]  # dx_prd_dc

        # Derivatives wrt dy and dx
        JP[i, 0, z, 3] = 1  # dy_prd_dy = dp_dy . (0, 0, 1) = (0, 0, 1) . (0, 0, 1)
        JP[i, 1, z, 3] = 0  # dx_prd_dy = dp_dy . (1, 0, 0) = (0, 0, 1) . (1, 0, 0)
        JP[i, 0, z, 4] = 0  # dy_prd_dx = dp_dx . (0, 0, 1) = (1, 0, 0) . (0, 0, 1)
        JP[i, 1, z, 4] = 1  # dx_prd_dx = dp_dx . (1, 0, 0) = (1, 0, 0) . (1, 0, 0)

        # Derivatives wrt point parameters
        d_dX = self.d_dX(R, z)
        JX[i, 0, index, 0] = d_dX[0][0]  # dy_prd_dX0
        JX[i, 0, index, 1] = d_dX[0][1]  # dy_prd_dX1
        JX[i, 0, index, 2] = d_dX[0][2]  # dy_prd_dX2
        JX[i, 1, index, 0] = d_dX[1][0]  # dx_prd_dX0
        JX[i, 1, index, 1] = d_dX[1][1]  # dx_prd_dX1
        JX[i, 1, index, 2] = d_dX[1][2]  # dx_prd_dX2

        # Return the Jacobian
        JP = JP.reshape((self.contours.shape[0] * 2, P.size))
        JX = JX.reshape((self.contours.shape[0] * 2, X.size))
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

    if not refine_yaw:
        target = TargetDyDx(P, X, image_size, contours)
    elif refine_yaw and not refine_pitch:
        target = TargetADyDx(P, X, image_size, contours)
    elif refine_pitch and not refine_roll:
        target = TargetABDyDx(P, X, image_size, contours)
    else:
        target = TargetABCDyDx(P, X, image_size, contours)

    # Flatten the parameters
    params = target.flatten_parameters(P, X)

    # Do the optimisation
    result = scipy.optimize.least_squares(
        target.residuals, params, jac=target.jacobian
    )  # , bounds=(lower, upper))
    rmsd = np.sqrt(result.cost / len(contours))

    # Unflatten the parameters
    P, X = target.unflatten_parameters(result.x)

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

    print("with translation")
    P, X, rmsd = estimate(
        P,
        X,
        contours,
        image_size,
        reference_image,
        refine_yaw=False,
        refine_pitch=False,
        refine_roll=False,
    )

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
