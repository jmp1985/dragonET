#
# refine.py
#
# Copyright (C) 2024 Diamond Light Source and Rosalind Franklin Institute
#
# Author: James Parkhurst
#
import os
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
        "--fix",
        type=str,
        default="c",
        dest="fix",
        choices=["abc", "bc", "c", "none"],
        help="Fix parameters in refinement",
    )
    parser.add_argument(
        "--reference_image",
        type=int,
        default=None,
        dest="reference_image",
        help="Set the reference image, if not set the angle closest to zero will be chosen",
    )
    parser.add_argument(
        "--plots_out",
        type=str,
        default="plots",
        dest="plots_out",
        help=(
            """
            The directory to write some plots
            """
        ),
    )
    parser.add_argument(
        "--info_out",
        type=str,
        default=None,
        dest="info_out",
        help=(
            """
            A YAML file containing refinement information.
            """
        ),
    )
    parser.add_argument(
        "-v",
        type=bool,
        default=False,
        dest="verbose",
        action="store_true",
        help=(
            """
            Set verbose output
            """
        ),
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
        args.plots_out,
        args.info_out,
        args.fix,
        args.reference_image,
        args.verbose,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def refine(args: List[str] = None):
    """
    Refine the model of the sample and align the images

    """
    refine_impl(get_parser().parse_args(args=args))


class Target:
    """
    Base class for refinement target

    """

    def __init__(
        self,
        P: np.ndarray,
        X: np.ndarray,
        image_size: tuple,
        contours: np.ndarray,
        cols: list = None,
        reference_image: int = None,
        verbose: bool = False,
    ):
        """
        Initialise the target

        """
        # Save some stuff
        self.P = P
        self.X = X
        self.image_size = image_size
        self.contours = contours
        self.verbose = verbose

        # Set the reference image. If the given index is None then select the
        # index of the smallest absolute angle
        if reference_image is None:
            self.reference_image = np.argmin(np.abs(P[:, 2]))
        else:
            self.reference_image = reference_image

        # Set the columns
        if cols is None:
            self.cols = list(range(P.shape[1]))
        else:
            self.cols = cols

        # Set the penalty term for the regularisation
        # FIXME - set better values
        self.weight = np.zeros_like(P)
        self.weight[:, 0] = 0  # Yaw
        self.weight[:, 1] = 0  # Pitch
        self.weight[:, 2] = 1  # Roll (tilt)
        self.weight[:, 3] = 1e-5  # Dy
        self.weight[:, 4] = 1e-5  # Dx

        # Regularise on pitch, dy and dx
        self.cols_with_penalty = [i for i, c in enumerate(self.cols) if c in [3, 4]]

        # P lower bounds
        self.P_lower = [
            -np.radians(180),
            -np.radians(90),
            -np.radians(90),
            -np.inf,
            -np.inf,
        ]

        # P upper bounds
        self.P_upper = [
            np.radians(180),
            np.radians(90),
            np.radians(90),
            np.inf,
            np.inf,
        ]

        # X upper bounds
        self.X_lower = -max(image_size)

        # X lower bounds
        self.X_upper = max(image_size)

        # Set the initial parameters
        self._parameters = self.flatten_parameters(self.P, self.X)

    @property
    def parameters(self) -> np.ndarray:
        """
        Get the initial parameters

        """
        return self._parameters

    @property
    def num_parameters(self) -> int:
        """
        Get the number of parameters

        """
        return len(self.parameters)

    @property
    def num_equations(self) -> int:
        """
        Get the number of equations

        """
        return len(self.contours) * 2 + self.P.shape[0] * len(self.cols_with_penalty)

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
        r = r.flatten()

        # Print if verbose
        if self.verbose:
            print(np.sqrt(np.mean(r**2)))

        # Add regularisation terms. Here we assume that the initial parameter
        # estimates are our prior distribution mean values and the weights are
        # the prior distribution inverse variance weights. So we add penalty
        # terms which force the parameter estimates to be close to their
        # initial values
        P_diff = P - self.P
        for col in self.cols_with_penalty:
            r = np.concatenate(
                [r, self.weight[:, self.cols[col]] * P_diff[:, self.cols[col]]]
            )

        # Return the residuals
        return r

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

        # The observation index
        i = np.arange(z.size)

        # Derivatives wrt image parameters
        for j, col in enumerate(self.cols):
            d_dP = self.d_dP(col, a, b, c, Ra, Rb, Rc, X[index], z)
            JP[i, 0, z, j] = d_dP[0]  # dy_prd_dcol
            JP[i, 1, z, j] = d_dP[1]  # dx_prd_dcol

        # Derivatives wrt point parameters
        d_dX = self.d_dX(R, z)
        JX[i, 0, index, 0] = d_dX[0][0]  # dy_prd_dX0
        JX[i, 0, index, 1] = d_dX[0][1]  # dy_prd_dX1
        JX[i, 0, index, 2] = d_dX[0][2]  # dy_prd_dX2
        JX[i, 1, index, 0] = d_dX[1][0]  # dx_prd_dX0
        JX[i, 1, index, 1] = d_dX[1][1]  # dx_prd_dX1
        JX[i, 1, index, 2] = d_dX[1][2]  # dx_prd_dX2

        # Construct the Jacobian
        JP = JP.reshape((self.contours.shape[0] * 2, P.shape[0] * len(self.cols)))
        JX = JX.reshape((self.contours.shape[0] * 2, X.size))
        J = np.concatenate([JP, JX], axis=1)

        # Add Jacobian elements for regularization
        k = np.arange(P.shape[0])
        for col in self.cols_with_penalty:
            J_col = np.zeros(shape=(P.shape[0], J.shape[1]))
            J_col[k, k * len(self.cols) + col] = self.weight[:, self.cols[col]]
            J = np.concatenate([J, J_col], axis=0)

        # Return Jacobian
        return J

    def bounds(self) -> tuple:
        """
        Define the bounds

        """

        # P bounds
        P_lower = np.repeat(
            [np.array(self.P_lower)[self.cols]], self.P.shape[0], axis=0
        )
        P_upper = np.repeat(
            [np.array(self.P_upper)[self.cols]], self.P.shape[0], axis=0
        )

        # Set the bounds for the reference image The tilt angle will be fixed
        # FIXME Better to just not refine these parameters
        for i, c in enumerate(self.cols):
            if c in [2]:
                P_lower[self.reference_image, i] = (
                    self.P[self.reference_image, c] - 1e-15
                )
                P_upper[self.reference_image, i] = (
                    self.P[self.reference_image, c] + 1e-15
                )

        # X lower and upper bounds
        X_lower = np.full(self.X.shape, self.X_lower)
        X_upper = np.full(self.X.shape, self.X_upper)

        # Lower and upper bounds
        lower = np.concatenate([P_lower.flatten(), X_lower.flatten()])
        upper = np.concatenate([P_upper.flatten(), X_upper.flatten()])

        # Return bounds
        return lower, upper

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
    ) -> tuple:
        """
        Predict the positions of the points on the images

        """

        # Create the translation vector for each image
        T = np.stack([dx, np.zeros(dx.size), dy], axis=1)

        # Create the rotation matrix for each image
        Ra, Rb, Rc = Target.compute_rotation_matrices(a, b, c)

        # The full rotation matrix
        R = Ra @ Rb @ Rc

        # Compute the predicted positions
        p = np.einsum("...ij,...j", R[z], X[index]) + T[z] + centre

        # Return y_prd and x_prd
        y_prd = p[:, 2]  # (p . (0, 0, 1))
        x_prd = p[:, 0]  # (p . (1, 0, 0))
        return y_prd, x_prd

    @staticmethod
    def compute_rotation_matrices(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple:
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
        z: np.ndarray,
    ) -> tuple:
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
        dp_da = np.einsum("...ij,...j", dR_da[z], X)

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
        z: np.ndarray,
    ) -> tuple:
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
        dp_db = np.einsum("...ij,...j", dR_db[z], X)

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
        z: np.ndarray,
    ) -> tuple:
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
        dp_dc = np.einsum("...ij,...j", dR_dc[z], X)

        # Derivatives wrt roll
        dy_prd_dc = dp_dc[:, 2]  # dy_prd_dc = dp_dc . (0, 0, 1)
        dx_prd_dc = dp_dc[:, 0]  # dx_prd_dc = dp_dc . (1, 0, 0)

        # Return the derivatives
        return dy_prd_dc, dx_prd_dc

    @staticmethod
    def d_dy() -> tuple:
        """
        Compute the derivatives wrt y shift

        """
        # dy_prd_dy = dp_dy . (0, 0, 1) = (0, 0, 1) . (0, 0, 1)
        # dy_prd_dx = dp_dx . (0, 0, 1) = (1, 0, 0) . (0, 0, 1)
        return 1, 0

    @staticmethod
    def d_dx() -> tuple:
        """
        Compute the derivatives wrt x shift

        """
        # dx_prd_dy = dp_dy . (1, 0, 0) = (0, 0, 1) . (1, 0, 0)
        # dx_prd_dx = dp_dx . (1, 0, 0) = (1, 0, 0) . (1, 0, 0)
        return 0, 1

    @staticmethod
    def d_dP(
        col: int,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        Ra: np.ndarray,
        Rb: np.ndarray,
        Rc: np.ndarray,
        X: np.ndarray,
        z: np.ndarray,
    ) -> tuple:
        """
        Compute derivatices wrt image parameters

        """
        return [
            lambda: Target.d_da(a, Rb, Rc, X, z),
            lambda: Target.d_db(Ra, b, Rc, X, z),
            lambda: Target.d_dc(Ra, Rb, c, X, z),
            lambda: Target.d_dy(),
            lambda: Target.d_dx(),
        ][col]()

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
        return (
            (dy_prd_dX0, dy_prd_dX1, dy_prd_dX2),
            (dx_prd_dX0, dx_prd_dX1, dx_prd_dX2),
        )


def make_target(
    P: np.ndarray,
    X: np.ndarray,
    image_size: np.ndarray,
    contours: np.ndarray,
    fix: list = None,
    reference_image: int = None,
    verbose: bool = False,
):
    """
    Make the refinement target

    """

    # Map the strings to indices
    column_index = {"a": 0, "b": 1, "c": 2, "y": 3, "x": 4}

    # Create the column array
    cols = np.arange(P.shape[1])

    # If we are fixing some items remove them
    if fix:
        for item in fix:
            cols = cols[np.where(cols != column_index[item])[0]]

    # Return the target
    return Target(P, X, image_size, contours, cols, reference_image, verbose)


def refine_model(
    P: np.ndarray,
    X: np.ndarray,
    image_size: np.ndarray,
    contours: np.ndarray,
    fix: list,
    reference_image: int = None,
    verbose: bool = False,
) -> tuple:
    """
    Estimate the parameters using least squares

    """
    print("Refining model with %s fixed" % str(fix))

    # Make the target
    target = make_target(P, X, image_size, contours, fix, reference_image, verbose)

    # Get the initial parameters
    print("Num parameters: %d" % target.num_parameters)
    print("Num equations: %d" % target.num_equations)

    # Check number of parameters
    if target.num_parameters > target.num_equations:
        print("- Warning: more parameters than equations")

    # Do the optimisation
    result = scipy.optimize.least_squares(
        target.residuals,
        target.parameters,
        jac=target.jacobian,
        bounds=target.bounds(),
    )

    # Unflatten the parameters
    P, X = target.unflatten_parameters(result.x)

    # Compute the RMSD
    rmsd = np.sqrt(result.cost / len(contours))

    print("RMSD: %f" % rmsd)

    # Return the refined parameters and RMSD
    return P, X, result.jac, rmsd


def _refine(
    model_in: str,
    model_out: str,
    points_in: str,
    points_out: str = None,
    plots_out: str = None,
    info_out: str = None,
    fix: str = None,
    reference_image: int = None,
    verbose: bool = False,
):
    """
    Do the refinement

    """

    def read_points(filename) -> dict:
        print("Reading points from %s" % filename)
        return yaml.safe_load(open(filename, "r"))

    def read_model(filename) -> dict:
        print("Reading model from %s" % filename)
        return yaml.safe_load(open(filename, "r"))

    def write_points(points, filename):
        print("Writing points to %s" % filename)
        yaml.safe_dump(points, open(filename, "w"), default_flow_style=None)

    def write_model(model, filename):
        print("Writing model to %s" % filename)
        yaml.safe_dump(model, open(filename, "w"), default_flow_style=None)

    def write_angles_vs_image_number(P, directory):
        width = 0.0393701 * 190
        height = (6 / 8) * width
        fig, ax = pylab.subplots(
            ncols=1, figsize=(width, height), constrained_layout=True
        )
        ax.plot(P[:, 0], label="a")
        ax.plot(P[:, 1], label="b")
        ax.plot(P[:, 2], label="c")
        ax.set_xlabel("Image number")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title("Angle vs image number")
        ax.legend()
        fig.savefig(os.path.join(directory, "angles_vs_image_number.png"), dpi=600)

    def write_shift_vs_image_number(P, directory):
        width = 0.0393701 * 190
        height = (6 / 8) * width
        fig, ax = pylab.subplots(
            ncols=1, figsize=(width, height), constrained_layout=True
        )
        ax.plot(P[:, 3], label="y")
        ax.plot(P[:, 4], label="x")
        ax.set_xlabel("Image number")
        ax.set_ylabel("Shift")
        ax.set_title("Shift vs image number")
        ax.legend()
        fig.savefig(os.path.join(directory, "shift_vs_image_number.png"), dpi=600)

    def write_xy_shift_distribution(P, directory):
        width = 0.0393701 * 190
        height = (6 / 8) * width
        fig, ax = pylab.subplots(
            ncols=2,
            figsize=(width, height),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        ax[0].hist(P[:, 3].flatten())
        ax[1].hist(P[:, 4].flatten())
        ax[0].set_xlabel("Y shift")
        ax[1].set_xlabel("X shift")
        fig.suptitle("Distribution of X and Y shifts")
        fig.savefig(os.path.join(directory, "xy_shift_histogram.png"), dpi=600)

    def write_points_distribution(P, directory):
        width = 0.0393701 * 190
        height = (6 / 8) * width
        fig, ax = pylab.subplots(
            ncols=3,
            figsize=(width, height),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        ax[0].hist(X[:, 0].flatten())
        ax[1].hist(X[:, 1].flatten())
        ax[2].hist(X[:, 2].flatten())
        ax[0].set_xlabel("X")
        ax[1].set_xlabel("Y")
        ax[2].set_xlabel("Z")
        fig.suptitle("Distribution of points")
        fig.savefig(os.path.join(directory, "points_histogram.png"), dpi=600)

    def write_covariance_matrix(Sigma, directory):
        width = 0.0393701 * 190
        height = (6 / 8) * width
        fig, ax = pylab.subplots(
            ncols=1,
            figsize=(width, height),
            constrained_layout=True,
            sharex=True,
            sharey=True,
        )
        ax.imshow(Sigma)
        ax.axis("off")
        fig.savefig(os.path.join(directory, "covariance_matrix.png"), dpi=600)

    def write_plots(P, X, Sigma, directory):
        print("Writing plots to %s" % directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        write_angles_vs_image_number(P, directory)
        write_shift_vs_image_number(P, directory)
        write_xy_shift_distribution(P, directory)
        write_points_distribution(X, directory)
        write_covariance_matrix(Sigma, directory)

    def write_info(info, filename):
        print("Writing info to %s" % filename)
        yaml.safe_dump(info, open(filename, "w"), default_flow_style=None)

    def get_contours(points):
        return np.array(
            [tuple(x) for x in points["contours"]],
            dtype=[("index", "int"), ("z", "int"), ("y", "float"), ("x", "float")],
        )

    def get_cycles(fix):
        # Convert to None
        if fix == "none":
            fix = None

        # Check input
        assert fix in ["abc", "bc", "c", None]

        # Refine with just translation
        # Then refine with translation and yaw
        # Then refine with translation, yaw and pitch
        # Then refine with translation, yaw, pitch and roll
        cycles = [["a", "b", "c"]]
        if not fix or "a" not in fix:
            cycles.append(["b", "c"])
        if not fix or "b" not in fix:
            cycles.append(["c"])
        if not fix or "c" not in fix:
            cycles.append([])

        # Return cycles
        return cycles

    def check_num_images(num_images, contours):
        # Check that the number of images matches the number of images
        # represented in the list of contours
        obs_images = len(set(contours["z"]))
        if num_images != obs_images:
            raise RuntimeError(
                "Expected %d images in contours, got %d" % (num_images, obs_images)
            )

    def check_obs_per_image(contours):
        # Check that each image has atleast 2 observations
        obs_per_image = np.bincount(contours["z"])
        if np.any(obs_per_image < 2):
            raise RuntimeError(
                "The following images have less than 2 observations: %s"
                % ("\n".join(map(str, np.where(obs_per_image < 2)[0])))
            )

    def check_obs_per_point(contours):
        # Check that each point has at least 2 observations
        obs_per_point = np.bincount(contours["index"])
        if np.any(obs_per_point < 2):
            raise RuntimeError(
                "The following points have less than 2 observations: %s"
                % ("\n".join(map(str, np.where(obs_per_point < 2)[0])))
            )

    def check_connections(contours):
        # Create lookup tables to check each point and each image is touched
        lookup_image = np.zeros(len(set(contours["z"])))
        lookup_point = np.zeros(len(set(contours["index"])))

        def traverse_point(contours, index):
            if not lookup_point[index]:
                lookup_point[index] = 1
                for z in contours["z"][contours["index"] == index]:
                    traverse_image(contours, z)

        def traverse_image(contours, z):
            if not lookup_image[z]:
                lookup_image[z] = 1
                for index in contours["index"][contours["z"] == z]:
                    traverse_point(contours, index)

        # Traverse the contours and check that each image and each point is
        # linked to all other points and images
        traverse_image(contours, contours["z"][0])

        # Check if any images are not touched
        if np.any(lookup_image == 0):
            raise RuntimeError(
                "The following images are not linked: %s"
                % ("\n".join(map(str, np.where(lookup_image == 0)[0])))
            )

        # Check if any points are not touched
        if np.any(lookup_point == 0):
            raise RuntimeError(
                "The following point are not linked: %s"
                % ("\n".join(map(str, np.where(lookup_point == 0)[0])))
            )

    # Read the model
    model = read_model(model_in)

    # Read the points
    points = read_points(points_in)

    # Get the contours
    contours = get_contours(points)

    # Read the initial model and convert degrees to radians for use here
    P = np.array(model["transform"], dtype=float)
    P[:, 0] = np.radians(P[:, 0])
    P[:, 1] = np.radians(P[:, 1])
    P[:, 2] = np.radians(P[:, 2])

    # The image size
    image_size = model["image_size"]

    # The number of images and points
    num_images = P.shape[0]
    num_points = len(set(contours["index"]))

    # Perform some checks on the contours
    check_num_images(num_images, contours)
    check_obs_per_image(contours)
    check_obs_per_point(contours)
    check_connections(contours)
    print("Num images: %d" % num_images)
    print("Num contours: %d" % num_points)
    print("Num observations: %d" % len(contours))

    # If the coordinates are specified then use these as a starting point
    if "coordinates" in points:
        X = np.array(points["coordinates"], dtype=float)
        assert X.shape[0] == num_points
    else:
        # Initialise the point parameters
        X = np.zeros((num_points, 3))

    # Run through the cycles of refinement
    for fixed in get_cycles(fix):
        P, X, J, rmsd = refine_model(
            P, X, image_size, contours, fixed, reference_image, verbose
        )

    # Update the model and convert back to degrees
    P[:, 0] = np.degrees(P[:, 0])
    P[:, 1] = np.degrees(P[:, 1])
    P[:, 2] = np.degrees(P[:, 2])
    model["transform"] = P.tolist()

    # Compute the covariance matrix of the parameters
    Sigma = np.linalg.inv(J.T @ J) * rmsd**2

    # Update the points
    points["coordinates"] = X.tolist()

    # Save the refined model
    write_model(model, model_out)

    # Save the refined points
    if points_out:
        write_points(points, points_out)

    # Save some plots of the geometry
    if plots_out:
        write_plots(P, X, Sigma, plots_out)

    # Save some refinement information
    if info_out:
        info = {
            "rmsd": float(rmsd),
        }
        write_info(info, info_out)


if __name__ == "__main__":
    refine()
