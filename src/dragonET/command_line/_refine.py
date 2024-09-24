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
        "--contours",
        type=str,
        default=None,
        dest="contours",
        required=True,
        help=(
            """
            A YAML file containing contour information.
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
        choices=["bc", "c", "none"],
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
        args.contours,
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


def residuals(parameters, active, W, M):
    """
    The refinement residuals

    """

    # Get parameters
    dx, dy, a, b, c = parameters

    # Get num frames and num points
    num_frames = W.shape[0]
    num_points = W.shape[1]

    # Get the rotation matrices
    Rabc = Rotation.from_euler("yxz", np.stack([c, b, a], axis=1)).as_matrix()
    R = np.concatenate([Rabc[:, 0, :], Rabc[:, 1, :]], axis=0)

    # Get the translation
    t = np.concatenate([dx, dy], axis=0)

    # Subtract the translation
    W = W - t[:, None]

    # For each point, compute the residuals
    C = np.zeros(3)
    r = []
    for j in range(num_points):
        Mj = M[:, j]
        W0 = W[Mj, j]
        Rj = R[Mj, :]
        Sj = np.linalg.inv(Rj.T @ Rj) @ Rj.T @ W0
        W1 = Rj @ Sj
        C += Sj
        r.extend((W0 - W1))
    r = np.array(r)

    print(
        "First image: dx=%.1f, dy=%.1f, a= %.1f, b=%.1f, c=%.1f; RMSD: %.3f; Centering: %.3f"
        % (
            dx[0],
            dy[0],
            np.degrees(a[0]),
            np.degrees(b[0]),
            np.degrees(c[0]),
            np.sqrt(np.mean(r**2)),
            np.sqrt(np.mean((C / num_points) ** 2)),
        )
    )

    # Add the centroids to the residuals
    r = np.concatenate([r, C])

    # Return the residuals and the regularisation
    return r


def penalties(parameters, active, W, M):
    """
    Penalty functions

    """
    dx, dy, a, b, c = parameters

    refine = "a"
    if np.count_nonzero(active[3, :]) > 0:
        refine += "b"
    if np.count_nonzero(active[4, :]) > 0:
        refine += "c"

    # Add the regularisations:
    # For b in to close to zero
    # For a to vary smoothly
    # For b to vary smoothly
    # For c to vary smoothly
    return np.concatenate(
        [
            # [100*np.degrees(b[45])] if "b" in refine else [],
            # 0.0*np.degrees(b) if "b" in refine else [],
            np.degrees(a[:-2] - 2 * a[1:-1] + a[2:]) if "a" in refine else [],
            np.degrees(b[:-2] - 2 * b[1:-1] + b[2:]) if "b" in refine else [],
            np.degrees(c[:-2] - 2 * c[1:-1] + c[2:]) if "c" in refine else [],
        ]
    )


def d_dt(dx, dy, a, b, c, W, M):
    # Get num frames and num points
    num_params = W.shape[0]
    num_points = W.shape[1]

    # Get the rotation matrices
    Ra = Rotation.from_euler("z", a).as_matrix()
    Rb = Rotation.from_euler("x", b).as_matrix()
    Rc = Rotation.from_euler("y", c).as_matrix()
    Rabc = Ra @ Rb @ Rc

    # Construct the rotation matrices
    R = np.concatenate([Rabc[:, 0, :], Rabc[:, 1, :]], axis=0)

    # Initialise the derivatives of the centroid w.r.t the parameters
    dC_dt = np.zeros((3, num_params))

    # For each point add the elements of the Jacobian
    J = []
    for j in range(num_points):
        # Get the mask, observations, rotation matrices
        Mj = M[:, j]
        Nj = np.count_nonzero(Mj)
        W0 = W[Mj, j]
        Rj = R[Mj, :]
        Qj = np.linalg.inv(Rj.T @ Rj) @ Rj.T

        # Compute the derivative of the residuals w.r.t dx and dy
        dtj_dt = np.identity(Nj)
        dSj_dt = -Qj @ dtj_dt
        drj_dt = -dtj_dt - Rj @ dSj_dt

        # We need to put these subset of the results into an array with
        # zeros for the other frames
        dr_dt = np.zeros((Nj, num_params))
        dS_dt = np.zeros((3, num_params))
        dr_dt[:, Mj] = drj_dt
        dS_dt[:, Mj] = dSj_dt

        # Add the derivatives to the centroid derivative
        dC_dt += dS_dt

        # Add the derivatives of the residuals w.r.t dy and dx
        J.extend(dr_dt)

    # Add the derivatives of the residuals w.r.t the centroid
    J.extend(dC_dt)

    # Return as a numpy array
    return np.array(J)


def d_dp(Rabc, dRabc_dp, W, M):
    # Get num frames and num points
    num_frames = W.shape[0] // 2
    num_points = W.shape[1]

    # Construct the rotation and derivative matrices
    R = np.concatenate([Rabc[:, 0, :], Rabc[:, 1, :]], axis=0)
    dR_dp = np.concatenate([dRabc_dp[:, 0, :], dRabc_dp[:, 1, :]], axis=0)

    # Initialise the derivatives of the centroid w.r.t the parameters
    dC_dp = np.zeros((3, num_frames))

    # For each point add the elements of the Jacobian
    J = []
    for j in range(num_points):
        # Get the mask, observations, rotation matrices and derivatives
        Mj = M[:, j]
        W0 = W[Mj, j]
        Rj = R[Mj, :]
        dRj_dp = dR_dp[Mj, :]

        # Construct an array of the derivatives of Rj w.r.t a
        num_frames_j = Rj.shape[0] // 2
        i1 = np.arange(num_frames_j)
        i2 = i1 + num_frames_j
        dRj_dp_i = np.zeros((num_frames_j, dRj_dp.shape[0], dRj_dp.shape[1]))
        dRj_dp_i[i1, i1] = dRj_dp[i1]
        dRj_dp_i[i1, i2] = dRj_dp[i2]

        # Prepare the transposes and ensure the correct array dimensions
        # for broadcasting
        Rj, RjT = Rj[None, :, :], Rj.T[None, :, :]
        dRj_dp_iT = np.transpose(dRj_dp_i, (0, 2, 1))

        # Compute the derivative of the residuals w.r.t a
        RjTRj_inv = np.linalg.inv(RjT @ Rj)
        H1 = dRj_dp_i @ RjTRj_inv @ RjT
        H2 = Rj @ RjTRj_inv @ dRj_dp_iT
        H3 = Rj @ RjTRj_inv @ RjT
        dr_dp_i = -(H1 + H2 - (H2 @ H3 + H3 @ H1)) @ W0
        # dr_dp_i = -(H1 + H2) @ W0

        # We need to put these subset of the results into an array with
        # zeros for the other frames
        dr_dp = np.zeros((dr_dp_i.shape[1], num_frames))
        dr_dp[:, Mj[:num_frames]] = dr_dp_i.T

        # Compute the derivative of S w.r.t the parameter
        I1 = -RjTRj_inv @ dRj_dp_iT @ Rj @ RjTRj_inv @ RjT
        I2 = -RjTRj_inv @ RjT @ dRj_dp_i @ RjTRj_inv @ RjT
        I3 = RjTRj_inv @ dRj_dp_iT
        dS_dp_i = (I1 + I2 + I3) @ W0
        dS_dp = np.zeros((3, num_frames))
        dS_dp[:, Mj[:num_frames]] = dS_dp_i.T

        # Add the derivatives to the derivatives of the centroid
        dC_dp += dS_dp

        # Add the derivatives of the residuals w.r.t all a
        J.extend(dr_dp)

    # Add the derivatives of the residuals w.r.t the centroid
    J.extend(dC_dp)

    # Return as a numpy array
    return np.array(J)


def d_da(dx, dy, a, b, c, W, M):
    # Get num frames and num points
    num_frames = a.shape[0]
    num_points = W.shape[1]

    # Get the rotation matrices
    Ra = Rotation.from_euler("z", a).as_matrix()
    Rb = Rotation.from_euler("x", b).as_matrix()
    Rc = Rotation.from_euler("y", c).as_matrix()
    Rabc = Ra @ Rb @ Rc

    # Compute the derivative of Ra w.r.t a
    dRa_da = np.zeros((num_frames, 3, 3))
    dRa_da[:, 0, 0] = -np.sin(a)
    dRa_da[:, 0, 1] = -np.cos(a)
    dRa_da[:, 1, 0] = np.cos(a)
    dRa_da[:, 1, 1] = -np.sin(a)
    dRabc_da = dRa_da @ Rb @ Rc

    # Compute derivatices of residuals w.r.t a
    return d_dp(Rabc, dRabc_da, W, M)


def d_db(dx, dy, a, b, c, W, M):
    # Get num frames and num points
    num_frames = a.shape[0]
    num_points = W.shape[1]

    # Get the rotation matrices
    Ra = Rotation.from_euler("z", a).as_matrix()
    Rb = Rotation.from_euler("x", b).as_matrix()
    Rc = Rotation.from_euler("y", c).as_matrix()
    Rabc = Ra @ Rb @ Rc

    # Compute the derivative of Ra w.r.t a
    dRb_db = np.zeros((num_frames, 3, 3))
    dRb_db[:, 1, 1] = -np.sin(b)
    dRb_db[:, 1, 2] = -np.cos(b)
    dRb_db[:, 2, 1] = np.cos(b)
    dRb_db[:, 2, 2] = -np.sin(b)
    dRabc_db = Ra @ dRb_db @ Rc

    # Compute derivatices of residuals w.r.t b
    return d_dp(Rabc, dRabc_db, W, M)


def d_dc(dx, dy, a, b, c, W, M):
    # Get num frames and num points
    num_frames = a.shape[0]
    num_points = W.shape[1]

    # Get the rotation matrices
    Ra = Rotation.from_euler("z", a).as_matrix()
    Rb = Rotation.from_euler("x", b).as_matrix()
    Rc = Rotation.from_euler("y", c).as_matrix()
    Rabc = Ra @ Rb @ Rc

    # Compute the derivative of Ra w.r.t a
    dRc_dc = np.zeros((num_frames, 3, 3))
    dRc_dc[:, 0, 0] = -np.sin(c)
    dRc_dc[:, 2, 0] = -np.cos(c)
    dRc_dc[:, 0, 2] = np.cos(c)
    dRc_dc[:, 2, 2] = -np.sin(c)
    dRabc_dc = Ra @ Rb @ dRc_dc

    # Compute derivatices of residuals w.r.t c
    return d_dp(Rabc, dRabc_dc, W, M)


def jacobian(parameters, active, W, M):
    """
    The Jacobian

    """

    # Check which parameters are to be refined
    refine = "a"
    if np.count_nonzero(active[3, :]) > 0:
        refine += "b"
    if np.count_nonzero(active[4, :]) > 0:
        refine += "c"
    if refine == "a":
        active = active[0:3, :]
    elif refine == "ab":
        active = active[0:4, :]

    # The derivatives
    derivatives = {
        "abc": [d_dt, d_da, d_db, d_dc],  # Derivatives w.r.t y, x, a, b and c
        "ab": [d_dt, d_da, d_db],  # Derivatives w.r.t y, x, a and b
        "a": [d_dt, d_da],  # Derivatives w.r.t y, x and a
    }

    # Get the parameters
    dx, dy, a, b, c = parameters

    # Get the translation
    t = np.concatenate([dx, dy], axis=0)

    # Subtract the translation
    W = W - t[:, None]

    # Compute the Jacobian elements
    J = np.concatenate(
        [d_dp(dx, dy, a, b, c, W, M) for d_dp in derivatives[refine]], axis=1
    )

    # Take the active derivatives
    return J[:, active.flatten()]


def jacobian_penalties(parameters, active, W, M):
    """
    The Jacobian of the penalty functions

    """

    dx, dy, a, b, c = parameters

    refine = "a"
    if np.count_nonzero(active[3, :]) > 0:
        refine += "b"
    if np.count_nonzero(active[4, :]) > 0:
        refine += "c"
    if refine == "a":
        active = active[0:3, :]
    elif refine == "ab":
        active = active[0:4, :]

    # Indices
    a0 = dy.shape[0] + dx.shape[0]
    a1 = b0 = a0 + a.shape[0]
    b1 = c0 = b0 + b.shape[0]
    c1 = c0 + c.shape[0]

    # Get the number of parameters
    num_params = {None: c1, "c": b1, "bc": a1}[restrain]

    # Derivative of angle w.r.t angle
    da_da = np.identity(a.shape[0])
    db_db = np.identity(b.shape[0])
    dc_dc = np.identity(c.shape[0])

    def db_dp():
        J = np.zeros((db_db.shape[0], num_params))
        J[:, b0:b1] = db_db
        return J

    def dfa_dp():
        J = np.zeros((da_da.shape[0] - 2, num_params))
        J[:, a0:a1] = da_da[:-2, :] - 2 * da_da[1:-1, :] + da_da[2:, :]
        return J

    def dfb_dp():
        J = np.zeros((db_db.shape[0] - 2, num_params))
        J[:, b0:b1] = db_db[:-2, :] - 2 * db_db[1:-1, :] + db_db[2:, :]
        return J

    def dfc_dp():
        J = np.zeros((dc_dc.shape[0] - 2, num_params))
        J[:, c0:c1] = dc_dc[:-2, :] - 2 * dc_dc[1:-1, :] + dc_dc[2:, :]
        return J

    # Add the regularisations:
    # For a to vary smoothly
    # For b to vary smoothly and be close to zero
    # For c to vary smoothly
    not_none = lambda x: [xx for xx in x if xx is not None]
    J = np.concatenate(
        not_none(
            [
                # 100 * np.degrees(db0_dp) if "b" in refine else None,
                # 0.0*np.degrees(db_dp()) if "b" in refine else None,
                np.degrees(dfa_dp()) if "a" in refine else None,
                np.degrees(dfb_dp()) if "b" in refine else None,
                np.degrees(dfc_dp()) if "c" in refine else None,
            ]
        ),
        axis=0,
    )

    # Return only active parameters
    return J[:, active.flatten()]


def refine_model(
    dx,
    dy,
    a,
    b,
    c,
    data,
    mask,
    restrain=None,
) -> tuple:
    """
    Estimate the parameters using least squares

    """
    print("Refining model with %s restrained" % str(restrain))

    def get_params_and_args(dx, dy, a, b, c, W, M, restrain=None):
        # Get the parameters and the active matrix
        parameters = np.stack([dx, dy, a, b, c])
        active = np.zeros(parameters.shape, dtype=bool)

        # Set the components to be active or not
        assert restrain in [None, "bc", "c"]
        if restrain is None:
            active[:, :] = 1
        elif restrain == "c":
            active[:4, :] = 1
        elif restrain == "bc":
            active[:3, :] = 1

        # Keep b and c fixed for the zero tilt image
        idx = np.argmin(np.abs(parameters[4, :]))
        active[3, idx] = 0
        active[4, idx] = 0

        # Return the parameters
        return parameters[active].flatten(), (parameters, active, W, M)

    def parse_params_and_args(x, parameters, active, W, M):
        parameters[active] = x
        return parameters, active, W, M

    def parse_results(x, parameters, active, W, M):
        parameters[active] = x
        return tuple(parameters)

    # def get_bounds(dx, dy, a, b, c, restrain):

    #     # The parameter bounds
    #     dx_min = np.full(dx.shape, -np.inf)
    #     dx_max = np.full(dx.shape, np.inf)
    #     dy_min = np.full(dy.shape, -np.inf)
    #     dy_max = np.full(dy.shape, np.inf)
    #     a_min = np.full(a.shape, -np.radians(180))
    #     a_max = np.full(a.shape, np.radians(180))
    #     b_min = np.full(a.shape, -np.radians(180))
    #     b_max = np.full(a.shape, np.radians(180))
    #     c_min = np.full(a.shape, -np.radians(180))
    #     c_max = np.full(a.shape, np.radians(180))

    #     # Set zero angle to zero
    #     b_min[45] = -1e-15
    #     b_max[45] = +1e-15

    #     if restrain is None:
    #         bounds_min = np.concatenate([dx_min, dy_min, a_min, b_min, c_min])
    #         bounds_max = np.concatenate([dx_max, dy_max, a_max, b_max, c_max])
    #     elif restrain == "c":
    #         bounds_min = np.concatenate([dx_min, dy_min, a_min, b_min])
    #         bounds_max = np.concatenate([dx_max, dy_max, a_max, b_max])
    #     elif restrain == "bc":
    #         bounds_min = np.concatenate([dx_min, dy_min, a_min])
    #         bounds_max = np.concatenate([dx_max, dy_max, a_max])
    #     return bounds_min, bounds_max

    def fun(x, *args):
        args = parse_params_and_args(x, *args)
        return np.concatenate([residuals(*args)])  # , penalties(*args)])

    def jac(x, *args):
        args = parse_params_and_args(x, *args)
        return np.concatenate(
            [jacobian(*args)]
        )  # , jacobian_penalties(*args)], axis=0)

    # Construct the input
    X = data[:, :, 0]
    Y = data[:, :, 1]
    t = np.concatenate([dx, dy], axis=0)
    M = np.concatenate([mask, mask], axis=0)
    Wc = np.concatenate([X, Y], axis=0)

    # Get the params and arguments
    params, args = get_params_and_args(dx, dy, a, b, c, Wc, M, restrain)

    # Get the bounds
    # bounds = get_bounds(dx, dy, a, b, c, restrain)

    # Perform the least squares minimisation
    result = scipy.optimize.least_squares(
        fun,
        params,
        args=args,
        jac=jac,
        loss="linear",
        # bounds=bounds,
        # max_nfev=1,
        # bounds=[np.radians(-180), np.radians(180)],
    )

    # Get the results
    dx, dy, a, b, c = parse_results(result.x, *args)

    # Compute the RMSD
    rmsd = np.sqrt(result.cost / np.count_nonzero(mask))

    print("RMSD: %f" % rmsd)

    # Return the refined parameters and RMSD
    return dx, dy, a, b, c, rmsd


def _refine(
    model_in: str,
    model_out: str,
    contours: str,
    plots_out: str = None,
    info_out: str = None,
    fix: str = None,
    reference_image: int = None,
    verbose: bool = False,
):
    """
    Do the refinement

    """

    def read_points(filename) -> tuple:
        print("Reading points from %s" % filename)
        handle = np.load(filename)
        return handle["data"], handle["mask"]

    def read_model(filename) -> dict:
        print("Reading model from %s" % filename)
        return yaml.safe_load(open(filename, "r"))

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

    def write_plots(P, directory):
        print("Writing plots to %s" % directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        write_angles_vs_image_number(P, directory)
        write_shift_vs_image_number(P, directory)
        write_xy_shift_distribution(P, directory)

    def write_info(info, filename):
        print("Writing info to %s" % filename)
        yaml.safe_dump(info, open(filename, "w"), default_flow_style=None)

    def get_cycles(fix):
        # Convert to None
        if fix == "none":
            fix = None

        # Check input
        assert fix in ["bc", "c", None]

        # Refine with translation and yaw
        # Then refine with translation, yaw and pitch
        # Then refine with translation, yaw, pitch and roll
        return {
            None: ["bc", "c", None],
            "c": ["bc", "c"],
            "bc": ["bc"],
        }[fix]

    def check_obs_per_image(mask):
        # Check that each image has atleast 4 observations
        obs_per_image = np.count_nonzero(mask, axis=1)
        if np.any(obs_per_image < 3):
            raise RuntimeError(
                "The following images have less than 4 observations: %s"
                % ("\n".join(map(str, np.where(obs_per_image < 4)[0])))
            )

    def check_obs_per_point(mask):
        # Check that each point has at least 3 observations
        obs_per_point = np.count_nonzero(mask, axis=0)
        if np.any(obs_per_point < 2):
            raise RuntimeError(
                "The following points have less than 3 observations: %s"
                % ("\n".join(map(str, np.where(obs_per_point < 2)[0])))
            )

    def check_connections(mask):
        # Create lookup tables to check each point and each image is touched
        r, n = scipy.ndimage.label(mask)
        if n != 1:
            raise RuntimeError(
                "All points and images are not connected: %d regions detected" % (n)
            )

    # Read the model
    model = read_model(model_in)

    # Read the points
    data, mask = read_points(contours)

    # The image size
    image_size = model["image_size"]

    # Read the initial model add 1/2 image size and convert degrees to radians
    P = np.array(model["transform"], dtype=float)
    a = np.radians(P[:, 0])
    b = np.radians(P[:, 1])
    c = np.radians(P[:, 2])
    dy = P[:, 3]
    dx = P[:, 4]
    dy += image_size[0] // 2
    dx += image_size[1] // 2

    # The number of images and points
    assert data.shape[0] == P.shape[0]
    num_images = data.shape[0]
    num_points = data.shape[1]

    # Select only points with 3 or more points
    select = np.count_nonzero(mask, axis=0) >= 3
    data = data[:, select]
    mask = mask[:, select]

    # Perform some checks on the contours
    check_obs_per_image(mask)
    check_obs_per_point(mask)
    # check_connections(mask)
    print("Num images: %d" % num_images)
    print("Num contours: %d" % num_points)
    print("Num observations: %d" % np.count_nonzero(mask))

    # Run through the cycles of refinement
    for restrain in get_cycles(fix):
        dx, dy, a, b, c, rmsd = refine_model(
            dx, dy, a, b, c, data, mask, restrain=restrain
        )

    # Update the model, remove 1/2 image size and convert back to degrees
    dy -= image_size[0] // 2
    dx -= image_size[1] // 2
    P = np.stack([np.degrees(a), np.degrees(b), np.degrees(c), dy, dx], axis=1)
    model["transform"] = P.tolist()

    # Save the refined model
    write_model(model, model_out)

    # Save some plots of the geometry
    if plots_out:
        write_plots(P, plots_out)

    # Save some refinement information
    if info_out:
        info = {
            "rmsd": float(rmsd),
        }
        write_info(info, info_out)


if __name__ == "__main__":
    refine()
