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


def refine_model(
    a,
    b,
    c,
    dy,
    dx,
    data,
    mask,
    restrain=None,
    max_iter=10,
    teps=0.1,
) -> tuple:
    """
    Estimate the parameters using least squares

    """
    print("Refining model with %s restrained" % str(restrain))

    def residuals(dy, dx, a, b, c, W, M, restrain):
        # Get num frames and num points
        num_frames = W.shape[0]
        num_points = W.shape[1]

        # Get the rotation matrices
        Rabc = Rotation.from_euler("yxz", np.stack([c, b, a], axis=1)).as_matrix()
        R = np.concatenate([Rabc[:, 0, :], Rabc[:, 1, :]], axis=0)

        t = np.concatenate([dx, dy], axis=0)

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
            "Global angle %.1f; Shift: %s; RMSD: %.3f"
            % (np.degrees(a[0]), t[0], np.sqrt(np.mean(r**2)))
        )

        da = a[:-2] - 2 * a[1:-1] + a[2:]
        db = b[:-2] - 2 * b[1:-1] + b[2:]
        r = np.concatenate([r, C])  # , np.degrees(b), np.degrees(da), np.degrees(db)])
        return r

    def jacobian(dy, dx, a, b, c, W, M, restrain):
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
                # H3 = Rj @ RjTRj_inv @ RjT
                # dr_dp_i = -(H1 + H2 - (H2 @ H3 + H3 @ H1)) @ W0
                dr_dp_i = -(H1 + H2) @ W0

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

        def d_dt(dy, dx, a, b, c, W, M):
            # Get num frames and num points
            num_frames = a.shape[0]
            num_points = W.shape[1]

            # Get the rotation matrices
            Ra = Rotation.from_euler("z", a).as_matrix()
            Rb = Rotation.from_euler("x", b).as_matrix()
            Rc = Rotation.from_euler("y", c).as_matrix()
            Rabc = Ra @ Rb @ Rc

            # Construct the rotation matrices
            R = np.concatenate([Rabc[:, 0, :], Rabc[:, 1, :]], axis=0)

            # Initialise the derivatives of the centroid w.r.t the parameters
            dC_dy = np.zeros((3, num_frames))
            dC_dx = np.zeros((3, num_frames))

            # For each point add the elements of the Jacobian
            Jy = []
            Jx = []
            for j in range(num_points):
                # Get the mask, observations, rotation matrices
                Mj = M[:, j]
                Nj = np.count_nonzero(Mj)
                W0 = W[Mj, j]
                Rj = R[Mj, :]
                Qj = np.linalg.inv(Rj.T @ Rj) @ Rj.T

                # Construct the derivatives w.r.t dx and dy
                dt_dy_i = np.zeros((Nj, Nj))
                dt_dx_i = np.zeros((Nj, Nj))
                dt_dy_i[Nj // 2 :, Nj // 2 :] = np.identity(Nj // 2)
                dt_dx_i[: Nj // 2, : Nj // 2] = np.identity(Nj // 2)

                # Multiply by derivative matrices
                Uj = Qj @ dt_dy_i
                Vj = Qj @ dt_dx_i

                # Compute derivatives of the residuals w.r.t dy and dx
                dr_dy_i = Rj @ Uj - dt_dy_i
                dr_dx_i = Rj @ Vj - dt_dx_i

                # We need to put these subset of the results into an array with
                # zeros for the other frames
                dr_dx = np.zeros((dr_dx_i.shape[1], num_frames))
                dr_dy = np.zeros((dr_dy_i.shape[1], num_frames))
                dr_dx[:, Mj[:num_frames]] = dr_dx_i[: Nj // 2].T
                dr_dy[:, Mj[:num_frames]] = dr_dy_i[Nj // 2 :].T

                # Compute the derivatives of S w.r.t to dy and dx
                dS_dy = np.zeros((3, num_frames))
                dS_dx = np.zeros((3, num_frames))
                dS_dy[:, Mj[:num_frames]] = -Uj[:, Nj // 2 :]
                dS_dx[:, Mj[:num_frames]] = -Vj[:, : Nj // 2]

                # Add the derivatives to the centroid derivative
                dC_dy += dS_dy
                dC_dx += dS_dx

                # Add the derivatives of the residuals w.r.t dy and dx
                Jy.extend(dr_dy)
                Jx.extend(dr_dx)

            # Add the derivatives of the residuals w.r.t the centroid
            Jy.extend(dC_dy)
            Jx.extend(dC_dx)

            # Return as a numpy array
            return np.concatenate([Jy, Jx], axis=1)

        def d_da(dy, dx, a, b, c, W, M):
            # Get num frames and num points
            num_frames = a.shape[0]
            num_points = W.shape[1]

            # Get the rotation matrices
            Ra = Rotation.from_euler("z", a).as_matrix()
            Rb = Rotation.from_euler("x", b).as_matrix()
            Rc = Rotation.from_euler("y", c).as_matrix()
            Rabc = Ra @ Rb @ Rc

            # Compute the derivative of Ra w.r.t a
            dRa_da = np.full((num_frames, 3, 3), np.eye(3))
            dRa_da[:, 0, 0] = -np.sin(a)
            dRa_da[:, 0, 1] = -np.cos(a)
            dRa_da[:, 1, 0] = np.cos(a)
            dRa_da[:, 1, 1] = -np.sin(a)
            dRabc_da = dRa_da @ Rb @ Rc

            # Compute derivatices of residuals w.r.t a
            return d_dp(Rabc, dRabc_da, W, M)

        def d_db(dy, dx, a, b, c, W, M):
            # Get num frames and num points
            num_frames = a.shape[0]
            num_points = W.shape[1]

            # Get the rotation matrices
            Ra = Rotation.from_euler("z", a).as_matrix()
            Rb = Rotation.from_euler("x", b).as_matrix()
            Rc = Rotation.from_euler("y", c).as_matrix()
            Rabc = Ra @ Rb @ Rc

            # Compute the derivative of Ra w.r.t a
            dRb_db = np.full((num_frames, 3, 3), np.eye(3))
            dRb_db[:, 1, 1] = -np.sin(b)
            dRb_db[:, 1, 2] = -np.cos(b)
            dRb_db[:, 2, 1] = np.cos(b)
            dRb_db[:, 2, 2] = -np.sin(b)
            dRabc_db = Ra @ dRb_db @ Rc

            # Compute derivatices of residuals w.r.t b
            return d_dp(Rabc, dRabc_db, W, M)

        def d_dc(dy, dx, a, b, c, W, M):
            # Get num frames and num points
            num_frames = a.shape[0]
            num_points = W.shape[1]

            # Get the rotation matrices
            Ra = Rotation.from_euler("z", a).as_matrix()
            Rb = Rotation.from_euler("x", b).as_matrix()
            Rc = Rotation.from_euler("y", c).as_matrix()
            Rabc = Ra @ Rb @ Rc

            # Compute the derivative of Ra w.r.t a
            dRc_dc = np.full((num_frames, 3, 3), np.eye(3))
            dRc_dc[:, 0, 0] = -np.sin(c)
            dRc_dc[:, 2, 0] = -np.cos(c)
            dRc_dc[:, 0, 2] = np.cos(c)
            dRc_dc[:, 2, 2] = -np.sin(c)
            dRabc_dc = Ra @ Rb @ dRc_dc

            # Compute derivatices of residuals w.r.t c
            return d_dp(Rabc, dRabc_dc, W, M)

        # Check which parameters are to be restrained
        assert restrain in [None, "bc", "c"]
        derivatives = {
            None: [d_dt, d_da, d_db, d_dc],  # Derivatives w.r.t y, x, a, b and c
            "c": [d_dt, d_da, d_db],  # Derivatives w.r.t y, x, a and b
            "bc": [d_dt, d_da],  # Derivatives w.r.t y, x and a
        }

        W = W - t[:, None]

        # Compute the Jacobian elements
        r = np.concatenate(
            [d_dp(dy, dx, a, b, c, W, M) for d_dp in derivatives[restrain]], axis=1
        )
        return r

    def predict(a, b, c, t, W, M):
        # Get number of points
        num_points = W.shape[1]

        # Get the rotation matrices
        Rabc = Rotation.from_euler("yxz", np.stack([c, b, a], axis=1)).as_matrix()
        R = np.concatenate([Rabc[:, 0, :], Rabc[:, 1, :]], axis=0)

        # Compute the 3D spot positions
        S = np.zeros((3, num_points))
        for j in range(num_points):
            Mj = M[:, j]
            W0 = W[Mj, j]
            Rj = R[Mj, :]
            S[:, j] = np.linalg.inv(Rj.T @ Rj) @ Rj.T @ W0

        # Compute the predicted positions
        return R @ S + t[:, None]

    def get_params_and_args(dy, dx, a, b, c, W, M, restrain=None):
        assert restrain in [None, "bc", "c"]
        params, args = {
            None: ([dy, dx, a, b, c], [W, M, restrain]),
            "c": ([dy, dx, a, b], [c, W, M, restrain]),
            "bc": ([dy, dx, a], [b, c, W, M, restrain]),
        }[restrain]
        return np.array(params).flatten(), tuple(args)

    def parse_params_and_args(x, *args):
        restrain = args[-1]
        assert restrain in [None, "bc", "c"]
        if restrain is None:
            (dy, dx, a, b, c), (W, M, _) = x.reshape(5, -1), args
        elif restrain == "c":
            (dy, dx, a, b), (c, W, M, _) = x.reshape(4, -1), args
        elif restrain == "bc":
            (dy, dx, a), (b, c, W, M, _) = x.reshape(3, -1), args
        return dy, dx, a, b, c, W, M, restrain

    def parse_results(x, dy, dx, a, b, c, restrain=None):
        assert restrain in [None, "bc", "c"]
        if restrain is None:
            dy, dx, a, b, c = x.reshape(5, -1)
        elif restrain == "c":
            dy, dx, a, b = x.reshape(4, -1)
        elif restrain == "bc":
            dy, dx, a = x.reshape(3, -1)
        return dy, dx, a, b, c

    def fun(x, *args):
        return residuals(*parse_params_and_args(x, *args))

    def jac(x, *args):
        return jacobian(*parse_params_and_args(x, *args))

    # Construct the input
    X = data[:, :, 0]
    Y = data[:, :, 1]
    t = np.concatenate([dx, dy], axis=0)
    M = np.concatenate([mask, mask], axis=0)
    Wc = np.concatenate([X, Y], axis=0)

    # Iterate a number of times until convergence
    for it in range(1):  # max_iter):
        # Get the centroid subtracted observations
        # W = Wc - t[:, None]
        # tcurr[:] = t[:]

        # Get the params and arguments
        params, args = get_params_and_args(dy, dx, a, b, c, Wc, M, restrain)

        # Perform the least squares minimisation
        result = scipy.optimize.least_squares(
            fun,
            params,
            args=args,
            jac=jac,
            loss="linear",
            # bounds=[np.radians(-180), np.radians(180)],
        )

        # Get the results
        dy, dx, a, b, c = parse_results(result.x, dy, dx, a, b, c, restrain)

        # t = tcurr
        t = np.concatenate([dx, dy], axis=0)

        # Predict the positions on each image and fill the missing data entries
        # W1 = predict(a, b, c, t, W, M)
        # Wc[~M] = W1[~M]

        # Compute the new centroid. If the difference between the new and old
        # translations is less than a given threshold then break
        # t, told = np.mean(Wc, axis=1), t
        # tdiff = np.sqrt(np.mean((t - told) ** 2))
        # print("Centroid shifted by %.3f pixels" % tdiff)

        # Break if translation is smaller than a threshold
        # if tdiff < teps:
        #    break

    # Get the translations
    dx, dy = t.reshape(2, -1)

    # Compute the RMSD
    rmsd = np.sqrt(result.cost / np.count_nonzero(mask))

    print("RMSD: %f" % rmsd)

    # Return the refined parameters and RMSD
    return a, b, c, dy, dx, rmsd


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
    data, mask = read_points(contours)

    # Read the initial model and convert degrees to radians for use here
    P = np.array(model["transform"], dtype=float)
    a = np.radians(P[:, 0])
    b = np.radians(P[:, 1])
    c = np.radians(P[:, 2])
    dy = P[:, 3]
    dx = P[:, 4]

    # The image size
    image_size = model["image_size"]

    # The number of images and points
    assert data.shape[0] == P.shape[0]
    num_images = data.shape[0]
    num_points = data.shape[1]

    # Perform some checks on the contours
    check_obs_per_image(mask)
    check_obs_per_point(mask)
    # check_connections(contours)
    print("Num images: %d" % num_images)
    print("Num contours: %d" % num_points)
    print("Num observations: %d" % np.count_nonzero(mask))

    select = np.count_nonzero(mask, axis=0) >= 3
    data = data[:, select]
    mask = mask[:, select]

    dy += 256
    dx += 256

    # Run through the cycles of refinement
    for restrain in get_cycles(fix):
        a, b, c, dy, dx, rmsd = refine_model(
            a, b, c, dy, dx, data, mask, restrain=restrain
        )

    dy -= 256
    dx -= 256

    # Update the model and convert back to degrees
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
