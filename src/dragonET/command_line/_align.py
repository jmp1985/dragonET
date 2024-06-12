#
# align.py
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
import scipy.optimize
import torch
import yaml

__all__ = ["align"]


def get_description():
    """
    Get the program description

    """
    return "Do a rough alignment of the projection images"


def get_parser(parser: ArgumentParser = None) -> ArgumentParser:
    """
    Get the align parser

    """

    # Initialise the parser
    if parser is None:
        parser = ArgumentParser(description=get_description())

    # Add some command line arguments
    parser.add_argument(
        "-p",
        type=str,
        default=None,
        dest="projections_in",
        required=True,
        help=(
            """
            The filename for the projection images
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
            A file describing the initial model.
            """
        ),
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="aligned_model.yaml",
        dest="model_out",
        help=(
            """
            A YAML file describing the refined model.
            """
        ),
    )
    parser.add_argument(
        "--reference_image",
        type=int,
        default=None,
        dest="reference_image",
        help="Set the reference image, if not set the angle closest to zero will be chosen",
    )
    parser.add_argument(
        "--max_shift",
        type=float,
        default=0.25,
        dest="max_shift",
        help="Maximum normalised image shift (between 0 and 1)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=10,
        dest="max_iter",
        help="Maximum number of iterations (> 0)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=10,
        dest="max_images",
        help="Maximum number of images to use in multiple correlation (> 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        dest="device",
        help="The device settings to use",
    )

    return parser


def align_impl(args):
    """
    Do a rough alignment of the projection images

    """

    # Get the start time
    start_time = time.time()

    # Do the work
    _align(
        args.projections_in,
        args.model_in,
        args.model_out,
        args.reference_image,
        args.max_shift,
        args.max_iter,
        args.max_images,
        args.device,
    )

    # Write some timing stats
    print("Time taken: %.2f seconds" % (time.time() - start_time))


def align(args: List[str] = None):
    """
    Do a rough alignment of the projection images

    """
    align_impl(get_parser().parse_args(args=args))


def align_single(X: torch.Tensor, Y: torch.Tensor) -> tuple:
    """

    Compute the cross coefficient of multiple correlation

    This function assumes that the data are already fourier transformed and
    normalised to zero mean and standard deviation of 1.

    Args:
        X: The stack of reference images
        Y: The target image

    Returns:
        Tuple containing the cross coefficient of multiple correlation and the
        shift (y, x)

    """

    # Get the device
    device = X.device

    # Compute normalisation
    norm = torch.numel(Y)

    # Compute correlation coefficients between every reference image
    # This is the slowest part of the calculation. Perhaps we can do it on
    # smaller rebinned data since this just provides correlation weights
    Rxx = torch.zeros((X.shape[0], X.shape[0]), device=device)
    for j in range(Rxx.shape[0]):
        for i in range(j, Rxx.shape[1]):
            Rji = torch.sum(X[j] * X[i].conj()).real
            Rxx[j, i] = Rji
            Rxx[i, j] = Rji
    Rxx /= norm**2

    # Invert the Rxx matrix
    Rxx_inv = torch.linalg.inv(Rxx)

    # Compute the cross correlation between the target image and the others
    c = torch.zeros(X.shape, device=device)
    for j in range(c.shape[0]):
        c[j] = torch.fft.ifftshift(torch.fft.ifft2(X[j] * Y.conj())).real / norm

    # For each translation (k,l) compute R2 = sqrt(c^T Rxx^-1 c)
    # Rxx_inv_c = torch.einsum("ij,jkl->kli", Rxx_inv, c)
    # R2 = torch.einsum("ikl,kli->kl", c, Rxx_inv_c)
    R = torch.sqrt(torch.einsum("ikl,ij,jkl->kl", c, Rxx_inv, c))

    # Find the maximum
    shift = np.unravel_index(torch.argmax(R).cpu(), R.shape)
    shift -= np.array(R.shape) // 2

    # Return the shift
    return R, shift


def select_reference_images(
    data: np.ndarray, ref_index: int, max_images: int = 10
) -> np.ndarray:
    """
    Select the N closest reference images to fit to

    Args:
        data: The filtered image data
        ref_index: The index of the reference image
        max_images: The maximum number of images to use

    Returns:
        The stack of selected images

    """
    if data.shape[0] > max_images:
        select = np.array(
            sorted(np.arange(data.shape[0]), key=lambda x: np.abs(x - ref_index))
        )[0:max_images]
        data = data[select, :, :]
    return data


def fourier_shift_image(data: torch.Tensor, shift) -> torch.Tensor:
    """
    Transform an image

    """
    phase_shift = torch.from_numpy(
        scipy.ndimage.fourier_shift(np.ones(data.shape), shift)
    ).to(data.device)
    return data * phase_shift


def align_stack(
    data: np.ndarray,
    angles: np.ndarray,
    shifts: np.ndarray,
    max_shift: float = 0.25,
    max_iter: int = 10,
    max_images: int = 10,
    device: str = "gpu",
):
    """
    Align the stack

    """

    def normalise(x):
        return (x - x.mean()) / x.std()

    def find_nearest(x, v):
        return x[np.abs(x - v).argmin()]

    def initialise(data, shifts, device):
        # Init FFTs, filter, do initial shift, and do the whole thing in Fourier space
        fft_data = torch.zeros(data.shape, dtype=torch.complex64)
        for j in range(fft_data.shape[0]):
            print(" Loading image %d/%d" % (j + 1, fft_data.shape[0]))
            fft_data[j] = fourier_shift_image(
                torch.fft.fft2(
                    normalise(torch.from_numpy(data[j].astype("float32")).to(device))
                ),
                (shifts[j, 0], shifts[j, 1]),
            ).to(fft_data.device)

        # Return the fft data
        return fft_data

    def apply_weights(X):
        # Apply weights to the data. Some things to note and possibly improve: 1.
        # By multiplying the image and reference by a window function we are
        # assuming the weights are uncorrelated.  2. Also, by multiplying the image
        # by the weights the weights are the same for each translation so this
        # needs to be done iteratively.  3. This does not give the correlation
        # between the image and the weigted least squares estimate but rather the
        # weighted correlation between image and weighted least squares estimate
        ysize, xsize = X.shape[-2:]
        W = np.hanning(ysize)[:, None] * np.hanning(xsize)[None, :]
        W = torch.sqrt(torch.from_numpy(W).to(X.device))
        if X.dim() == 2:
            X = torch.fft.fft2(normalise(W * torch.fft.ifft2(X)))
        else:
            assert X.dim() == 3
            for j in range(X.shape[0]):
                X[j] = torch.fft.fft2(normalise(W * torch.fft.ifft2(X[j])))
        return X

    # Check input
    assert data.shape[0] == angles.size
    assert data.shape[0] == shifts.shape[0]
    assert len(shifts.shape) == 2 and (shifts.shape[1] == 2)
    assert (max_shift > 0) and (max_shift <= 1)
    assert max_iter > 0
    assert max_images > 0

    # Get the device
    device = torch.device(
        "cuda" if (device == "gpu" and torch.cuda.is_available()) else "cpu"
    )

    # Print some details
    algorithm = "multiple correlation"
    print("Running %s alignment using %s" % (algorithm, str(device)))

    # Save the original parameters
    shifts_orig = shifts.copy()

    # Initialise the data
    fft_data = initialise(data, shifts, device)

    # Generate the index ordering
    order = list(sorted(range(len(angles)), key=lambda x: abs(angles[x])))

    # Don't align the zero tilt image
    zero = order[0]
    order = order[1:]

    # Create the coefficients tensor
    num_images, ysize, xsize = data.shape

    # Get the max shift in pixels
    max_shift_px = max_shift * np.sqrt(ysize**2 + xsize**2)

    # Create an array to check if the image has been aligned at least once
    # This is reset at each iteration because we do iterative reweighting
    is_aligned = np.zeros(angles.size, dtype=bool)
    is_aligned[zero] = True

    # Select each image sequentially
    for i, tar_index in enumerate(order):
        # Get the reference index
        ref_index = find_nearest(np.where(is_aligned)[0], tar_index)
        ref_index_is_aligned = np.where(np.where(is_aligned)[0] == ref_index)[0][0]

        # Select the reference images and apply real space weights
        fft_data_stack = apply_weights(
            select_reference_images(
                fft_data[is_aligned],
                ref_index_is_aligned,
                max_images,
            ).to(device)
        )

        # Do the iterations for each image
        for it in range(max_iter):
            # Apply the real space weights to the target data
            # Align the image with the stack
            I, (ry, rx) = align_single(
                fft_data_stack, apply_weights(fft_data[tar_index].to(device))
            )

            # Enforce a maximum shift
            r = np.sqrt(rx**2 + ry**2)
            if r > max_shift_px:
                rx = rx * max_shift_px / r
                ry = ry * max_shift_px / r
            shifts[tar_index] += np.array((ry, rx))
            print(
                " Aligning image %d (%.1f deg): shift y = %.1f; shift x = %.1f"
                % (
                    tar_index + 1,
                    angles[tar_index],
                    shifts[tar_index, 0],
                    shifts[tar_index, 1],
                )
            )

            # Transform the image for the next iteration
            fft_data[tar_index] = fourier_shift_image(
                fft_data[tar_index].to(device), (ry, rx)
            ).to(fft_data.device)

            # If we move less than the pixel size then break
            if r < 1.0:
                break

        # Delete to free up memory
        del fft_data_stack

        # Select the images to use in the model. Only aligned images will be used
        is_aligned[tar_index] = True

    # Compute rmsd
    err = np.sqrt(np.sum((shifts - shifts_orig) ** 2) / len(order))

    # Print some info
    print(" Finished alignment; RMS shift: %.4g" % err)

    # Return the parameters
    return shifts


def _align(
    projections_in: str,
    model_in: str,
    model_out: str,
    reference_image: int = None,
    max_shift: float = 0.25,
    max_iter: int = 10,
    max_images: int = 10,
    device: str = "gpu",
):
    """
    Do the alignment

    """

    def read_projections(filename):
        print("Reading projections from %s" % filename)
        return mrcfile.mmap(filename).data

    def read_model(filename) -> dict:
        print("Reading model from %s" % filename)
        return yaml.safe_load(open(filename, "r"))

    def write_model(model, filename):
        print("Writing model to %s" % filename)
        yaml.safe_dump(model, open(filename, "w"), default_flow_style=None)

    # Read the projections
    projections = read_projections(projections_in)

    # Read the model
    model = read_model(model_in)

    # Read the initial model and convert degrees to radians for use here
    P = np.array(model["transform"], dtype=float)

    # Check the pitch is zero
    if np.any(np.abs(P[:, 1] > 0)):
        print("- Warning pitch is not zero in initial model but will be ignored.")

    # Align the stack
    P[:, 3:5] = align_stack(
        projections,
        P[:, 2],
        P[:, 3:5],
        max_shift,
        max_iter,
        max_images,
        device,
    )

    # Update the model and convert back to degrees
    model["transform"] = P.tolist()

    # Write the model to file
    write_model(model, model_out)


if __name__ == "__main__":
    align()
