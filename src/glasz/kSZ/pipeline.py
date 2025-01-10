from __future__ import annotations

from collections.abc import Callable
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d  # type: ignore[import-untyped]

from .. import constants as const
from .beam import generate_beam_profile
from .fht import RadialFourierTransform

Numeric = Union[np.int32, np.int64, np.float32, np.float64]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                         Functions                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def project_density_profile(
    rho_3D: Callable[[NDArray[Numeric] | Numeric], NDArray[Numeric] | Numeric],
    d_A: Numeric,
    los_distance: NDArray[Numeric],
    theta_smooth: NDArray[Numeric],
) -> Any:
    """
    A function which takes in a 3D density profile and projects it into a 2D density profile by
    integrating along the line of sight.

     int_{rm los} rho(sqrt{l^2 + d_A(z)^2 |theta|^2}) dl


    Arguments:
        rho_3D (float): 3D density profile [Msun/Mpc^3]
        d_A (float): angular diameter distance
        los_distance (float): radial coordinate [Mpc]
        theta_smooth (float): angular coordinate high resolution array extending to 30'

    Returns:
        rho_2D (float): 2D density profile [g/cm^2]
    """

    # perform the integral
    r_int = np.sqrt(los_distance**2 + theta_smooth[:, None] ** 2 * d_A**2)
    return 2 * np.trapz(
        rho_3D(r_int) * const.Msun_Mpc3_to_CGS,
        x=los_distance * const.Mpc_to_CGS,
        axis=1,
    )  # g/cm^2


def convolve_density_with_beam(
    rho_2D: Any,
    frequency: str,
    theta_smooth: NDArray[Numeric],
    theta_use: NDArray[Numeric],
    method: str,
) -> Any:
    """
    A function which takes in a 2D density profile and convolves it with a beam profile.

    Arguments:
        rho_2D (float):
            2D density profile (units of g/cm^2)
        frequency (str):
            frequency of the beam profile
        theta_smooth (float):
            angular coordinate (high resolution array extending to 30')
        theta_use (float):
            angular coordinate (lower resolution array extending to theta)
        method (str):
            method to use for convolution. Options are 'brute_force', 'hankel', 'fft'

    Returns:
        rho_2D_beam (float):
            2D density profile convolved with beam [units of g/cm^2]
        rho_2D_beam_annulus (float):
            2D density profile convolved with beam in annulus [units of g/cm^2]
    """

    if method == "brute_force":
        """
        This method performs a brute force convolution of the 2D density profile with the beam profile
        using the form of a cylindrical convolution. This method is computationally expensive and should
        not be used in MCMC.
        """
        f_beam = generate_beam_profile(frequency, space="real")
        # convolve with beam (integrate over a circle using azimuthal angle phi from 0 to 2π)
        phi = np.linspace(0.0, 2 * np.pi, 100)

        # reformat arrays for convolution - shape: ((theta_use, theta_smooth, phi))
        theta_use_ = theta_use[:, None, None]
        theta_smooth_ = theta_smooth[None, :, None]
        phi_ = phi[None, None, :]
        rho_2D_ = rho_2D[None, :, None]

        integrand = (
            theta_smooth_
            * rho_2D_
            * f_beam(
                np.sqrt(
                    theta_use_**2
                    + theta_smooth_**2
                    - 2 * theta_use_ * theta_smooth_ * np.cos(phi_)
                )
            )
        )

        rho_2D_beam0 = np.trapz(integrand, x=phi_, axis=2)
        rho_2D_beam = np.trapz(rho_2D_beam0, x=theta_smooth, axis=1)

    elif method == "hankel":
        """
        This method performs a convolution of the 2D density profile with the beam profile
        using the Fast-Hankel transform. This method is computationally efficient and should be used
        in MCMC.
        """

        f_beam = generate_beam_profile(frequency, space="harmonic")

        rht = RadialFourierTransform(n=200, pad=100, lrange=[170, 1.4e6])

        rho_2D_ = np.interp(rht.r, theta_smooth, rho_2D)
        rho_2D_ell_ = rht.real2harm(rho_2D_)
        rho_2D_beam_ = rht.harm2real(rho_2D_ell_ * f_beam(rht.ell))
        r_unpad, rho_2D_beam_ = rht.unpad(rht.r, rho_2D_beam_)

        rho_2D_beam = interp1d(
            r_unpad.flatten(),
            rho_2D_beam_.flatten(),
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )(theta_use)

    else:  # pragma: no cover
        msg = "method must be either 'brute_force' or 'hankel'"
        raise ValueError(msg)

    return rho_2D_beam


def compute_T_kSZ(
    rho_2D_beam: Any, rho_2D_beam_annulus: Any, theta_use: NDArray[Numeric]
) -> Any:
    """
    A function which takes in a beam convolved 2D density profile and computes the kSZ temperature profile.

    Arguments:
        rho_2D_beam (float): 2D density profile convolved with beam
        rho_2D_beam_annulus (float): 2D density profile convolved with beam in annulus
        theta_use (float): angular coordinate (lower resolution array extending to theta)

    Returns:
        T_kSZ (float): kSZ temperature profile [μK * arcmin^2]
    """

    dtheta = theta_use[0]
    dtheta_annulus = theta_use[0] * np.sqrt(
        2
    )  # we make use of small angle approximation here
    theta_use_annulus = theta_use * np.sqrt(
        2
    )  # we make use of small angle approximation here

    sig = 2.0 * np.pi * dtheta * np.sum(theta_use * rho_2D_beam)
    sig_annulus = (
        2.0 * np.pi * dtheta_annulus * np.sum(theta_use_annulus * rho_2D_beam_annulus)
    )

    return (
        (2 * sig - sig_annulus)
        * const.v_rms
        * const.ST_CGS
        * const.TCMB
        * 1e6
        * ((1.0 + const.XH) / 2.0)
        / const.MP_CGS
    )


@np.vectorize
def create_T_kSZ_profile(
    theta: Numeric,
    z: Numeric,
    rho_3D: Callable[[NDArray[Numeric] | Numeric], NDArray[Numeric] | Numeric],
    frequency: str,
    cosmo: Any,
    NNR: int = 100,
    resolution_factor: float = 3.0,
    method: str = "hankel",
) -> Any:
    """
    This function computes the projected density profile and projects
    it into an observable T_kSZ profile which we can compare to kSZ measurements.
    The function can be broken down into three steps:

    1. Compute the projected density profile rho(r⊥) by integrating the 3D density
    profile rho(r) along the line of sight.

    2. Convolve the projected 2D density profile with the beam profile.

    3. Compute the average T_kSZ within disks of varying radii theta_d and subtract off
    the mean T_kSZ of an adjacent annulus of external radius sqrt(2)*theta_d with equal area.

    In the end we are left with T_kSZ(theta_d) which is the observable T_kSZ profile.

    Arguments:
        theta (float): angular radius in arcmin
        z (float): redshift
        rho_3D (function): 3D density profile (assume radial only) [Msun/Mpc^3]
        frequency (str): frequency of the beam profile
        cosmo (ccl Cosmology object): cosmology object created by pyccl
        NNR (int): number of radial bins
        resolution_factor (float): resolution factor for angular coordinates
        method (str): method to use for convolution. Options are 'brute_force', 'hankel', 'fft'

    Returns:
        float: observable T_kSZ profile in μK
    """

    # Define the resolution of the angular coordinates
    NNR2 = resolution_factor * NNR

    # angular diameter distance (in Mpc)
    d_A = cosmo.angular_diameter_distance(1 / (1 + z))

    # radial coordinate
    los_distance = np.logspace(-3, 1, 100)  # Mpc

    # angular coordinate (r -> theta)
    r_use = d_A * np.arctan(np.radians(theta / 60.0))
    dtheta_use = np.arctan(r_use / d_A) / NNR  # rads
    theta_use = (np.arange(NNR) + 1.0) * dtheta_use

    theta_max = np.radians(30.0 / 60.0)  # 30' in radians
    r_lim = d_A * np.arctan(theta_max)
    dtheta_smooth = np.arctan(r_lim / d_A) / NNR2  # rads
    theta_smooth = (np.arange(NNR2) + 1.0) * (dtheta_smooth)

    # - - - - - - -
    #    Step 1.
    # - - - - - - -

    # Compute the projected density profile rho(r⊥)
    # by integrating the 3D density profile rho(r) along the line of sight.
    rho_2D = project_density_profile(rho_3D, d_A, los_distance, theta_smooth)

    # - - - - - - -
    #    Step 2.
    # - - - - - - -
    rho_2D_beam = convolve_density_with_beam(
        rho_2D, frequency, theta_smooth, theta_use, method
    )
    rho_2D_beam_annulus = convolve_density_with_beam(
        rho_2D, frequency, theta_smooth, theta_use * np.sqrt(2), method
    )

    # - - - - - - -
    #    Step 3.
    # - - - - - - -

    return compute_T_kSZ(rho_2D_beam, rho_2D_beam_annulus, theta_use)
