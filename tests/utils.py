from __future__ import annotations

import numpy as np

import GLaSZ


def compute_kSZ(theta, z_lens, rho, frequency, cosmo):
    """
    This function computes the kSZ temperature profile for a given set of parameters.

    Arguments:
    - theta (array): the angular separation array in units of [arcmin]
    - z_lens (float): the lens redshift
    - rho (array): the 3D density profile in units of [Msun/Mpc^3]
    - frequency (str): the frequency of the beam function
    - cosmo (ccl.Cosmology): the cosmology object

    Returns:
    - T_kSZ (array): the kSZ temperature profile in units of [μK * arcmin^2]
    """
    if frequency in ("f150", "f090"):
        T_kSZ = GLaSZ.kSZ.create_T_kSZ_profile(theta, z_lens, rho, frequency, cosmo)

    elif frequency == "f150 - f090":
        T_kSZ_150 = GLaSZ.kSZ.create_T_kSZ_profile(
            theta[: len(theta) // 2], z_lens, rho, "f150", cosmo
        )
        T_kSZ_090 = GLaSZ.kSZ.create_T_kSZ_profile(
            theta[len(theta) // 2 :], z_lens, rho, "f090", cosmo
        )
        T_kSZ = np.concatenate((T_kSZ_150, T_kSZ_090))

    else:
        raise AssertionError(
            "Not a valid frequency. Choose from 'f150', 'f090', 'f150 - f090'"
        )

    return T_kSZ
