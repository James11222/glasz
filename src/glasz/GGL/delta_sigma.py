from __future__ import annotations

from typing import cast

import numpy as np
import pyccl as ccl  # type: ignore[import-untyped]
from numpy.typing import NDArray

from .. import constants as const


def calc_ds(
    cosmo: ccl.cosmology.Cosmology,
    R: float | NDArray[np.float64],
    M: float | NDArray[np.float64],
    a: float | NDArray[np.float64],
    prof: ccl.halos.profile_base.HaloProfile,
) -> NDArray[np.float64]:
    """
    a function to compute the comoving excess surface mass density profile given a halo profile.

    Args:
        cosmo: (`pyccl.cosmology.Cosmology`): a Cosmology object.
        R: (`float` or `NDArray[float]`): projected comoving radius in Mpc.
        M: (`float` or `NDArray[float]`): halo mass in units of M_sun.
        a: (`float` or `NDArray[float]`): scale factor.
        prof: (`pyccl.halos.profiles.profile_base.HaloProfile`): halo profile.

    Returns:
        ds: (`float` or `NDArray[float]`): comoving excess surface mass density ($\Delta \Sigma$)
    """

    prof.update_precision_fftlog(
        padding_hi_fftlog=1e5,
        padding_lo_fftlog=1e-5,
    )

    sigma_lR = prof.cumul2d(cosmo, R, M, a)
    sigma_R = prof.projected(cosmo, R, M, a)

    sigma_R = sigma_R * const.Msun_Mpc2_to_Msun_pc2
    sigma_lR = sigma_lR * const.Msun_Mpc2_to_Msun_pc2

    return cast(NDArray[np.float64], sigma_lR - sigma_R)
