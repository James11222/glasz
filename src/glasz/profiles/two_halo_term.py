from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pyccl as ccl  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.interpolate import interp1d  # type: ignore[import-untyped]


def calc_xi_mm_2h(cosmo: ccl.cosmology.Cosmology, 
                  mass_def: ccl.halos.MassDef, 
                  concentration: ccl.halos.concentration, 
                  hmc: ccl.halos.HMCalculator, 
                  k_arr: NDArray[np.float64], 
                  a_arr: NDArray[np.float64], 
                  r_arr: NDArray[np.float64], 
                  a_sf: np.float64) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    This function computes the matter-matter 2-halo term of
    the 3D correlation function.

    Args:
        cosmo: (`ccl.cosmology.Cosmology`): cosmology object
        hmc: (`ccl.halos.HMCalculator`): the halo mass calculator object
        k_arr: (`NDArray[float]`): the wavenumber array in comoving units of [1/Mpc]
        a_arr: (`NDArray[float]`): the scale factor array
        r_arr: (`NDArray[float]`): the radial array in comoving units of [Mpc]
        a_sf: (`float`): the scale factor of interest

    Returns:
        xi_mm_2h: (`Callable`): the interpolated 2-halo term of the matter-matter correlation function
    """

    pM = ccl.halos.HaloProfileNFW(
        mass_def=mass_def,
        concentration=concentration,
        truncated=True,
        fourier_analytic=True,
    )

    # compute the galaxy-matter power spectrum
    pk_mm_2h = ccl.halos.halomod_Pk2D(
        cosmo,
        hmc,
        pM,
        prof2=pM,
        lk_arr=np.log(k_arr),
        a_arr=a_arr,
        get_1h=False,
        get_2h=True,
    )

    return cast(Callable[[NDArray[np.float64]], NDArray[np.float64]], interp1d(
        r_arr,
        ccl.correlation_3d(cosmo, r=r_arr, a=a_sf, p_of_k_a=pk_mm_2h),
        bounds_error=False,
        fill_value=0.0,
    ))
