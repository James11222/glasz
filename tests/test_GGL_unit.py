from __future__ import annotations

import numpy as np
import pyccl as ccl  # type: ignore[import-untyped]

import glasz

from .test_init_halo_model import (  # type: ignore[import-untyped]
    a_sf,
    all_param_defaults,
    cM_relation,
    cosmo,
    hmd,
    r_arr,
)

param_dict = all_param_defaults


def test_GGL():
    prof_NFW = ccl.halos.profiles.HaloProfileNFW(
        mass_def=hmd,
        concentration=cM_relation,
        truncated=False,
        cumul2d_analytic=True,
        projected_analytic=True,
        fourier_analytic=True,
    )
    ds = glasz.GGL.calc_ds(cosmo, r_arr, 10 ** param_dict["log10_M"], a_sf, prof_NFW)

    assert ds is not None
    assert np.all(r_arr * ds >= 0)
    assert np.all(np.isfinite(ds))
    assert np.all(np.isreal(ds))
    assert np.all(r_arr * ds < 10)
