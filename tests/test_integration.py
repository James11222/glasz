from __future__ import annotations

import numpy as np
import pyccl as ccl  # type: ignore[import-untyped]

import glasz

from .test_init_halo_model import (  # type: ignore[import-untyped]
    a_sf,
    all_param_defaults,
    bM,
    cM_relation,
    cosmo,
    fb,
    fc,
    hmd,
    xi_mm_2h,
    z_lens,
)
from .test_utils import compute_kSZ  # type: ignore[import-untyped]


def test_import():
    import glasz

    assert glasz.__version__ is not None


param_dict = all_param_defaults


def test_full_pipeline():
    x_GGL = np.geomspace(1e-1, 1e2, 50)  # Mpc/h
    x_kSZ = np.geomspace(0.5, 6.5, 20)  # arcmins

    Rb = 10 * hmd.get_radius(cosmo, 10 ** param_dict["log10_M"], a_sf)

    # COMPUTE GNFW AMPLITUDE
    prof_nfw = ccl.halos.HaloProfileNFW(
        mass_def=hmd, concentration=cM_relation, truncated=False, fourier_analytic=True
    )

    prof_baryons = glasz.profiles.HaloProfileGNFW(
        hmd,
        rho0=1.0,
        alpha=param_dict["alpha"],
        beta=param_dict["beta"],
        gamma=param_dict["gamma"],
        x_c=param_dict["x_c"],
    )

    prof_baryons.normalize(cosmo, Rb, 10 ** param_dict["log10_M"], a_sf, prof_nfw)

    # COMPUTE 3D DENSITY PROFILES
    def rho_2h(r):
        return (
            xi_mm_2h(r)
            * bM(cosmo, 10 ** param_dict["log10_M"], a_sf)
            * ccl.rho_x(cosmo, a_sf, "matter", is_comoving=True)
            * param_dict["A_2h"]
        )

    prof_baryons.rho_2h = rho_2h  # add 2-halo term to baryon profile

    prof_matter = glasz.profiles.MatterProfile(
        mass_def=hmd, concentration=cM_relation, rho_2h=rho_2h
    )

    # COMPUTE ds PROFILE
    ds_b = (
        fb
        * glasz.GGL.calc_ds(
            cosmo,
            x_GGL / cosmo["h"],  # convert from Mpc/h to Mpc
            10 ** param_dict["log10_M"],
            a_sf,
            prof_baryons,
        )
        / cosmo["h"]
    )  # convert from Msun/pc^2 to h Msun/pc^2

    ds_dm = (
        fc
        * glasz.GGL.calc_ds(
            cosmo,
            x_GGL / cosmo["h"],  # convert from Mpc/h to Mpc
            10 ** param_dict["log10_M"],
            a_sf,
            prof_matter,
        )
        / cosmo["h"]
    )  # convert from Msun/pc^2 to h Msun/pc^2

    # COMPUTE kSZ PROFILE
    def rho_gas_3D(r):
        return fb * prof_baryons.real(cosmo, r, 10 ** param_dict["log10_M"], a_sf)

    T_kSZ = compute_kSZ(x_kSZ, z_lens, rho_gas_3D, "f150 - f090", cosmo)

    assert ds_b is not None
    assert ds_dm is not None
    assert T_kSZ is not None
    assert np.all(ds_b >= 0)
    assert np.all(ds_dm >= 0)
    assert np.all(T_kSZ >= 0)
    assert np.all(np.isfinite(ds_b))
    assert np.all(np.isfinite(ds_dm))
    assert np.all(np.isfinite(T_kSZ))
    assert np.all(ds_b < ds_dm)
    assert np.all(x_GGL * ds_dm < 20)
    assert np.all(x_GGL * ds_dm > 2)
    assert np.all(T_kSZ > 1e-11)
    assert np.all(T_kSZ < 1e-5)
