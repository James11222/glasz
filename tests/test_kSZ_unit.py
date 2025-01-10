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
    z_lens,
)

param_dict = all_param_defaults


def test_kSZ_frequencies():
    theta = np.geomspace(0.5, 6.5, 50)  # arcmins

    prof_nfw = ccl.halos.HaloProfileNFW(
        mass_def=hmd, concentration=cM_relation, truncated=False, fourier_analytic=True
    )

    prof_gas = glasz.profiles.HaloProfileGNFW(
        hmd,
        rho0=1.0,
        alpha=param_dict["alpha"],
        beta=param_dict["beta"],
        gamma=param_dict["gamma"],
        x_c=param_dict["x_c"],
    )

    Rb = 10 * hmd.get_radius(cosmo, 10 ** param_dict["log10_M"], a_sf)
    prof_gas.normalize(cosmo, Rb, 10 ** param_dict["log10_M"], a_sf, prof_nfw)

    rho_3D = lambda r: prof_gas.real(cosmo, r, 10 ** param_dict["log10_M"], a_sf)

    T_kSZ_f150 = glasz.kSZ.create_T_kSZ_profile(
        theta, z_lens, rho_3D, frequency="f150", cosmo=cosmo
    )

    T_kSZ_f090 = glasz.kSZ.create_T_kSZ_profile(
        theta, z_lens, rho_3D, frequency="f090", cosmo=cosmo
    )

    assert T_kSZ_f150 is not None
    assert T_kSZ_f090 is not None

    assert np.all(T_kSZ_f150 >= 0)
    assert np.all(T_kSZ_f090 >= 0)

    assert np.all(np.isfinite(T_kSZ_f150))
    assert np.all(np.isfinite(T_kSZ_f090))

    assert np.all(T_kSZ_f150 != T_kSZ_f090)


def test_kSZ_convolution():
    theta = np.geomspace(0.5, 6.5, 50)  # arcmins

    prof_nfw = ccl.halos.HaloProfileNFW(
        mass_def=hmd, concentration=cM_relation, truncated=False, fourier_analytic=True
    )

    prof_gas = glasz.profiles.HaloProfileGNFW(
        hmd,
        rho0=1.0,
        alpha=param_dict["alpha"],
        beta=param_dict["beta"],
        gamma=param_dict["gamma"],
        x_c=param_dict["x_c"],
    )

    Rb = 10 * hmd.get_radius(cosmo, 10 ** param_dict["log10_M"], a_sf)
    prof_gas.normalize(cosmo, Rb, 10 ** param_dict["log10_M"], a_sf, prof_nfw)

    rho_3D = lambda r: prof_gas.real(cosmo, r, 10 ** param_dict["log10_M"], a_sf)

    T_kSZ_f150_hankel = glasz.kSZ.create_T_kSZ_profile(
        theta, z_lens, rho_3D, frequency="f150", cosmo=cosmo, method="hankel"
    )

    T_kSZ_f150_bf = glasz.kSZ.create_T_kSZ_profile(
        theta, z_lens, rho_3D, frequency="f150", cosmo=cosmo, method="brute_force"
    )

    assert T_kSZ_f150_hankel is not None
    assert T_kSZ_f150_bf is not None

    assert np.all(T_kSZ_f150_hankel >= 0)
    assert np.all(T_kSZ_f150_bf >= 0)

    assert np.all(np.isfinite(T_kSZ_f150_hankel))
    assert np.all(np.isfinite(T_kSZ_f150_bf))

    assert np.all(np.abs((T_kSZ_f150_hankel / T_kSZ_f150_bf) - 1) < 0.15)

    T_kSZ_f090_hankel = glasz.kSZ.create_T_kSZ_profile(
        theta, z_lens, rho_3D, frequency="f090", cosmo=cosmo, method="hankel"
    )

    T_kSZ_f090_bf = glasz.kSZ.create_T_kSZ_profile(
        theta, z_lens, rho_3D, frequency="f090", cosmo=cosmo, method="brute_force"
    )

    assert T_kSZ_f090_hankel is not None
    assert T_kSZ_f090_bf is not None

    assert np.all(T_kSZ_f090_hankel >= 0)
    assert np.all(T_kSZ_f090_bf >= 0)

    assert np.all(np.isfinite(T_kSZ_f090_hankel))
    assert np.all(np.isfinite(T_kSZ_f090_bf))

    assert np.all(np.abs((T_kSZ_f090_hankel / T_kSZ_f090_bf) - 1) < 0.15)
