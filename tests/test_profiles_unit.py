from __future__ import annotations

import numpy as np
import pyccl as ccl  # type: ignore[import-untyped]

import glasz

from .test_init_halo_model import (  # type: ignore[import-untyped]
    M_arr,
    a_sf,
    all_param_defaults,
    bM,
    cM_relation,
    cosmo,
    fb,
    fc,
    hmd,
    r_arr,
    xi_mm_2h,
)

param_dict = all_param_defaults


def test_GNFW_base_1h():
    prof_nfw = ccl.halos.HaloProfileNFW(
        mass_def=hmd, concentration=cM_relation, truncated=False, fourier_analytic=True
    )

    profile_gas = glasz.profiles.HaloProfileGNFW(
        hmd,
        rho0=1.0,
        alpha=param_dict["alpha"],
        beta=param_dict["beta"],
        gamma=param_dict["gamma"],
        x_c=param_dict["x_c"],
    )
    Rb = 10 * hmd.get_radius(cosmo, 10 ** param_dict["log10_M"], a_sf)
    assert profile_gas.rho0 == 1.0
    profile_gas.normalize(cosmo, Rb, 10 ** param_dict["log10_M"], a_sf, prof_nfw)
    assert profile_gas.rho0 != 1.0
    rho0 = profile_gas.rho0
    profile_gas.normalize(cosmo, Rb, 10 ** param_dict["log10_M"], a_sf, prof_nfw)
    assert profile_gas.rho0 == rho0

    assert isinstance(profile_gas, glasz.profiles.HaloProfileGNFW)
    assert profile_gas.real is not None
    assert profile_gas.fourier is not None


def test_GNFW_base_1h_truncated():
    profile_gas = glasz.profiles.HaloProfileGNFW(
        hmd,
        rho0=1.0,
        alpha=param_dict["alpha"],
        beta=param_dict["beta"],
        gamma=param_dict["gamma"],
        x_c=param_dict["x_c"],
        truncated=True,
    )

    assert isinstance(profile_gas, glasz.profiles.HaloProfileGNFW)
    R = (hmd.get_radius(cosmo, 10 ** param_dict["log10_M"], a_sf) / a_sf) * 2
    # make sure that the profile is properly truncated
    assert profile_gas.real(cosmo, R, 10 ** param_dict["log10_M"], a_sf) == 0
    assert profile_gas.real is not None


def test_GNFW_feedback_1h():
    profile_gas_AGN = glasz.profiles.HaloProfileGNFW(
        mass_def=hmd,
        feedback_model="AGN",
        truncated=False,
    )

    profile_gas_SH = glasz.profiles.HaloProfileGNFW(
        mass_def=hmd,
        feedback_model="SH",
        truncated=False,
    )

    assert isinstance(profile_gas_AGN, glasz.profiles.HaloProfileGNFW)
    assert isinstance(profile_gas_SH, glasz.profiles.HaloProfileGNFW)
    assert profile_gas_AGN.real is not None

    rho_arr_AGN = profile_gas_AGN.real(cosmo, r_arr, M_arr, a_sf)
    rho_arr_SH = profile_gas_SH.real(cosmo, r_arr, M_arr, a_sf)

    assert rho_arr_AGN.shape == (len(M_arr), len(r_arr))
    assert rho_arr_SH.shape == (len(M_arr), len(r_arr))

    # we expect less massive halos to have lower gas density amplitudes at small radii
    assert rho_arr_AGN[-1, :][0] > rho_arr_AGN[0, :][0]
    assert rho_arr_SH[-1, :][0] > rho_arr_SH[0, :][0]

    factor_high_mass = rho_arr_AGN[-1][-1] / rho_arr_SH[-1][-1]

    factor_low_mass = rho_arr_AGN[0][-1] / rho_arr_SH[0][-1]
    # We expect that at low halo masses, AGN feedback should much more gas out to large radii
    # than SH feedback alone, but at large halo masses the difference should be much less pronounced
    assert 1e5 > (factor_low_mass / factor_high_mass) > 1e4


def test_GNFW_include_2h():
    # COMPUTE 3D DENSITY PROFILES
    rho_2h = lambda r: (
        xi_mm_2h(r)
        * bM(cosmo, 10 ** param_dict["log10_M"], a_sf)
        * ccl.rho_x(cosmo, a_sf, "matter", is_comoving=True)
        * param_dict["A_2h"]
    )

    profile_gas_with_2h = glasz.profiles.HaloProfileGNFW(
        mass_def=hmd,
        feedback_model="AGN",
        rho_2h=rho_2h,
    )

    profile_gas_without_2h = glasz.profiles.HaloProfileGNFW(
        mass_def=hmd,
        feedback_model="AGN",
    )

    assert profile_gas_with_2h.real(
        cosmo, 1e1, 10 ** param_dict["log10_M"], a_sf
    ) > profile_gas_without_2h.real(cosmo, 1e1, 10 ** param_dict["log10_M"], a_sf)


def test_total_matter_profile():
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
    rho_2h = lambda r: (
        xi_mm_2h(r)
        * bM(cosmo, 10 ** param_dict["log10_M"], a_sf)
        * ccl.rho_x(cosmo, a_sf, "matter", is_comoving=True)
        * param_dict["A_2h"]
    )

    prof_baryons.rho_2h = rho_2h  # add 2-halo term to baryon profile

    profile_matter = glasz.profiles.MatterProfile(
        mass_def=hmd, concentration=cM_relation, rho_2h=rho_2h
    )

    assert isinstance(profile_matter, glasz.profiles.MatterProfile)
    assert profile_matter.real is not None
    rho_matter = profile_matter.real(cosmo, r_arr, 10 ** param_dict["log10_M"], a_sf)
    rho_dm = rho_matter * fc
    rho_baryons = (
        prof_baryons.real(cosmo, r_arr, 10 ** param_dict["log10_M"], a_sf) * fb
    )

    assert np.all(
        rho_matter > rho_baryons
    )  # there should be more matter than gas everywhere
    assert (
        profile_matter.real(cosmo, 1e1, 10 ** param_dict["log10_M"], a_sf) > 0
    )  # matter should be above zero

    assert np.all(
        rho_dm > rho_baryons
    )  # there should be more dark matter than gas everywhere
