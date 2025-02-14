from __future__ import annotations

import numpy as np
import pyccl as ccl  # type: ignore[import-untyped]

import glasz

# import sys, os
# sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))


# Cosmological parameters
Omega_b = 0.044
Omega_c = 0.25 - Omega_b
h = 0.7
sigma8 = 0.8344
n_s = 0.9624

cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s)

# CMASS PARAMETERS
z_lens = 0.55  # Median z for CMASS
a_sf = 1 / (1 + z_lens)

rho_m = ccl.rho_x(cosmo, a_sf, "matter", is_comoving=True)
fb = cosmo["Omega_b"] / cosmo["Omega_m"]
fc = cosmo["Omega_c"] / cosmo["Omega_m"]

k_arr = np.geomspace(1e-4, 1e4, 128)
a_arr = np.linspace(0.1, 1, 16)
r_arr = np.geomspace(1e-2, 1e2, 100)

# bounds we choose for our mass integral
mmin = 1e10
mmax = 1e15
num_mass = 32
M_arr = np.geomspace(mmin, mmax, num_mass)

# We will use the virial mass definition
hmd = ccl.halos.MassDef200m

# The Tinker 2008 mass function
nM = ccl.halos.MassFuncTinker08(mass_def=hmd)

# The Duffy 2008 concentration-mass relation
cM_relation = ccl.halos.concentration.ConcentrationDuffy08(mass_def=hmd)

# The Tinker 2010 halo bias
bM = ccl.halos.HaloBiasTinker10(mass_def=hmd)

# The HMF and bias are combined in a `HMCalculator` object, along with mass definition
hmc = ccl.halos.HMCalculator(
    mass_function=nM,
    halo_bias=bM,
    mass_def=hmd,
    log10M_min=np.log10(mmin),
    log10M_max=np.log10(mmax),
    nM=num_mass,
)

xi_mm_2h = glasz.profiles.calc_xi_mm_2h(
    cosmo, hmd, cM_relation, hmc, k_arr, a_arr, r_arr, a_sf
)

# - - - - - - - - - - - - - - - - - -
#         DEFAULT PARAMETERS
# - - - - - - - - - - - - - - - - - -
log10_M_default = np.log10(3e13)
c_default = cM_relation(cosmo, M=10**log10_M_default, a=a_sf)

x_c_default = 0.5
alpha_default = 0.88 * ((10**log10_M_default) / 1e14) ** (-0.03) * (1 / a_sf) ** 0.19
beta_default = 3.83 * ((10**log10_M_default) / 1e14) ** 0.04 * (1 / a_sf) ** (-0.025)
gamma_default = 0.2
A_2h_default = 1.0

defaults = [
    ### DM parameters ###
    log10_M_default,
    c_default,
    ### GAS parameters ###
    x_c_default,
    alpha_default,
    beta_default,
    gamma_default,
    A_2h_default,
]

param_names = [
    ### DM parameters ###
    "log10_M",
    "c",
    ### GAS parameters ###
    "x_c",
    "alpha",
    "beta",
    "gamma",
    "A_2h",
]

priors = [
    ### DM parameters ###
    (12.0, 14.0),
    (2.0, 10.0),
    ### GAS parameters ###
    (0.1, 1.0),
    (0.5, 1.5),
    (1.0, 5.0),
    (0.1, 1.5),
    (0.0, 5.0),
]

all_param_defaults = dict(zip(param_names, defaults))
all_param_priors = dict(zip(param_names, priors))

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
