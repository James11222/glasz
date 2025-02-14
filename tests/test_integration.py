from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pyccl as ccl  # type: ignore[import-untyped]
from numpy.typing import NDArray

import glasz

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


def compute_kSZ(
    theta: NDArray[np.float64],
    z_lens: float,
    rho: Callable[[NDArray[np.float64] | np.float64], NDArray[np.float64] | np.float64],
    frequency: str,
    cosmo: Any,
) -> NDArray[np.float64]:
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
        T_kSZ = glasz.kSZ.create_T_kSZ_profile(theta, z_lens, rho, frequency, cosmo)

    elif frequency == "f150 - f090":
        T_kSZ_150 = glasz.kSZ.create_T_kSZ_profile(
            theta[: len(theta) // 2], z_lens, rho, "f150", cosmo
        )
        T_kSZ_090 = glasz.kSZ.create_T_kSZ_profile(
            theta[len(theta) // 2 :], z_lens, rho, "f090", cosmo
        )
        T_kSZ = np.concatenate((T_kSZ_150, T_kSZ_090))

    return cast(NDArray[np.float64], T_kSZ)


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
