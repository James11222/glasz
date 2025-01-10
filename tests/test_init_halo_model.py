from __future__ import annotations

import numpy as np
import pyccl as ccl  # type: ignore[import-untyped]

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
