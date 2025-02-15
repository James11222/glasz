# preamble 
import numpy as np
import pyccl as ccl

# Cosmological parameters
Omega_b = 0.044
Omega_c = 0.25 - Omega_b
h = 0.7
sigma8 = 0.8344
n_s = 0.9624

cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s)

# CMASS PARAMETERS
z_lens = 0.55  # Mean z for CMASS
a_sf = 1 / (1 + z_lens)

fb = cosmo["Omega_b"] / cosmo["Omega_m"] # Baryon fraction
fc = cosmo["Omega_c"] / cosmo["Omega_m"] # CDM fraction

k_arr = np.geomspace(1e-4, 1e4, 128) # Wavenumber array
a_arr = np.linspace(0.1, 1, 16) # Scale factor array
r_arr = np.geomspace(1e-2, 1e2, 100) # Distance array

# bounds we choose for our mass integral
M_min = 1e10
M_max = 1e15
num_mass = 32
M_arr = np.geomspace(M_min, M_max, num_mass)

# We will use the Δ=200 mass definition
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
    log10M_min=np.log10(M_min),
    log10M_max=np.log10(M_max),
    nM=num_mass,
)

