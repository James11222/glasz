from __future__ import annotations

import numpy as np
import pyccl as ccl # type: ignore[import-untyped]
import scipy.integrate # type: ignore[import-untyped]

from numpy.typing import NDArray
from collections.abc import Callable
from typing import Any, cast

class HaloProfileGNFW(ccl.halos.HaloProfileMatter): # type: ignore[misc]
    """
    Generalized NFW Density Profile. This class implements the 
    generalized NFW profile as described in Zhao et al. 1996. 
    (https://arxiv.org/pdf/astro-ph/9509122). The profile is
    defined as:

    $$\rho_{\rm GNFW}(r) = \rho_0 \left(\frac{x}{x_c}\right)^{-\gamma} \left( 1 + \left( \frac{x}{x_c} \right)^{1/\alpha} \right)^{-(\beta - \gamma) \alpha}$$

    where $x = r/r_{200c}$, $r_{200c}$ is the comoving 200c radius. The profile can be 
    truncated according to whatever mass definition is used. The profile can also be
    normalized to enforce cosmic baryon abundance at a given radius.
    """

    def __init__(
        self,
        mass_def: ccl.halos.MassDef,
        rho0: float | Any = 1.0,
        alpha: float | Any = 1.0,
        beta: float | Any = 3.0,
        gamma: float | Any = 1.0,
        x_c: float | Any = 0.5,
        feedback_model: str | None = None,
        truncated: bool = False,
        rho_2h: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        version: str = "Zhao96",
    ):
        super(HaloProfileGNFW, self).__init__(mass_def=mass_def)
        """
        Args:
            mass_def: (`ccl.halos.MassDef`): ccl mass definition object.

            log10_rho0: (`float`): amplitude of the profile.
            alpha: (`float`): inner slope of the profile.
            beta: (`float`): outer slope of the profile.
            gamma: (`float`): slope of the transition between the inner and outer slopes.
            x_c: (`float`): scale radius of the profile.
            feedback_model: (`str` or `None`): feedback model to use. Either 'AGN' or 'SH' or None.
            truncated: (`bool`): whether to truncate the profile at the virial radius.
            rho_2h: (`Callable` or `None`): function to compute the 2-halo term.
        """
        self.mass_def = mass_def

        # GNFW Parameters
        self.rho0 = rho0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.x_c = x_c

        self.feedback_model = feedback_model
        self.truncated = truncated
        self.rho_2h = rho_2h
        self.version = version

    def normalize(self, 
                  cosmo: ccl.cosmology.Cosmology, 
                  rb: np.float64 | NDArray[np.float64], 
                  M: np.float64 | NDArray[np.float64], 
                  a: np.float64 | NDArray[np.float64], 
                  prof: ccl.halos.profile_base.HaloProfile, 
                  rmin: float = 1e-16, 
                  n_steps: int = 1000) -> None:
        """
        compute the value of $\rho_0$ for the GNFW profile by
        enforcing a sphere of radius $r_b$ to have cosmic abundance of
        baryons. Assigns a new value of $\rho_0$ to the object.

        Args:
            cosmo: (`pyccl.cosmology.Cosmology`): a Cosmology object.
            rb: (`float` or `NDArray[float]`): comoving baryon radius in Mpc.
            M: (`float` or `NDArray[float]`): halo mass in units of M_sun.
            a: (`float` or `NDArray[float]`): scale factor.
            prof: (`pyccl.halos.profiles.profile_base.HaloProfile`): halo profile.
            rmin: (`float`): minimum radius to integrate to.
            n_steps: (`int`): number of steps to use in the integration.
        """

        if self.rho0 != 1.0:
            self.rho0 = 1.0

        _r = np.geomspace(rmin, rb, n_steps)

        numerator = scipy.integrate.simps(y=_r**2 * prof.real(cosmo, _r, M, a), x=_r)
        denominator = scipy.integrate.simps(y=_r**2 * self.real(cosmo, _r, M, a), x=_r)

        self.rho0 = numerator / denominator

    def _real(self, 
              cosmo: ccl.cosmology.Cosmology, 
              r: np.float64 | NDArray[np.float64],
              M: np.float64 | NDArray[np.float64], 
              a: np.float64 | NDArray[np.float64]) -> NDArray[np.float64]:
        """comoving real space profile in units of [M_sun/Mpc^3].

        Args:
            cosmo: (`pyccl.cosmology.Cosmology`): a Cosmology object.
            r: (`float` or `NDArray[float]`): comoving radius in Mpc.
            M: (`float` or `NDArray[float]`): halo mass in units of M_sun.
            a: (`float` or `NDArray[float]`): scale factor.

        Returns:
            (`float` or `NDArray[float]`): halo profile. The shape of the
            output will be `(N_M, N_r)` where `N_r` and `N_m` are
            the sizes of `r` and `M` respectively.
        """

        # Generate 2D array by default
        r_use = cast(NDArray[np.float64], np.atleast_1d(r))
        M_use = cast(NDArray[np.float64], np.atleast_1d(M))

        if self.feedback_model is None:
            # FITTING PARAMETERS
            rho0 = self.rho0
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
            x_c = self.x_c

        elif self.feedback_model == "AGN": 
            # These relations were fit using the
            # Battaglia16 version of the GNFW profile
            self.version = "Battaglia16"
            # AGN FEEDBACK PARAMETERS

            rho0 = 4e3 * (M_use / 1e14) ** 0.29 * (1 / a) ** (-0.66)
            alpha = 0.88 * (M_use / 1e14) ** (-0.03) * (1 / a) ** 0.19
            beta = 3.83 * (M_use / 1e14) ** 0.04 * (1 / a) ** (-0.025)
            gamma = np.ones_like(M_use) * -0.2
            x_c = np.ones_like(M_use) * 0.5

            rho0 = np.repeat(rho0[:, np.newaxis], r_use.shape, 1)
            alpha = np.repeat(alpha[:, np.newaxis], r_use.shape, 1)
            beta = np.repeat(beta[:, np.newaxis], r_use.shape, 1)
            gamma = np.repeat(gamma[:, np.newaxis], r_use.shape, 1)
            x_c = np.repeat(x_c[:, np.newaxis], r_use.shape, 1)

        elif self.feedback_model == "SH":
            # These relations were fit using the
            # Battaglia16 version of the GNFW profile
            self.version = "Battaglia16"
            # SH FEEDBACK PARAMETERS
            rho0 = 1.9e4 * (M_use / 1e14) ** 0.09 * (1 / a) ** (-0.95)
            alpha = 0.70 * (M_use / 1e14) ** (-0.017) * (1 / a) ** 0.27
            beta = 4.43 * (M_use / 1e14) ** 0.005 * (1 / a) ** (0.037)
            gamma = np.ones_like(M_use) * -0.2
            x_c = np.ones_like(M_use) * 0.5

            rho0 = np.repeat(rho0[:, np.newaxis], r_use.shape, 1)
            alpha = np.repeat(alpha[:, np.newaxis], r_use.shape, 1)
            beta = np.repeat(beta[:, np.newaxis], r_use.shape, 1)
            gamma = np.repeat(gamma[:, np.newaxis], r_use.shape, 1)
            x_c = np.repeat(x_c[:, np.newaxis], r_use.shape, 1)

        else:  # pragma: no cover
            msg = "Feedback model not recognized. Please choose between AGN and SH or provide a param_dict."
            raise ValueError(msg)

        # comoving 200c radius
        R_200c = ccl.halos.MassDef200c.get_radius(cosmo, M_use, a) / a

        # comoving virial radius for truncation
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a

        # Compute profile
        x = (r_use[None, :]) / (R_200c[:, None])  # r/r_200c with shape of (N_M, N_r)

        # GNFW profile
        if self.version == "Zhao96":
            prof = rho0 * (
                (x / x_c) ** (-gamma)
                * (1 + (x / x_c) ** (1 / alpha)) ** (-(beta - gamma) * alpha)
            )
        elif self.version == "Battaglia16":
            gamma *= -1
            prof = rho0 * (
                (x / x_c) ** (gamma)
                * (1 + (x / x_c) ** alpha) ** (-(beta + gamma) / alpha)
            )
        else:  # pragma: no cover
            msg = "Version not recognized. Please choose between Zhao96, Battaglia16 or Amodeo23."
            raise ValueError(msg)

        # include 2-halo term
        if self.rho_2h is not None:
            prof += self.rho_2h(r_use)

        # truncate the profile if desired
        if self.truncated:
            prof[r_use[None, :] > R_M[:, None]] = 0

        # Make sure the output has the right shape
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
            
        return cast(NDArray[np.float64], prof)


class MatterProfile(ccl.halos.HaloProfile): # type: ignore[misc]
    def __init__(self, 
                 mass_def: ccl.halos.MassDef, 
                 concentration: ccl.halos.concentration, 
                 rho_2h: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
                 ):
        super(MatterProfile, self).__init__(
            mass_def=mass_def, concentration=concentration
        )

        self.mass_def = mass_def
        self.concentration = concentration
        self.rho_2h = rho_2h

        self.NFW = ccl.halos.HaloProfileNFW(
            mass_def=self.mass_def,
            concentration=self.concentration,
            truncated=False,
            fourier_analytic=True,
            projected_analytic=True,
            cumul2d_analytic=True,
        )

    def _real(self, 
              cosmo: ccl.cosmology.Cosmology, 
              r: np.float64 | NDArray[np.float64],
              M: np.float64 | NDArray[np.float64], 
              a: np.float64 | NDArray[np.float64]) -> NDArray[np.float64]:
        """Real space profile.

         Args:
            cosmo: (`pyccl.cosmology.Cosmology`): a Cosmology object.
            r: (`float` or `NDArray[float]`): comoving radius in Mpc.
            M: (`float` or `NDArray[float]`): halo mass in units of M_sun.
            a: (`float` or `NDArray[float]`): scale factor.

        Returns:
             (`float` or `NDArray[float]`): halo profile. The shape of the
             output will be `(N_M, N_r)` where `N_r` and `N_m` are
             the sizes of `r` and `M` respectively.
        """
        r_use = cast(NDArray[np.float64], np.atleast_1d(r))
        M_use = cast(NDArray[np.float64], np.atleast_1d(M))

        prof = self.NFW.real(cosmo, r_use, M_use, a)

        if self.rho_2h is not None:
            prof += self.rho_2h(r_use)

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return cast(NDArray[np.float64], prof)
