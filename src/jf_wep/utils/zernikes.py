"""Functions for calculating Zernikes and related values."""
from typing import Tuple

import galsim
import numpy as np


def createGalsimZernike(
    zkCoeff: np.ndarray,
    obscuration: float = 0.61,
) -> galsim.zernike.Zernike:
    """Create a GalSim Zernike object with the given coefficients.

    Parameters
    ----------
    zkCoeff : np.ndarray
        Zernike coefficients for Noll indices >= 4, in any units.
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.61, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    galsim.zernike.Zernike
        A GalSim Zernike object
    """
    return galsim.zernike.Zernike(
        np.concatenate([np.zeros(4), zkCoeff]), R_inner=obscuration
    )


def createZernikeBasis(
    u: np.ndarray,
    v: np.ndarray,
    jmax: int = 28,
    obscuration: float = 0.61,
) -> np.ndarray:
    """Create a basis of Zernike polynomials for Noll indices >= 4.

    This function is evaluated on a grid of normalized pupil coordinates,
    where these coordinates are normalized pupil coordinates. Normalized
    pupil coordinates are defined such that u^2 + v^2 = 1 is the edge of
    the pupil, and u^2 + v^2 = obscuration^2 is the edge of the central
    obscuration.
    """
    return galsim.zernike.zernikeBasis(jmax, u, v, R_inner=obscuration)[4:]


def createZernikeGradBasis(
    u: np.ndarray,
    v: np.ndarray,
    jmax: int = 28,
    obscuration: float = 0.61,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a basis of Zernike gradient polynomials for Noll indices >= 4."""
    # Get the Noll coefficient array for the u derivates
    nollCoeffU = galsim.zernike._noll_coef_array_xy_gradx(jmax, obscuration)
    nollCoeffU = nollCoeffU[:, :, 3:]  # Keep only Noll indices >= 4

    # Evaluate the polynomials
    dzkdu = np.array(
        [
            galsim.utilities.horner2d(u, v, nc, dtype=float)
            for nc in nollCoeffU.transpose(2, 0, 1)
        ]
    )

    # Repeat for v
    nollCoeffV = galsim.zernike._noll_coef_array_xy_grady(jmax, obscuration)
    nollCoeffV = nollCoeffV[:, :, 3:]  # Keep only Noll indices >= 4
    dzkdv = np.array(
        [
            galsim.utilities.horner2d(u, v, nc, dtype=float)
            for nc in nollCoeffV.transpose(2, 0, 1)
        ]
    )

    return dzkdu, dzkdv


def zernikeEval(
    u: np.ndarray,
    v: np.ndarray,
    zkCoeff: np.ndarray,
    obscuration: float = 0.61,
) -> None:
    """Evaluate the Zernike series.

    This function is evaluated at the provided u and v coordinates, where
    these coordinates are normalized pupil coordinates. Normalized pupil
    coordinates are defined such that u^2 + v^2 = 1 is the edge of the
    pupil, and u^2 + v^2 = obscuration^2 is the edge of the central
    obscuration.

    Parameters
    ----------
    u : np.ndarray
        The x normalized pupil coordinate(s).
    v : np.ndarray
        The y normalized pupil coordinate(s). Must be same shape as u.
    zkCoeff : np.ndarray
        Zernike coefficients for Noll indices >= 4, in any units.
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.61, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    np.ndarray
        Values of the Zernike series at the given points. Has the same
        shape as u and v, and the same units as zkCoeff.
    """
    # Create the Galsim Zernike object
    galsimZernike = createGalsimZernike(zkCoeff, obscuration)

    # And evaluate on the grid
    return galsimZernike(u, v)


def zernikeGradEval(
    u: np.ndarray,
    v: np.ndarray,
    uOrder: int,
    vOrder: int,
    zkCoeff: np.ndarray,
    obscuration: float = 0.61,
) -> np.ndarray:
    """Evaluate the gradient of the Zernike series.

    This function is evaluated at the provided u and v coordinates, where
    these coordinates are normalized pupil coordinates. Normalized pupil
    coordinates are defined such that u^2 + v^2 = 1 is the edge of the
    pupil, and u^2 + v^2 = obscuration^2 is the edge of the central
    obscuration.

    Parameters
    ----------
    u : np.ndarray
        The x normalized pupil coordinate(s).
    v : np.ndarray
        The y normalized pupil coordinate(s). Must be same shape as u.
    uOrder : int
        The number of u derivatives to apply.
    vOrder : int
        The number of v derivatives to apply.
    zkCoeff : np.ndarray
        Zernike coefficients for Noll indices >= 4, in any units.
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.61, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    np.ndarray
        Values of the Zernike series at the given points. Has the same
        shape as u and v, and the same units as zkCoeff.
    """
    # Create the Galsim Zernike object
    galsimZernike = createGalsimZernike(zkCoeff, obscuration)

    # Apply derivatives
    for _ in range(uOrder):
        galsimZernike = galsimZernike.gradX
    for _ in range(vOrder):
        galsimZernike = galsimZernike.gradY

    # And evaluate on the grid
    return galsimZernike(u, v)


def getPsfGradPerZernike(
    jmax: int = 28,
    diameter: float = 8.36,
    obscuration: float = 0.612,
) -> np.ndarray:
    """Get the derivative of the PSF FWHM with respect to each Zernike.

    Parameters
    ----------
    jmax : int, optional
        The maximum Zernike Noll index to estimate.
        (the default is 28.)
    diameter : float
        The diameter of the telescope aperture, in meters.
        (the default, 8.36, corresponds to the LSST primary mirror.)
    obscuration : float
        The fractional obscuration of the telescope aperture.
        (the default, 0.612, corresponds to the LSST primary mirror.)

    Returns
    -------
    np.ndarray
        Gradient of the PSF FWHM with respect to the corresponding Zernike.
        Units are arcsec / meter.
    """
    # Calculate the aperture radii
    Router = diameter / 2
    Rinner = obscuration * Router

    # Calculate the conversion factors
    conversionFactors = np.zeros(jmax + 1)
    for i in range(4, jmax + 1):
        # Set coefficients for this Noll index: coefs = [0, 0, ..., 1]
        # Note the first coefficient is Noll index 0, which does not exist and
        # is therefore always ignored by galsim
        coefs = [0] * i + [1]

        # Create the Zernike polynomial with these coefficients
        Z = galsim.zernike.Zernike(coefs, R_outer=Router, R_inner=Rinner)

        # We can calculate the size of the PSF from the RMS of the gradient of
        # the wavefront. The gradient of the wavefront perturbs photon paths.
        # The RMS quantifies the size of the collective perturbation.
        # If we expand the wavefront gradient in another series of Zernike
        # polynomials, we can exploit the orthonormality of the Zernikes to
        # calculate the RMS from the Zernike coefficients.
        rmsTilt = np.sqrt(np.sum(Z.gradX.coef**2 + Z.gradY.coef**2) / 2)

        # Convert to arcsec per meter
        rmsTilt = np.rad2deg(rmsTilt) * 3600

        # Convert rms -> fwhm
        fwhmTilt = 2 * np.sqrt(2 * np.log(2)) * rmsTilt

        # Save this conversion factor
        conversionFactors[i] = fwhmTilt

    return conversionFactors[4:]


def convertZernikesToPsfWidth(
    zkCoeff: np.ndarray,
    diameter: float = 8.36,
    obscuration: float = 0.612,
) -> np.ndarray:
    """Convert Zernike amplitudes to quadrature contribution to the PSF FWHM.

    Parameters
    ----------
    zkCoeff : np.ndarray
        Zernike coefficients for Noll indices >= 4, in meters.
    diameter : float
        The diameter of the telescope aperture, in meters.
        (the default, 8.36, corresponds to the LSST primary mirror.)
    obscuration : float
        The fractional obscuration of the telescope aperture.
        (the default, 0.612, corresponds to the LSST primary mirror.)

    Returns
    -------
    np.ndarray
        Quadrature contribution of each Zernike mode to the PSF FWHM
        (in arcseconds).

    Notes
    -----
    Converting Zernike amplitudes to their quadrature contributions to the PSF
    FWHM allows for easier physical interpretation of Zernike amplitudes and
    the performance of the AOS system.

    For example, image we have a true set of zernikes, [Z4, Z5, Z6], such that
    ConvertZernikesToPsfWidth([Z4, Z5, Z6]) = [0.1, -0.2, 0.3] arcsecs.
    These Zernike perturbations increase the PSF FWHM by
    sqrt[(0.1)^2 + (-0.2)^2 + (0.3)^2] ~ 0.37 arcsecs.

    If the AOS perfectly corrects for these perturbations, the PSF FWHM will
    not increase in size. However, imagine the AOS estimates zernikes, such
    that ConvertZernikesToPsfWidth([Z4, Z5, Z6]) = [0.1, -0.3, 0.4] arcsecs.
    These estimated Zernikes, do not exactly match the true Zernikes above.
    Therefore, the post-correction PSF will still be degraded with respect to
    the optimal PSF. In particular, the PSF FWHM will be increased by
    sqrt[(0.1 - 0.1)^2 + (-0.2 - (-0.3))^2 + (0.3 - 0.4)^2] ~ 0.14 arcsecs.

    This conversion depends on a linear approximation that begins to break down
    for RSS(dFWHM) > 0.20 arcsecs. Beyond this point, the approximation tends
    to overestimate the PSF degradation. In other words, if
    sqrt(sum( dFWHM^2 )) > 0.20 arcsec, it is likely that dFWHM is
    over-estimated. However, the point beyond which this breakdown begins
    (and whether the approximation over- or under-estimates dFWHM) can change,
    depending on which Zernikes have large amplitudes. In general, if you have
    large Zernike amplitudes, proceed with caution!
    Note that if the amplitudes Z_est and Z_true are large, this is okay, as
    long as |Z_est - Z_true| is small.

    For a notebook demonstrating where the approximation breaks down:
    https://gist.github.com/jfcrenshaw/24056516cfa3ce0237e39507674a43e1
    """
    # Calculate jmax
    jmax = 4 + zkCoeff.shape[-1] - 1

    # Calculate the conversion factors for each zernike
    conversionFactors = getPsfGradPerZernike(
        jmax=jmax,
        diameter=diameter,
        obscuration=obscuration,
    )

    return conversionFactors * zkCoeff
