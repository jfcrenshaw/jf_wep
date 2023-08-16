"""Functions for calculating Zernikes and related values."""
import galsim
import numpy as np

from jf_wep.instrument import Instrument


def createGalsimZernike(
    coeffs: np.ndarray,
    obscuration: float = 0.61,
) -> galsim.zernike.Zernike:
    """Create a GalSim Zernike object with the given coefficients.

    Parameters
    ----------
    coeffs : np.ndarray
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
        np.concatenate([np.zeros(4), coeffs]), R_inner=obscuration
    )


def createZernikeBasis(
    jmax: int,
    instrument: Instrument,
    nPixels: int,
) -> np.ndarray:
    """Create a basis of Zernike polynomials for Noll indices >= 4.

    This function is evaluated on a grid of normalized pupil coordinates,
    where these coordinates are normalized pupil coordinates. Normalized
    pupil coordinates are defined such that u^2 + v^2 = 1 is the edge of
    the pupil, and u^2 + v^2 = obscuration^2 is the edge of the central 
    obscuration.
    """
    return galsim.zernike.zernikeBasis(
        jmax,
        *instrument.createPupilGrid(nPixels),
        R_inner=instrument.obscuration,
    )


def zernikeEval(
    u: np.ndarray,
    v: np.ndarray,
    coeffs: np.ndarray,
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
    coeffs : np.ndarray
        Zernike coefficients for Noll indices >= 4, in any units.
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.61, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    np.ndarray
        Values of the Zernike series at the given points. Has the same
        shape as u and v, and the same units as coeffs.
    """
    # Create the Galsim Zernike object
    galsimZernike = createGalsimZernike(coeffs, obscuration)

    # And evaluate on the grid
    return galsimZernike(u, v)


def zernikeGradEval(
    u: np.ndarray,
    v: np.ndarray,
    uOrder: int,
    vOrder: int,
    coeffs: np.ndarray,
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
    coeffs : np.ndarray
        Zernike coefficients for Noll indices >= 4, in any units.
    obscuration : float, optional
        The fractional obscuration.
        (the default is 0.61, corresponding to the Simonyi Survey Telescope.)

    Returns
    -------
    np.ndarray
        Values of the Zernike series at the given points. Has the same
        shape as u and v, and the same units as coeffs.
    """
    # Create the Galsim Zernike object
    galsimZernike = createGalsimZernike(coeffs, obscuration)

    # Apply derivatives
    for _ in range(uOrder):
        galsimZernike = galsimZernike.gradX
    for _ in range(vOrder):
        galsimZernike = galsimZernike.gradY

    # And evaluate on the grid
    return galsimZernike(u, v)


def convertZernikesToPsfWidth(coeffs: np.ndarray, obscuration: float) -> np.ndarray:
    raise NotImplementedError
