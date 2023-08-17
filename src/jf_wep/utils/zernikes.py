"""Functions for calculating Zernikes and related values."""
from typing import Tuple

import galsim
import numpy as np

from jf_wep.instrument import Instrument

from functools import lru_cache


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


@lru_cache
def createZernikeBasis(
    jmax: int = 28,
    instrument: Instrument = Instrument(),
    nPixels: int = 160,
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
    )[4:]


@lru_cache
def createZernikeGradBasis(
    jmax: int = 28,
    instrument: Instrument = Instrument(),
    nPixels: int = 160,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a basis of Zernike gradient polynomials for Noll indices >= 4."""
    # Create the normalized pupil grid
    uPupil, vPupil = instrument.createPupilGrid(nPixels)

    # Get the Noll coefficient array for the u derivates
    nollCoeffU = galsim.zernike._noll_coef_array_xy_gradx(
        jmax, instrument.obscuration
    )
    nollCoeffU = nollCoeffU[:, :, 3:]  # Keep only Noll indices >= 4

    # Evaluate the polynomials
    dzkdu = np.array(
        [
            galsim.utilities.horner2d(uPupil, vPupil, nc, dtype=float)
            for nc in nollCoeffU.transpose(2, 0, 1)
        ]
    )

    # Repeat for v
    nollCoeffV = galsim.zernike._noll_coef_array_xy_grady(
        jmax, instrument.obscuration
    )
    nollCoeffV = nollCoeffV[:, :, 3:]  # Keep only Noll indices >= 4
    dzkdv = np.array(
        [
            galsim.utilities.horner2d(uPupil, vPupil, nc, dtype=float)
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


def convertZernikesToPsfWidth(
    zkCoeff: np.ndarray, obscuration: float
) -> np.ndarray:
    raise NotImplementedError
