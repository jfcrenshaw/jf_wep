"""Class for Zernike polynomials and related methods."""
from pathlib import Path
from typing import Optional, Union

import galsim
import numpy as np

from jf_wep.instrument import Instrument
from jf_wep.utils import loadConfig, mergeParams


class ZernikeObject:
    def __init__(
        self,
        configFile: Union[Path, str, None] = "policy/zernikeObject.yaml",
        instConfig: Union[Path, str, dict, Instrument, None] = None,
        jmax: Optional[int] = None,
    ) -> None:
        """Class that wraps methods for Zernike polynomials.

        Parameters
        ----------
        configFile : Path or str, optional
            Path to file specifying values for the other parameters. If the
            path starts with "policy/", it will look in the policy directory.
            Any explicitly passed parameters override values found in this file
            (the default is policy/zernikeObject.yaml)
        instConfig : Path or str or dict or Instrument, optional
            Instrument configuration. If a Path or string, it is assumed this
            points to a config file, which is used to configure the Instrument.
            If a dictionary, it is assumed to hold keywords for configuration.
            If an Instrument object, that object is just used.
        jmax : int, optional
            The maximum Zernike Noll index, which must be an integer >= 4.
            Note the minimum Noll index is always 4.

        Raises
        ------
        TypeError
            jmax is not an integer
        ValueError
            jmax is not >= 4
        """
        self.config(
            configFile=configFile,
            jmax=jmax,
            instConfig=instConfig,
        )

    def config(
        self,
        configFile: Union[Path, str, None] = None,
        jmax: Optional[int] = None,
        instConfig: Union[Path, str, dict, Instrument, None] = None,
    ) -> None:
        # Merge keyword arguments with the default parameters
        params = mergeParams(
            configFile,
            jmax=jmax,
            instConfig=instConfig,
        )

        # Set jmax
        jmax = params["jmax"]
        if jmax is not None:
            if not isinstance(jmax, int) or (
                isinstance(jmax, float) and jmax % 1 != 0
            ):
                raise TypeError("jmax must be an integer.")
            if jmax < 4:
                raise ValueError("jmax must be > 4.")
            self._jmax = int(jmax)

        # Set the Instrument
        instConfig = params["instConfig"]
        if instConfig is not None:
            self._instrument = loadConfig(instConfig, Instrument)

    @property
    def jmax(self) -> int:
        """The maximum Noll index for the Zernikes."""
        return self._jmax

    @property
    def instrument(self) -> Instrument:
        """The Instrument."""
        return self._instrument

    def createZernikeBasis(self, nPixels: int) -> np.ndarray:
        """Create a Zernike basis polynomials for Noll indices 4 - self.jmax.

        The Noll indices 4 - self.jmax are inclusive.

        The Zernike basis polynomials are evaluated on a grid of normalized
        pupil coordinates, u and v. These coordinates are defined such that
        u^2 + v^2 = 1 is the edge of the pupil, and u^2 + v^2 = obscuration^2
        is the edge of the central obscuration.

        Parameters
        ----------
        nPixels : int
            The number of pixels on a side.

        Returns
        -------
        np.ndarray
            3D array of Zernike basis polynomials. The first axis enumerates
            the Zernike polynomials; the 2nd and 3rd axes are the v and u
            axes, respectively.
        """
        return galsim.zernike.zernikeBasis(
            self.jmax,
            *self.instrument.createPupilGrid(nPixels),
            R_inner=self.instrument.obscuration,
        )

    def _galsimZk(self, coeffs: np.ndarray) -> galsim.zernike.Zernike:
        """Return a Galsim Zernike Object with the given coefficients.

        Parameters
        ----------
        coeffs : np.ndarray
            Zernike coefficients for Noll indices 4-self.jmax, inclusive.

        Returns
        -------
        galsim.zernike.Zernike
            A Galsim Zernike object
        """
        return galsim.zernike.Zernike(
            np.concatenate([np.zeros(4), coeffs]),
            R_inner=self.instrument.obscuration,
        )

    def eval(
        self,
        coeffs: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the Zernike series.

        This function is evaluated at the provided u and v coordinates, where
        these coordinates are normalized pupil coordinates. Normalized pupil
        coordinates are defined such that u^2 + v^2 = 1 is the edge of the
        pupil, and u^2 + v^2 = obscuration^2 is the edge of the central
        obscuration.

        Parameters
        ----------
        coeffs : np.ndarray
            Array of Zernike coefficients, starting at Noll index 4.
            Units can be whatever you like.
        u : np.ndarray
            The x normalized pupil coordinate(s).
        v : np.ndarray
            The y normalized pupil coordinate(s). Must be same shape as u.

        Returns
        -------
        np.ndarray
            Values of the Zernike series at the given points. Has the same
            shape as u and v, and the same units as coeffs.
        """
        # Get the Galsim Zernike object
        galsimZk = self._galsimZk(coeffs)

        return galsimZk(u, v)

    def gradEval(
        self,
        coeffs: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        uOrder: int,
        vOrder: int,
    ) -> np.ndarray:
        """Evaluate the gradient of the Zernike series.

        This function is evaluated at the provided u and v coordinates, where
        these coordinates are normalized pupil coordinates. Normalized pupil
        coordinates are defined such that u^2 + v^2 = 1 is the edge of the
        pupil, and u^2 + v^2 = obscuration^2 is the edge of the central
        obscuration.

        Parameters
        ----------
        coeffs : np.ndarray
            Array of Zernike coefficients, starting at Noll index 4.
            Units can be whatever you like.
        u : np.ndarray
            The x normalized pupil coordinate(s).
        v : np.ndarray
            The y normalized pupil coordinate(s). Must be same shape as u.
        uOrder : int
            The number of u derivatives to apply.
        vOrder : int
            The number of v derivatives to apply.

        Returns
        -------
        np.ndarray
            Values of the Zernike series at the given points. Has the same
            shape as u and v.
        """
        galsimZk = self._galsimZk(coeffs)

        for _ in range(uOrder):
            galsimZk = galsimZk.gradX
        for _ in range(vOrder):
            galsimZk = galsimZk.gradY

        return galsimZk(u, v)

    def gradUEval(
        self,
        coeffs: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the u-derivative of the Zernike series.

        This function is evaluated at the provided u and v coordinates, where
        these coordinates are normalized pupil coordinates. Normalized pupil
        coordinates are defined such that u^2 + v^2 = 1 is the edge of the
        pupil, and u^2 + v^2 = obscuration^2 is the edge of the central
        obscuration.

        Parameters
        ----------
        coeffs : np.ndarray
            Array of Zernike coefficients, starting at Noll index 4.
            Units can be whatever you like.
        u : np.ndarray
            The x normalized pupil coordinate(s).
        v : np.ndarray
            The y normalized pupil coordinate(s). Must be same shape as u.

        Returns
        -------
        np.ndarray
            Values of the u-derivative of the Zernike series at the given
            points. Has the same shape as u and v, and same units as coeffs.
        """
        return self.gradEval(coeffs, u, v, uOrder=1, vOrder=0)

    def gradVEval(
        self,
        coeffs: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the v-derivative of the Zernike series.

        This function is evaluated at the provided u and v coordinates, where
        these coordinates are normalized pupil coordinates. Normalized pupil
        coordinates are defined such that u^2 + v^2 = 1 is the edge of the
        pupil, and u^2 + v^2 = obscuration^2 is the edge of the central
        obscuration.

        Parameters
        ----------
        coeffs : np.ndarray
            Array of Zernike coefficients, starting at Noll index 4.
            Units can be whatever you like.
        u : np.ndarray
            The x normalized pupil coordinate(s).
        v : np.ndarray
            The y normalized pupil coordinate(s). Must be same shape as u.

        Returns
        -------
        np.ndarray
            Values of the v-derivative of the Zernike series at the given
            points. Has the same shape as u and v, and same units as coeffs.
        """
        return self.gradEval(coeffs, u, v, uOrder=0, vOrder=1)

    def convertZernikesToPsfWidth(
        self,
        zk: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError
