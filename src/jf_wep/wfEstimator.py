"""Class that provides the main interface for wavefront estimation.

This class wraps all the boilerplate for choosing different wavefront
estimation algorithms, supplying images in the correct format, etc.
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np

from jf_wep.donutStamp import DonutStamp
from jf_wep.instrument import Instrument
from jf_wep.utils import loadConfig, mergeParams, convertZernikesToPsfWidth
from jf_wep.wfAlgorithms.wfAlgorithm import WfAlgorithm
from jf_wep.wfAlgorithms.wfAlgorithmFactory import WfAlgorithmFactory


class WfEstimator:
    """Class providing a high-level interface for wavefront estimation.

    Any explicitly passed parameters override the values found in configFile.

    Parameters
    ----------
    configFile : Path or str, optional
        Path to file specifying values for the other parameters. If the
        path starts with "policy/", it will look in the policy directory.
        Any explicitly passed parameters override values found in this file
        (the default is policy/wfEstimator.yaml)
    algo : str, optional
        Name of the algorithm to use. Options are "tie".
    algoConfig : Path or str or dict or WfAlgorithm, optional
        Algorithm configuration. If a Path or string, it is assumed this
        points to a config file, which is used to configure the algorithm.
        If the path begins with "policy/", then it is assumed the path is
        relative to the policy directory. If a dictionary, it is assumed
        to hold keywords for configuration. If a WfAlgorithm object, that
        object is just used. If None, the algorithm defaults are used.
    instConfig : Path or str or dict or Instrument, optional
        Instrument configuration. If a Path or string, it is assumed this
        points to a config file, which is used to configure the Instrument.
        If a dictionary, it is assumed to hold keywords for configuration.
        If an Instrument object, that object is just used.
    jmax : int, optional
        The maximum Zernike Noll index to estimate.
    units : str, optional
        Units in which the wavefront is returned. Options are "nm", "um",
        or "arcsecs".
    """

    def __init__(
        self,
        configFile: Union[Path, str, None] = "policy/wfEstimator.yaml",
        algoName: Optional[str] = None,
        algoConfig: Union[Path, str, dict, WfAlgorithm, None] = None,
        instConfig: Union[Path, str, dict, Instrument, None] = None,
        jmax: Optional[int] = None,
        units: Optional[str] = None,
    ) -> None:
        # Merge keyword arguments with defaults from configFile
        params = mergeParams(
            configFile,
            algoName=algoName,
            algoConfig=algoConfig,
            jmax=jmax,
            units=units,
        )

        # Set the algorithm
        self._algo = WfAlgorithmFactory.createWfAlgorithm(
            params["algoName"], params["algoConfig"]
        )

        # Set the instrument
        self._instrument = loadConfig(instConfig, Instrument)

        # Set the other parameters
        self.jmax = params["jmax"]
        self.units = params["units"]

    @property
    def algo(self) -> WfAlgorithm:
        """Return the WfAlgorithm object."""
        return self._algo

    @property
    def instrument(self) -> Instrument:
        """Return the Instrument object."""
        return self._instrument

    @property
    def jmax(self) -> int:
        """Return the maximum Zernike Noll index that will be estimated."""
        return self._jmax

    @jmax.setter
    def jmax(self, value: int) -> None:
        """Set jmax"""
        value = int(value)
        if value < 4:
            raise ValueError("jmax must be greater than or equal to 4.")
        self._jmax = value

    @property
    def units(self) -> str:
        """Return the wavefront units.

        For details about this parameter, see the class docstring.
        """
        return self._units

    @units.setter
    def units(self, value: str) -> None:
        """Set the units of the Zernike coefficients."""
        allowed_units = ["m", "um", "nm", "arcsecs"]
        if value not in allowed_units:
            raise ValueError(
                f"Unit '{value}' not supported. "
                f"Please choose one of {str(allowed_units)[1:-1]}."
            )
        self._units = value

    def estimateWf(
        self,
        I1: DonutStamp,
        I2: Optional[DonutStamp] = None,
    ) -> np.ndarray:
        """Estimate the wavefront from the stamp or pair of stamps.

        Parameters
        ----------
        I1 : DonutStamp
            A stamp object containing an intra- or extra-focal donut image.
        I2 : DonutStamp, optional
            A second stamp, on the opposite side of focus from I1.
            (the default is None)

        Returns
        -------
        np.ndarray
            Numpy array of the Zernike coefficients estimated from the stamp
            or pair of stamp. The array contains Noll coefficients from
            4 - self.jmax, inclusive. The unit is determined by self.units.

        Raises
        ------
        ValueError
            If I1 and I2 are on the same side of focus.
        """
        # Estimated wavefront (in meters)
        zk = self.algo.estimateWf(I1, I2, self.jmax, self.instrument)

        # Convert to desired units
        if self.units == "m":
            return zk
        elif self.units == "um":
            return 1e6 * zk
        elif self.units == "nm":
            return 1e9 * zk
        elif self.units == "arcsecs":
            return convertZernikesToPsfWidth(
                zk,
                self.instrument.diameter,
                self.instrument.obscuration,
            )
        else:
            raise RuntimeError(
                f"Conversion to unit '{self.units}' not supported."
            )

        return zk
