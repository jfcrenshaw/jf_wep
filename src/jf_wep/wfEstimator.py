"""Class that provides the main interface for wavefront estimation.

This class wraps all the boilerplate for choosing different wavefront
estimation algorithms, supplying images in the correct format, etc.
"""
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from jf_wep.donutImage import DonutImage
from jf_wep.instrument import Instrument
from jf_wep.utils import mergeParams, loadConfig
from jf_wep.wfAlgorithms.tie import TIEAlgorithm
from jf_wep.wfAlgorithms.wfAlgorithm import WfAlgorithm
from jf_wep.zernikeObject import ZernikeObject


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
        If a dictionary, it is assumed to hold keywords for configuration.
        If a WfAlgorithm object, that object is just used.
        If None, the algorithm defaults are used.
        Note that setting camType, instParams, or jmax will cause
        WfEstimator to overwrite the algorithm's Instrument and
        ZernikeObject configurations that were read from here.
    instConfig : Path or str or dict or Instrument, optional
        Instrument configuration. If a Path or string, it is assumed this
        points to a config file, which is used to configure the Instrument.
        If a dictionary, it is assumed to hold keywords for configuration.
        If an Instrument object, that object is just used.
    jmax : int, optional
        The wavefront estimators return Zernike coefficients from Noll
        index 4 up to jmax, inclusive. Must be an integer >= 4.
    units : str, optional
        Units in which the wavefront is returned. Options are "nm", "um",
        or "arcsecs".
    shapeMode : str, optional
        How to handle the image shape. If "strict", an error will be
        raised when trying to estimate the wavefront for an image of
        the wrong size. NEED TO WRITE MORE OF THIS LATER.

    Raises
    ------
    ValueError
        Invalid algo name, unit name, or shapeMode
    TypeError
        If algo_params is not a dictionary or None
    """

    def __init__(
        self,
        configFile: Union[Path, str, None] = "policy/wfEstimator.yaml",
        algo: Optional[str] = None,
        algoConfig: Union[Path, str, dict, WfAlgorithm, None] = None,
        instConfig: Union[Path, str, dict, Instrument, None] = None,
        jmax: Optional[int] = None,
        units: Optional[str] = None,
        shapeMode: Optional[str] = None,
    ) -> None:
        self.config(
            configFile=configFile,
            algo=algo,
            algoConfig=algoConfig,
            instConfig=instConfig,
            jmax=jmax,
            units=units,
            shapeMode=shapeMode,
        )

    def config(
        self,
        configFile: Union[Path, str, None] = None,
        algo: Optional[str] = None,
        algoConfig: Union[WfAlgorithm, Path, str, dict, None] = None,
        instConfig: Union[Instrument, Path, str, dict, None] = None,
        jmax: Optional[int] = None,
        units: Optional[str] = None,
        shapeMode: Optional[str] = None,
    ) -> None:
        """Configure the wavefront estimator.

        For details on the parameters, see the class docstring.
        """
        # Merge the keyword arguments with the default parameters
        params = mergeParams(
            configFile,
            algo=algo,
            algoConfig=algoConfig,
            instConfig=instConfig,
            jmax=jmax,
            units=units,
            shapeMode=shapeMode,
        )

        # Instantiate the algorithm
        algo = params["algo"]
        if algo is not None:
            allowed_algos = ["tie"]
            if algo not in allowed_algos:
                raise ValueError(
                    f"Algorithm '{algo}' not supported. "
                    f"Please choose one of {str(allowed_algos)[1:-1]}."
                )
            if algo == "tie":
                self._algo = TIEAlgorithm()

        # Configure the algorithm
        algoConfig = params["algoConfig"]
        if algoConfig is not None:
            self._algo = loadConfig(algoConfig, self.algo)

        # Set the Instrument
        instConfig = params["instConfig"]
        if instConfig is not None:
            self._instrument = loadConfig(instConfig, Instrument)

        # Create the ZernikeObject
        jmax = params["jmax"]
        if jmax is not None:
            self._zernikeObject = ZernikeObject(
                jmax=jmax,
                instConfig=self.instrument,
            )
        # If we updated the instrument, we also must update the ZernikeObject
        elif instConfig is not None:
            self._zernikeObject = ZernikeObject(
                jmax=self.jmax,
                instConfig=self.instrument,
            )

        # Propagate changes to Instrument and jmax to the algorithm
        if instConfig is not None or jmax is not None:
            self.algo.config(instConfig=self.instrument, jmax=jmax)

        # Set the units
        units = params["units"]
        if units is not None:
            allowed_units = ["nm", "um", "arcsecs"]
            if units not in allowed_units:
                raise ValueError(
                    f"Unit '{units}' not supported. "
                    f"Please choose one of {str(allowed_units)[1:-1]}."
                )
            self._units = units

        # Set the shape mode
        shapeMode = params["shapeMode"]
        if shapeMode is not None:
            warnings.warn("\nSHAPE MODE NOT IMPLEMENTED + DOCSTRING\n")
            allowed_shapeModes = ["strict"]
            if shapeMode not in allowed_shapeModes:
                raise ValueError(
                    f"shapeMode '{shapeMode}' not supported. "
                    f"Please choose one of {str(allowed_shapeModes)[1:-1]}."
                )
            self._shapeMode = shapeMode

    @property
    def instrument(self) -> Instrument:
        """Return the Instrument."""
        return self._instrument

    @property
    def zernikeObject(self) -> ZernikeObject:
        """Return the ZernikeObject."""
        return self._zernikeObject

    @property
    def jmax(self) -> int:
        """Return the maximum Zernike Noll index to be estimated.

        For details about this parameter, see the class docstring.
        """
        return self.zernikeObject.jmax

    @property
    def algo(self) -> WfAlgorithm:
        """Return the wavefront estimation algorithm.

        For details about this parameter, see the class docstring.
        """
        return self._algo

    @property
    def units(self) -> str:
        """Return the wavefront units.

        For details about this parameter, see the class docstring.
        """
        return self._units

    @property
    def shapeMode(self) -> str:
        """Return the shape mode.

        For details about this parameter, see the class docstring.
        """
        return self._shapeMode

    def estimateWf(
        self,
        I1: DonutImage,
        I2: Union[DonutImage, None] = None,
    ) -> np.ndarray:
        """Estimate the wavefront for the image or pair of images.

        The wavefront is returned as an array of Zernike coefficients.
        The number of Zernike coefficients and the units of the
        coefficients are determined by the jmax and units parameters in
        the self.config() method.

        Parameters
        ----------
        I1 : DonutImage
            An image object containing an intra- or extra-focal donut image.
        I2 : DonutImage, optional
            A second image, on the opposite side of focus from I1.
            (the default is None)

        Returns
        -------
        np.ndarray
            Numpy array of the Zernike coefficients estimated from the image
            or pair of images.

        Raises
        ------
        ValueError
            If I1 and I2 are on the same side of focus.
        """
        # If I2 provided, check that I1 and I2 are on opposite sides of focus
        if I2 is not None and I2.defocalType == I1.defocalType:
            raise ValueError(
                "If you provide I2, it must be on the "
                "opposite side of focus from I1."
            )

        # Get the estimated wavefront in terms of Zernike coefficients in nm
        zk = self.algo.estimateWf(I1, I2)  # type: ignore

        # Convert to desired units
        if self.units in ["um", "arcsecs"]:
            zk = 1e-3 * zk
        if self.units == "arcsecs":
            zk = self.zernikeObject.convertZernikesToPsfWidth(zk)

        return zk
