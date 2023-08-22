"""Class that provides the main interface for wavefront estimation.

This class wraps all the boilerplate for choosing different wavefront
estimation algorithms, supplying images in the correct format, etc.
"""
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from jf_wep.donutStamp import DonutStamp
from jf_wep.instrument import Instrument
from jf_wep.utils import loadConfig, mergeParams
from jf_wep.wfAlgorithms.tie import TIEAlgorithm
from jf_wep.wfAlgorithms.wfAlgorithm import WfAlgorithm


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
        units: Optional[str] = None,
        shapeMode: Optional[str] = None,
    ) -> None:
        self.config(
            configFile=configFile,
            algo=algo,
            algoConfig=algoConfig,
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
        I1: DonutStamp,
        I2: Union[DonutStamp, None] = None,
    ) -> np.ndarray:
        """Estimate the wavefront for the image or pair of images.

        The wavefront is returned as an array of Zernike coefficients.
        The number of Zernike coefficients and the units of the
        coefficients are determined by the jmax and units parameters in
        the self.config() method.

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
            Numpy array of the Zernike coefficients estimated from the image
            or pair of images.

        Raises
        ------
        ValueError
            If I1 and I2 are on the same side of focus.
        """
        # Get the estimated wavefront in terms of Zernike coefficients in nm
        zk = self.algo.estimateWf(I1, I2)  # type: ignore

        # Convert to desired units
        if self.units in ["um", "arcsecs"]:
            zk = 1e-3 * zk
        if self.units == "arcsecs":
            raise NotImplementedError

        return zk
