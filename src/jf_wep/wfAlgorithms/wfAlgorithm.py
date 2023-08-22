"""Base class for wavefront estimation algorithms.

Use this class to define the universal structure for all algorithm classes.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jf_wep.donutStamp import DonutStamp
from jf_wep.instrument import Instrument
from jf_wep.utils import loadConfig, mergeParams


class WfAlgorithm(ABC):
    """Base class for wavefront estimation algorithms

    Parameters
    ----------
    configFile : str, optional
        Path to file specifying values for the other parameters. If the
        path starts with "policy/", it will look in the policy directory.
        Any explicitly passed parameters override values found in this file
    instConfig : Path or str or dict or Instrument, optional
        Instrument configuration. If a Path or string, it is assumed this
        points to a config file, which is used to configure the Instrument.
        If a dictionary, it is assumed to hold keywords for configuration.
        If an Instrument object, that object is just used.

    ...

    """

    def __init__(
        self,
        configFile: Union[Path, str, None] = None,
        instConfig: Union[Path, str, dict, Instrument, None] = None,
        **kwargs: Any,
    ) -> None:
        # Merge keyword arguments with defaults from configFile
        params = mergeParams(
            configFile,
            instConfig=instConfig,
            **kwargs,
        )

        # Configure the instrument
        self.configInstrument(params.pop("instConfig"))

        # Configure other parameters
        for key, value in params.items():
            setattr(self, key, value)

    def configInstrument(
        self, instConfig: Union[Instrument, Path, str, dict]
    ) -> None:
        """Configure the instrument.

        For details about this parameter, see the class docstring.
        """
        self._instrument = loadConfig(instConfig, Instrument)

    @property
    def instrument(self) -> Instrument:
        """Return the instrument object.

        For details about this parameter, see the class docstring.
        """
        return self._instrument

    @abstractmethod
    def estimateWf(
        self,
        I1: DonutStamp,
        I2: Optional[DonutStamp],
    ) -> np.ndarray:
        """Return the wavefront Zernike coefficients in meters.

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
            Zernike coefficients (for Noll indices >= 4) estimated from
            the image (or pair of images), in meters.
        """
        ...
