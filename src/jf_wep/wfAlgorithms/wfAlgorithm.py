"""Base class for wavefront estimation algorithms.

Use this class to define the universal structure for all algorithm classes.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jf_wep.donutImage import DonutImage
from jf_wep.instrument import Instrument
from jf_wep.utils import mergeParams


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
    zkConfig : Path or str or dict or ZernikeObject, optional
        ZernikeObject configuration. If a Path or string, it is assumed this
        points to a config file, which is used to configure the ZernikeObject.
        If a dictionary, it is assumed to hold keywords for configuration.
        If a ZernikeObject, that object is just used.

    ...

    """

    def __init__(
        self,
        configFile: Union[Path, str, None] = None,
        instConfig: Union[Instrument, Path, str, dict, None] = None,
        **kwargs: Any,
    ) -> None:
        self.config(
            configFile=configFile,
            instConfig=instConfig,
            zkConfig=zkConfig,
            **kwargs,
        )

    @abstractmethod
    def config(
        self,
        configFile: Union[Path, str, None] = None,
        instConfig: Union[Path, str, dict, Instrument, None] = None,
        **kwargs: Any,
    ) -> None:
        """Configure the algorithm.

        For details on the parameters, see the class docstring.
        """
        # Merge keyword arguments with the default parameters
        params = mergeParams(  # noqa: F841
            configFile,
            instConfig=instConfig,
            **kwargs,
        )

        # Rest of configuration
        ...

    @abstractmethod
    def estimateWf(
        self,
        I1: DonutImage,
        I2: Optional[DonutImage],
    ) -> np.ndarray:
        """Return the wavefront Zernike coefficients in nm.

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
        """
        ...
