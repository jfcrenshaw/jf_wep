"""Base class for wavefront estimation algorithms.

Use this class to define the universal structure for all algorithm classes.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from jf_wep.donutStamp import DonutStamp
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

    ...

    """

    def __init__(
        self,
        configFile: Union[Path, str, None] = None,
        **kwargs: Any,
    ) -> None:
        # Merge keyword arguments with defaults from configFile
        params = mergeParams(
            configFile,
            **kwargs,
        )

        # Configure parameters
        for key, value in params.items():
            setattr(self, key, value)

    @staticmethod
    def _validateInputs(
        I1: DonutStamp,
        I2: Optional[DonutStamp],
        jmax: int = 28,
        instrument: Instrument = Instrument(),
    ) -> None:
        """Validate the inputs to estimateWf.

        Parameters
        ----------
        I1 : DonutStamp
            A stamp object containing an intra- or extra-focal donut image.
        I2 : DonutStamp, optional
            A second stamp, on the opposite side of focus from I1.
            (the default is None)
        jmax : int, optional
            The maximum Zernike Noll index to estimate.
            (the default is 28)
        instrument : Instrument, optional
            The Instrument object associated with the DonutStamps.
            (the default is the default Instrument)
        """
        # Validate I1
        if not isinstance(I1, DonutStamp):
            raise TypeError("I1 must be a DonutStamp")
        if len(I1.image.shape) != 2 or not np.allclose(
            *I1.image.shape  # type: ignore
        ):
            raise ValueError("I1.image must be square.")

        # Validate I2 if provided
        if I2 is not None:
            if not isinstance(I2, DonutStamp):
                raise TypeError("I2 must be a DonutStamp")
            if len(I2.image.shape) != 2 or not np.allclose(
                *I2.image.shape  # type: ignore
            ):
                raise ValueError("I2.image must be square.")
            if I2.defocalType == I1.defocalType:
                raise ValueError(
                    "I1 and I2 must be on opposite sides of focus."
                )

        # Validate jmax
        if not isinstance(jmax, int):
            raise TypeError("jmax must be an integer.")
        if jmax < 4:
            raise ValueError("jmax must be greater than or equal to 4.")

        # Validate the instrument
        if not isinstance(instrument, Instrument):
            raise TypeError("instrument must be an Instrument.")

    @abstractmethod
    def estimateWf(
        self,
        I1: DonutStamp,
        I2: Optional[DonutStamp],
        jmax: int = 28,
        instrument: Instrument = Instrument(),
    ) -> np.ndarray:
        """Return the wavefront Zernike coefficients in meters.

        Parameters
        ----------
        I1 : DonutStamp
            A stamp object containing an intra- or extra-focal donut image.
        I2 : DonutStamp, optional
            A second stamp, on the opposite side of focus from I1.
            (the default is None)
        jmax : int, optional
            The maximum Zernike Noll index to estimate.
            (the default is 28)
        instrument : Instrument, optional
            The Instrument object associated with the DonutStamps.
            (the default is the default Instrument)

        Returns
        -------
        np.ndarray
            Zernike coefficients (for Noll indices >= 4) estimated from
            the image (or pair of images), in meters.
        """
        ...
