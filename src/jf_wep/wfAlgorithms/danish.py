import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from jf_wep.donutStamp import DonutStamp
from jf_wep.imageMapper import ImageMapper
from jf_wep.instrument import Instrument
from jf_wep.utils import DefocalType
from jf_wep.wfAlgorithms.wfAlgorithm import WfAlgorithm
from jf_wep.zernikes import createZernikeBasis, createZernikeGradBasis


class DanishAlgorithm(WfAlgorithm):
    """Wavefront estimation algorithm class for Danish.

    Danish uses the Zernike coefficients to forward model a donut image,
    and optimizes the coefficients so that the forward model image best
    matches the true image.

    Parameters
    ----------
    configFile : Path or str, optional
        Path to file specifying values for the other parameters. If the
        path starts with "policy/", it will look in the policy directory.
        Any explicitly passed parameters override values found in this file
        (the default is policy/wfAlgorithms/danish.yaml)
    addIntrinsic : bool, optional
        Whether to explicitly add the intrinsic Zernike coefficients to
        those solved for by Danish. If False, the coefficients returned
        by Danish represent the full OPD. If True, the coefficients returned
        by Danish represent the wavefront deviation (i.e. OPD - intrinsic).
    """

    def __init__(
        self,
        configFile: Union[Path, str, None] = "policy/wfAlgorithms/tie.yaml",
        addIntrinsic: Optional[bool] = None,
    ) -> None:
        super().__init__(configFile=configFile, addIntrinsic=addIntrinsic)

    @property
    def addIntrinsic(self) -> bool:
        """Flag indicating whether intrinsic Zernikes are explicitly added.

        For details about this parameter, see the class docstring.
        """
        return self._addIntrinsic

    @addIntrinsic.setter
    def addIntrinsic(self, value: bool) -> None:
        """Set the addIntrinsic flag."""
        if not isinstance(value, bool):
            raise TypeError("addIntrinsic must be a bool.")
        self._addIntrinsic = value

    def estimateWf(
        self,
        I1: DonutStamp,
        I2: Optional[DonutStamp] = None,
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
            the images, in meters.
        """
        # Validate the inputs
        self._validateInputs(I1, I2, jmax, instrument)

        # Create an ImageMapper for forward modeling
        imageMapper = ImageMapper(
            configFile=None,
            instConfig=instrument,
            addIntrinsic=self.addIntrinsic,
        )

        # TODO: ADD SUPPORT FOR I2

        # Define a function to calculate the chi value
        # Chi = (data - model) / error
        def chi(params: np.ndarray) -> np.ndarray:
            # Unpack the parameters
            zkCoeff = params

            # Forward model a template of the image
            template = imageMapper.createImageTemplate(I1, zkCoeff)

            # Create an image model by renormalizing the template
            model = template.image * I1.image.sum() / template.image.sum()

            return np.ravel(I1.image - model) / 1

        return
