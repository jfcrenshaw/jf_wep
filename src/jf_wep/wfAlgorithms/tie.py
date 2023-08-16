"""Define the classes associated with the TIE solver."""
import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from jf_wep.donutImage import DonutImage
from jf_wep.instrument import Instrument
from jf_wep.utils import DefocalType, ImageMapper, loadConfig, mergeParams
from jf_wep.wfAlgorithms.wfAlgorithm import WfAlgorithm
from jf_wep.zernikeObject import ZernikeObject


class TIEAlgorithm(WfAlgorithm):
    """Wavefront estimation algorithm class for the TIE solver.

    Parameters
    ----------
    configFile : Path or str, optional
        Path to file specifying values for the other parameters. If the
        path starts with "policy/", it will look in the policy directory.
        Any explicitly passed parameters override values found in this file
        (the default is policy/wfAlgorithms/tie.yaml)
    instConfig : Path or str or dict or Instrument, optional
        Instrument configuration. If a Path or string, it is assumed this
        points to a config file, which is used to configure the Instrument.
        If a dictionary, it is assumed to hold keywords for configuration.
        If an Instrument object, that object is just used.
    jmax : int, optional
        Maximum Zernike Noll index for which to solve.
    solver : str, optional
        Method used to solve the TIE. If "exp", the TIE is solved via
        directly expanding the wavefront in a Zernike series. If "fft",
        the TIE is solved using fast Fourier transforms.
    opticalModel : str, optional
        The optical model used for image compensation. If "paraxial", the
        original algorithm from Rodier & Rodier (1993) is used, which is
        suitable for images near the optical axis on telescope with large
        focal ratios. If "onAxis", the modification for small focal ratios
        (i.e. fast optics) introduced by Xin (2015) is used. If "offAxis",
        an empirical compensation polynomial is used. This is suitable for
        fast telescopes, far from the optical axis.
    maxIter : int, optional
        The maximum number of iterations of the TIE loop.
    compSequence : iterable, optional
        An iterable that determines the maximum Noll index to compensate on
        each iteration of the TIE loop. For example, if compSequence = [4, 10],
        then on the first iteration, only Zk4 is used in image compensation and
        on iteration 2, Zk4-Zk10 are used. Once the end of the sequence has
        been reached, all Zernike coefficients are used during compensation.
    compGain : float, optional
        The gain used to update the Zernikes for image compensation.

    Raises
    ------
    ValueError
        Invalid solver name
    """

    def __init__(
        self,
        configFile: Union[Path, str, None] = "policy/wfAlgorithms/tie.yaml",
        instConfig: Union[Path, str, dict, Instrument, None] = None,
        jmax: Optional[int] = None,
        solver: Optional[str] = None,
        opticalModel: Optional[str] = None,
        maxIter: Optional[int] = None,
        compSequence: Optional[Iterable] = None,
        compGain: Optional[float] = None,
    ) -> None:
        self.config(
            configFile=configFile,
            instConfig=instConfig,
            jmax=jmax,
            solver=solver,
            opticalModel=opticalModel,
            maxIter=maxIter,
            compSequence=compSequence,
            compGain=compGain,
        )

    def config(  # type: ignore[override]
        self,
        configFile: Union[Path, str, None] = None,
        instConfig: Union[Path, str, dict, Instrument, None] = None,
        jmax: Optional[int] = None,
        solver: Optional[str] = None,
        opticalModel: Optional[str] = None,
        maxIter: Optional[int] = None,
        compSequence: Optional[Iterable] = None,
        compGain: Optional[float] = None,
    ) -> None:
        """Configure the TIE solver.

        For details on the parameters, see the class docstring.
        """
        # Merge keyword arguments with the default parameters
        params = mergeParams(
            configFile,
            instConfig=instConfig,
            jmax=jmax,
            solver=solver,
            opticalModel=opticalModel,
            maxIter=maxIter,
            compSequence=compSequence,
            compGain=compGain,
        )

        # Set the instrument
        instConfig = params["instConfig"]
        if instConfig is not None:
            self._instrument = loadConfig(instConfig, Instrument)

        # Create the Zernike object
        jmax = params["jmax"]
        if jmax is not None:
            self._zernikeObject = ZernikeObject(
                jmax=jmax,
                instConfig=self.instrument,
            )

        # Set the solver
        solver = params["solver"]
        if solver is not None:
            allowed_solvers = ["exp", "fft"]
            if solver not in allowed_solvers:
                raise ValueError(
                    f"Solver '{solver}' not supported. "
                    f"Please choose one of {str(allowed_solvers)[1:-1]}."
                )
            self._solver = solver

        # Create the compensator
        # Allow ImageMapper to handle parameter validation
        warnings.warn("ImageMapper not configured!")

        # Set maxIter
        maxIter = params["maxIter"]
        if maxIter is not None:
            if not isinstance(maxIter, int) or (
                isinstance(maxIter, float) and maxIter % 1 != 0
            ):
                raise TypeError("maxIter must be an integer.")
            if maxIter < 0:
                raise ValueError("maxIter must be non-negative.")
            self._maxIter = int(maxIter)

        # Set compSequence
        compSequence = params["compSequence"]
        if compSequence is not None:
            compSequence = np.array(compSequence, dtype=int)
            if compSequence.ndim != 1:
                raise ValueError("compSequence must be a 1D iterable.")
            self._compSequence = compSequence

        # Set compGain
        compGain = params["compGain"]
        if compGain is not None:
            compGain = float(compGain)
            if compGain <= 0:
                raise ValueError("compGain must be positive.")
            self._compGain = compGain

    @property
    def instrument(self) -> Instrument:
        """Return the instrument object.

        For details about this parameter, see the class docstring.
        """
        return self._instrument

    @property
    def zernikeObject(self) -> ZernikeObject:
        """Return the ZernikeObject."""
        return self._zernikeObject

    @property
    def jmax(self) -> int:
        """Return the maximum Zernike Noll index.

        For details about this parameter, see the class docstring.
        """
        return self.zernikeObject.jmax

    @property
    def solver(self) -> Union[str, None]:
        """Return the name of the TIE solver.

        For details about this parameter, see the class docstring.
        """
        return self._solver

    @property
    def opticalModel(self) -> Union[str, None]:
        """Return the optical model of the TIECompensator.

        For details about this parameter, see the class docstring.
        """
        return self.compensator.opticalModel

    @property
    def maxIter(self) -> int:
        """Return the maximum number of iterations in the TIE loop.

        For details about this parameter, see the class docstring.
        """
        return self._maxIter

    @property
    def compSequence(self) -> np.ndarray:
        """Return the compensation sequence for the TIE loop.

        For details about this parameter, see the class docstring.
        """
        return self._compSequence

    @property
    def compGain(self) -> float:
        """Return the compensation gain for the TIE loop.

        For details about this parameter, see the class docstring.
        """
        return self._compGain

    def _expSolve(self, I0: np.ndarray, dIdz: np.ndarray) -> np.ndarray:
        """Solve the TIE directly using a Zernike expansion.

        Parameters
        ----------
        I0
            The beam intensity at the exit pupil
        dIdz
            The gradient of the beam intensity across the exit pupil

        Returns
        -------
        np.ndarray
            Numpy array of the Zernike coefficients estimated from the image
            or pair of images, in nm.
        """
        raise NotImplementedError

    def _fftSolve(self, I0: np.ndarray, dIdz: np.ndarray) -> np.ndarray:
        """Solve the TIE using fast Fourier transforms.

        Parameters
        ----------
        I0
            The beam intensity at the exit pupil
        dIdz
            The gradient of the beam intensity across the exit pupil

        Returns
        -------
        np.ndarray
            Numpy array of the Zernike coefficients estimated from the image
            or pair of images, in nm.
        """
        raise NotImplementedError

    def estimateWf(
        self,
        I1: DonutImage,
        I2: DonutImage,  # type: ignore[override]
    ) -> np.ndarray:
        """Return the wavefront Zernike coefficients in nm.

        Parameters
        ----------
        I1 : DonutImage
            An image object containing an intra- or extra-focal donut image.
        I2 : DonutImage
            A second image, on the opposite side of focus from I1.

        Returns
        -------
        np.ndarray
            Numpy array of the Zernike coefficients estimated from the image
            or pair of images.
        """
        # Assign the intra and extrafocal images
        intra = I1 if I1.defocalType == DefocalType.Intra else I2
        extra = I1 if I1.defocalType == DefocalType.Extra else I2

        # Initialize Zernike arrays at zero
        zkComp = np.zeros(self.jmax - 4 + 1)  # Zernikes for compensation
        zkResid = np.zeros_like(zkComp)  # Residual Zernikes after compensation

        # Get the compensation sequence
        compSequence = iter(self.compSequence)

        # Loop through every iteration in the sequence
        for _ in range(self.jmax):
            # Determine the maximum Noll index to compensate for
            jmaxComp = next(compSequence, self.jmax)

            # Calculate zkComp for this iteration
            zkComp += self.compGain * zkResid
            zkComp[(jmaxComp - 3) :] = 0

            # Compensate images using the Zernikes
            intraPupil = self.imageMapper.imageToPupil(intra.image, intra.defocalType, zkComp)
            extraPupil = self.imageMapper.imageToPupil(extra.image, extra.defocalType, zkComp)

            warnings.warn("\nNEED TO CHECK FOR CAUSTICS.\n")
            warnings.warn("\nNOT CURRENTLY APPLYING PUPIL MASK.\n")

            # Approximate I0 = I(x, 0) and dI/dz = dI(x, z)/dz at z=0
            I0 = (intraComp + extraComp) / 2
            dIdz = (intraComp - extraComp) / (2 * self.instrument.pupilOffset)

            # Estimate the Zernikes
            if self.solver == "exp":
                zkResid = self._expSolve(I0, dIdz)
            elif self.solver == "fft":
                zkResid = self._fftSolve(I0, dIdz)

            # Update our best estimate of the Zernikes
            zkBest = zkComp + zkResid

            warnings.warn("\nCHECK FOR CONVERGENCE.\n")

        return zkBest
