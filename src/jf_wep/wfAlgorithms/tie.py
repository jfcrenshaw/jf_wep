"""Define the classes associated with the TIE solver."""
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from jf_wep.donutStamp import DonutStamp
from jf_wep.instrument import Instrument
from jf_wep.utils.enums import DefocalType
from jf_wep.utils.imageMapper import ImageMapper
from jf_wep.utils.paramReaders import loadConfig, mergeParams
from jf_wep.utils.zernikes import createZernikeBasis, createZernikeGradBasis
from jf_wep.wfAlgorithms.wfAlgorithm import WfAlgorithm

import warnings


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
    opticalModel : str, optional
        The optical model used for image compensation. If "paraxial", the
        original algorithm from Rodier & Rodier (1993) is used, which is
        suitable for images near the optical axis on telescope with large
        focal ratios. If "onAxis", the modification for small focal ratios
        (i.e. fast optics) introduced by Xin (2015) is used. If "offAxis",
        an empirical compensation polynomial is used. This is suitable for
        fast telescopes, far from the optical axis.
    solver : str, optional
        Method used to solve the TIE. If "exp", the TIE is solved via
        directly expanding the wavefront in a Zernike series. If "fft",
        the TIE is solved using fast Fourier transforms.
    jmax : int, optional
        Maximum Zernike Noll index for which to solve.
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
    saveHistory : bool, optional
        Whether to save the algorithm history in the self.history attribute.
        If True, then self.history contains information about the most recent
        time the algorithm was run.

    Raises
    ------
    ValueError
        Invalid solver name
    """

    def __init__(
        self,
        configFile: Union[Path, str, None] = "policy/wfAlgorithms/tie.yaml",
        instConfig: Union[Path, str, dict, Instrument, None] = None,
        opticalModel: Optional[str] = None,
        solver: Optional[str] = None,
        jmax: Optional[int] = None,
        maxIter: Optional[int] = None,
        compSequence: Optional[Iterable] = None,
        compGain: Optional[float] = None,
        saveHistory: Optional[bool] = None,
    ) -> None:
        self.config(
            configFile=configFile,
            instConfig=instConfig,
            opticalModel=opticalModel,
            solver=solver,
            jmax=jmax,
            maxIter=maxIter,
            compSequence=compSequence,
            compGain=compGain,
            saveHistory=saveHistory,
        )

    def config(  # type: ignore[override]
        self,
        configFile: Union[Path, str, None] = None,
        instConfig: Union[Path, str, dict, Instrument, None] = None,
        opticalModel: Optional[str] = None,
        solver: Optional[str] = None,
        jmax: Optional[int] = None,
        maxIter: Optional[int] = None,
        compSequence: Optional[Iterable] = None,
        compGain: Optional[float] = None,
        saveHistory: Optional[bool] = None,
    ) -> None:
        """Configure the TIE solver.

        For details on the parameters, see the class docstring.
        """
        # Merge keyword arguments with the default parameters
        params = mergeParams(
            configFile,
            instConfig=instConfig,
            opticalModel=opticalModel,
            solver=solver,
            jmax=jmax,
            maxIter=maxIter,
            compSequence=compSequence,
            compGain=compGain,
            saveHistory=saveHistory,
        )

        # Set the instrument
        instConfig = params["instConfig"]
        if instConfig is not None:
            self._instrument = loadConfig(instConfig, Instrument)

        # Set the image mapper
        opticalModel = params["opticalModel"]
        if opticalModel is not None:
            self._imageMapper = ImageMapper(
                configFile=None,
                opticalModel=opticalModel,
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

        # Set jmax
        jmax = params["jmax"]
        if jmax is not None:
            if not isinstance(jmax, int) and not (
                isinstance(jmax, float) and int(jmax) == jmax
            ):
                raise TypeError("jmax must be an integer")
            if jmax < 4:
                raise ValueError("jmax must be >= 4.")
            self._jmax = int(jmax)

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

        # Set whether to save the algorithm history
        saveHistory = params["saveHistory"]
        if saveHistory is not None:
            if not isinstance(saveHistory, bool):
                raise TypeError("saveHistory must be a bool.")
            self._saveHistory = saveHistory

            # If we are turning history-saving off, delete any old history
            # This is to avoid confusion
            self._history = {}  # type: ignore

    @property
    def instrument(self) -> Instrument:
        """Return the instrument object.

        For details about this parameter, see the class docstring.
        """
        return self._instrument

    @property
    def imageMapper(self) -> ImageMapper:
        """Return the ImageMapper."""
        return self._imageMapper

    @property
    def opticalModel(self) -> str:
        """Return the optical model.

        For details about this parameter, see the class docstring.
        """
        return self.imageMapper.opticalModel

    @property
    def solver(self) -> Union[str, None]:
        """Return the name of the TIE solver.

        For details about this parameter, see the class docstring.
        """
        return self._solver

    @property
    def jmax(self) -> int:
        """Return the maximum Zernike Noll index.

        For details about this parameter, see the class docstring.
        """
        return self._jmax

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

    @property
    def history(self) -> dict:
        """The algorithm history.

        The history is a dictionary saving the intermediate products from
        each iteration of the TIE solver. The first iteration is saved as
        history[0].

        The entry for each iteration is itself a dictionary containing
        the following keys:
            - "intraComp" - the compensated intrafocal image
            - "extraComp" - the compensated extrafocal image
            - "I0" - the estimate of the beam intensity on the pupil
            - "dIdz" - estimate of z-derivative of intensity across the pupil
            - "zkComp" - the Zernikes used for image compensation
            - "zkResid" - the estimated residual Zernikes
            - "zkBest" - the best estimate of the Zernikes after this iteration
            - "converged" - flag indicating if Zernike estimation has converged
            - "caustic" - flag indicating if a caustic has been hit

        Note the units for all Zernikes are in meters, and the z-derivative
        in dIdz is also in meters.
        """
        if not self._saveHistory:
            warnings.warn(
                "saveHistory is False. If you want the history to be saved, "
                "run self.config(saveHistory=True)."
            )
            return {}

        # If the history exists, return it, otherwise return an empty dict
        return getattr(self, "_history", {})

    def _expSolve(self, I0: np.ndarray, dIdz: np.ndarray) -> np.ndarray:
        """Solve the TIE directly using a Zernike expansion.

        Parameters
        ----------
        I0
            The beam intensity at the exit pupil
        dIdz
            The z-derivative of the beam intensity across the exit pupil

        Returns
        -------
        np.ndarray
            Numpy array of the Zernike coefficients estimated from the image
            or pair of images, in nm.
        """
        # Get Zernike Bases
        zk = createZernikeBasis(self.jmax, self.instrument, I0.shape[0])
        dzkdu, dzkdv = createZernikeGradBasis(
            self.jmax, self.instrument, I0.shape[0]
        )

        # Calculate quantities for the linear equation
        b = -np.einsum("ab,jab->j", dIdz, zk, optimize=True)
        M = np.einsum("ab,jab,kab->jk", I0, dzkdu, dzkdu, optimize=True)
        M += np.einsum("ab,jab,kab->jk", I0, dzkdv, dzkdv, optimize=True)
        M /= self.instrument.radius**2

        # Invert to get Zernike coefficients in meters
        zkCoeff, *_ = np.linalg.lstsq(M, b, rcond=None)

        return zkCoeff

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
        I1: DonutStamp,
        I2: DonutStamp,  # type: ignore[override]
    ) -> np.ndarray:
        """Return the wavefront Zernike coefficients in meters.

        Parameters
        ----------
        I1 : DonutStamp
            A stamp object containing an intra- or extra-focal donut image.
        I2 : DonutStamp
            A second stamp, on the opposite side of focus from I1.

        Returns
        -------
        np.ndarray
            Zernike coefficients (for Noll indices >= 4) estimated from 
            the images, in meters.
        """
        # If I2 provided, check that I1 and I2 are on opposite sides of focus
        if I2.defocalType == I1.defocalType:
            raise ValueError("I1 and I2 must be on opposite sides of focus.")

        # Assign the intra and extrafocal images
        intra = I1 if I1.defocalType == DefocalType.Intra else I2
        extra = I1 if I1.defocalType == DefocalType.Extra else I2

        # Initialize Zernike arrays at zero
        zkComp = np.zeros(self.jmax - 4 + 1)  # Zernikes for compensation
        zkResid = np.zeros_like(zkComp)  # Residual Zernikes after compensation
        zkBest = np.zeros_like(zkComp)  # Current best Zernike estimate

        # Get the compensation sequence
        compSequence = iter(self.compSequence)

        # Set the caustic and converged flags to False
        caustic = False
        converged = False

        # Loop through every iteration in the sequence
        for i in range(self.maxIter):
            # Determine the maximum Noll index to compensate
            # Once the compensation sequence is exhausted, jmaxComp = jmax
            jmaxComp = next(compSequence, self.jmax)

            # Calculate zkComp for this iteration
            # The gain scales how much of previous residual we incorporate
            # Everything past jmaxComp is set to zero
            zkComp += self.compGain * zkResid
            zkComp[(jmaxComp - 3) :] = 0

            # Compensate images using the Zernikes
            intraComp = self.imageMapper.imageToPupil(intra, zkComp).image
            extraComp = self.imageMapper.imageToPupil(extra, zkComp).image

            # Check for caustics
            if (
                intraComp.max() <= 0
                or extraComp.max() <= 0
                or not np.isfinite(intraComp).all()
                or not np.isfinite(extraComp).all()
            ):
                caustic = True

                # Dummy NaNs for the missing objects
                I0 = np.full_like(intraComp, np.nan)
                dIdz = np.full_like(intraComp, np.nan)
                zkResid = np.nan * zkResid
                zkBest = np.nan * zkResid

            # If no caustic, proceed with Zernike estimation
            else:
                # Normalize the images
                intraComp /= intraComp.sum()
                extraComp /= extraComp.sum()

                # Approximate I0 = I(x, 0) and dI/dz = dI(x, z)/dz at z=0
                I0 = (intraComp + extraComp) / 2
                dIdz = (intraComp - extraComp) / (
                    2 * self.instrument.pupilOffset
                )

                # Estimate the Zernikes
                if self.solver == "exp":
                    zkResid = self._expSolve(I0, dIdz)
                elif self.solver == "fft":
                    zkResid = self._fftSolve(I0, dIdz)

                # Check for convergence
                newBest = zkComp + zkResid
                diffZk = zkBest - newBest
                if np.sum(np.abs(diffZk)) < 0:
                    converged = True

                # Set the new best estimate
                zkBest = newBest

            # Time to wrap up this iteration!
            # Should we save intermediate products in the algorithm history?
            if self._saveHistory:
                # Save the images and Zernikes from this iteration
                self._history[i] = {
                    "intraComp": intraComp.copy(),
                    "extraComp": extraComp.copy(),
                    "I0": I0.copy(),
                    "dIdz": dIdz.copy(),
                    "zkComp": zkComp.copy(),
                    "zkResid": zkResid.copy(),
                    "zkBest": zkBest.copy(),
                    "converged": converged,
                    "caustic": caustic,
                }

                # If we are using the FFT solver, save the inner loop as well
                if self.solver == "fft":
                    # TODO: Need to add inner loop here
                    self._history[i]["innerLoop"] = None

            # If we've hit a caustic or converged, we will stop early
            if caustic or converged:
                break

        return zkBest
