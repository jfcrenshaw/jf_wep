"""Define the classes associated with the TIE solver."""
import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from jf_wep.donutStamp import DonutStamp
from jf_wep.imageMapper import ImageMapper
from jf_wep.instrument import Instrument
from jf_wep.utils import (
    DefocalType,
    createZernikeBasis,
    createZernikeGradBasis,
)
from jf_wep.wfAlgorithms.wfAlgorithm import WfAlgorithm


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
    addIntrinsic : bool, optional
        Whether to explicitly add the intrinsic Zernike coefficients solved
        for by the TIE. If False, the coefficients returned by the TIE
        represent the full OPD. If True, the coefficients returned by the
        TIE represent the wavefront deviation (i.e. OPD - intrinsic). If
        the donuts provided to the TIE are at different field positions,
        it is best to set addIntrinsic=True.
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
    convergeTol : float, optional
        The mean absolute deviation, in meters, between the Zernike estimates
        of subsequent TIE iterations below which convergence is declared.
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
        addIntrinsic: Optional[bool] = None,
        solver: Optional[str] = None,
        jmax: Optional[int] = None,
        maxIter: Optional[int] = None,
        compSequence: Optional[Iterable] = None,
        compGain: Optional[float] = None,
        saveHistory: Optional[bool] = None,
    ) -> None:
        super().__init__(
            configFile=configFile,
            instConfig=instConfig,
            addIntrinsic=addIntrinsic,
            solver=solver,
            jmax=jmax,
            maxIter=maxIter,
            compSequence=compSequence,
            compGain=compGain,
            saveHistory=saveHistory,
        )

        # Instantiate an empty history
        self._history = {}  # type: ignore

    @property
    def addIntrinsic(self) -> bool:
        """Flag indicating whether intrinsic Zernikes are explicitly added.

        For details about this parameter, see the class docstring.
        """
        return self.imageMapper.addIntrinsic

    @addIntrinsic.setter
    def addIntrinsic(self, value: bool) -> None:
        """Set the addIntrinsic flag."""
        self._imageMapper = ImageMapper(
            configFile=None,
            instConfig=self.instrument,
            addIntrinsic=value,
        )

    @property
    def imageMapper(self) -> ImageMapper:
        """Return the ImageMapper."""
        return self._imageMapper

    @property
    def solver(self) -> Union[str, None]:
        """Return the name of the TIE solver.

        For details about this parameter, see the class docstring.
        """
        return self._solver

    @solver.setter
    def solver(self, value: str) -> None:
        """Set the solver."""
        allowed_solvers = ["exp", "fft"]
        if value not in allowed_solvers:
            raise ValueError(
                f"Solver '{value}' not supported. "
                f"Please choose one of {str(allowed_solvers)[1:-1]}."
            )
        self._solver = value

    @property
    def jmax(self) -> int:
        """Return the maximum Zernike Noll index.

        For details about this parameter, see the class docstring.
        """
        return self._jmax

    @jmax.setter
    def jmax(self, value: int) -> None:
        """Set jmax."""
        if not isinstance(value, int) and not (
            isinstance(value, float) and int(value) == value
        ):
            raise TypeError("jmax must be an integer")
        if value < 4:
            raise ValueError("jmax must be >= 4.")
        self._jmax = int(value)

    @property
    def maxIter(self) -> int:
        """Return the maximum number of iterations in the TIE loop.

        For details about this parameter, see the class docstring.
        """
        return self._maxIter

    @maxIter.setter
    def maxIter(self, value: int) -> None:
        """Set maxIter."""
        if not isinstance(value, int) or (
            isinstance(value, float) and value % 1 != 0
        ):
            raise TypeError("maxIter must be an integer.")
        if value < 0:
            raise ValueError("maxIter must be non-negative.")
        self._maxIter = int(value)

    @property
    def compSequence(self) -> np.ndarray:
        """Return the compensation sequence for the TIE loop.

        For details about this parameter, see the class docstring.
        """
        return self._compSequence

    @compSequence.setter
    def compSequence(self, value: Iterable) -> None:
        """Set compSequence."""
        value = np.array(value, dtype=int)
        if value.ndim != 1:
            raise ValueError("compSequence must be a 1D iterable.")
        self._compSequence = value

    @property
    def compGain(self) -> float:
        """Return the compensation gain for the TIE loop.

        For details about this parameter, see the class docstring.
        """
        return self._compGain

    @compGain.setter
    def compGain(self, value: float) -> None:
        """Set compGain."""
        value = float(value)
        if value <= 0:
            raise ValueError("compGain must be positive.")
        self._compGain = value

    @property
    def convergeTol(self) -> float:
        """The mean absolute deviation, in meters, between the Zernike estimates
        of subsequent TIE iterations below which convergence is declared.
        """
        return self._convergeTol

    @convergeTol.setter
    def convergeTol(self, value: float) -> None:
        """Set the convergence tolerance."""
        value = float(value)
        if value <= 0:
            raise ValueError("convergeTol must be positive.")
        self._convergeTol = value

    @property
    def saveHistory(self) -> bool:
        """Return the bool indicating whether the algorithm history is saved."""
        return self._saveHistory

    @saveHistory.setter
    def saveHistory(self, value: bool) -> None:
        """Set saveHistory."""
        if not isinstance(value, bool):
            raise TypeError("saveHistory must be a bool.")
        self._saveHistory = value

        # If we are turning history-saving off, delete any old history
        # This is to avoid confusion
        if value is False:
            self._history = {}

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

        return self._history

    def _validateStamps(
        self,
        I1: DonutStamp,
        I2: DonutStamp,  # type: ignore[override]
    ) -> None:
        """Validate the DonutStamps for TIE estimation.

        Parameters
        ----------
        I1 : DonutStamp
            A stamp object containing an intra- or extra-focal donut image.
        I2 : DonutStamp
            A second stamp, on the opposite side of focus from I1.
        """
        # Make sure both stamps have been provided
        if I1 is None or I2 is None:
            raise ValueError(
                "TIEAlgorithm requires a pair of intrafocal and extrafocal "
                "donuts to estimate Zernikes. Please provide both I1 and I2."
            )

        # Check that I1 and I2 are on opposite sides of focus
        if I2.defocalType == I1.defocalType:
            raise ValueError("I1 and I2 must be on opposite sides of focus.")

        # Check that the images are square
        if len(I1.image.shape) != 2 or not np.allclose(*I1.image.shape):  # type: ignore
            raise ValueError("I1 must be a square image.")
        if len(I2.image.shape) != 2 or not np.allclose(*I2.image.shape):  # type: ignore
            raise ValueError("I1 must be a square image.")

        # Warn that addIntrinsic==False is a bad idea when the donuts are at
        # different field angles
        if (
            not np.allclose(I1.fieldAngle, I2.fieldAngle)
            and not self.addIntrinsic
        ):
            warnings.warn(
                "When estimating Zernikes from a pair of donuts at different "
                "field angles, it is wise to set self.addIntrinsic=True, "
                "since telescopes have different intrinsic aberrations at "
                "different field angles."
            )

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
        # Validate the DonutStamps
        self._validateStamps(I1, I2)

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
                # (1) The mean absolute difference with the previous iteration
                # must be below self.convergeTol
                # (2) We must be compensating all the Zernikes
                newBest = zkComp + zkResid
                diffZk = zkBest - newBest
                if (
                    np.mean(np.abs(diffZk)) < self.convergeTol
                    and jmaxComp == self.jmax
                ):
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
