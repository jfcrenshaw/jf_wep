from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union

import batoid
import numpy as np

from jf_wep.utils import BandLabel, DefocalType, mergeParams


class Instrument:
    """Object with relevant geometry of the primary mirror and focal plane.

    Parameters
    ----------
    configFile: Path or str, optional
        Path to file specifying values for the other parameters. If the
        path starts with "policy/", it will look in the policy directory.
        Any explicitly passed parameters override values found in this file
        (the default is policy/instruments/LsstCam.yaml)
    name: str, optional
        The name of the instrument.
    diameter : float, optional
        The diameter of the primary mirror in meters.
    obscuration : float, optional
        The fractional obscuration of the primary mirror.
    focalLength : float, optional
        The effective focal length in meters.
    defocalOffset : float, optional
        The defocal offset of the images in meters.
    pixelSize : float, optional
        The pixel size in meters.
    wavelength : float or dict, optional
        The effective wavelength of the instrument in meters. Can be a float, or
        a dictionary that corresponds to different bands. The keys in this
        dictionary are expected to correspond to the strings specified in the
        BandLabel enum in jf_wep.utils.enums.
    batoidModelName : str, optional
        The name used to load the Batoid model, via
        batoid.Optic.fromYaml(batoidModelName). If the string contains "{band}",
        then it is assumed there are different Batoid models for different
        photometric bands, and the names of these bands will be filled in at
        runtime using the strings specified in the BandLabel enum in
        jf_wep.utils.enums.
    """

    def __init__(
        self,
        configFile: Union[Path, str, None] = "policy/instruments/LsstCam.yaml",
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        obscuration: Optional[float] = None,
        focalLength: Optional[float] = None,
        defocalOffset: Optional[float] = None,
        pixelSize: Optional[float] = None,
        wavelength: Union[float, dict, None] = None,
        batoidModelName: Optional[str] = None,
    ) -> None:
        # Merge keyword arguments with defaults from configFile
        params = mergeParams(
            configFile,
            name=name,
            diameter=diameter,
            obscuration=obscuration,
            focalLength=focalLength,
            defocalOffset=defocalOffset,
            pixelSize=pixelSize,
            wavelength=wavelength,
            batoidModelName=batoidModelName,
        )

        # Set each parameter
        for key, value in params.items():
            setattr(self, key, value)

    @property
    def name(self) -> str:
        """The name of the instrument."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def diameter(self) -> float:
        """The primary mirror diameter in meters."""
        return self._diameter

    @diameter.setter
    def diameter(self, value: float) -> None:
        """Set the diameter."""
        value = float(value)
        if value <= 0:
            raise ValueError("diameter must be positive.")
        self._diameter = value

    @property
    def radius(self) -> float:
        """The primary mirror radius in meters."""
        return self.diameter / 2

    @property
    def area(self) -> float:
        """The primary mirror area in square meters."""
        return np.pi * self.radius**2 * (1 - self.obscuration**2)

    @property
    def obscuration(self) -> float:
        """The fractional obscuration."""
        return self._obscuration

    @obscuration.setter
    def obscuration(self, value: float) -> None:
        value = float(value)
        if value < 0 or value > 1:
            raise ValueError(
                "The obscuration must be between 0 and 1 (inclusive)."
            )
        self._obscuration = value

    @property
    def focalLength(self) -> float:
        """The focal length in meters."""
        return self._focalLength

    @focalLength.setter
    def focalLength(self, value: float) -> None:
        """Set the focal length."""
        value = float(value)
        if value <= 0:
            raise ValueError("focalLength must be positive.")
        self._focalLength = value

    @property
    def focalRatio(self) -> float:
        """The f-number."""
        return self.focalLength / self.diameter

    @property
    def defocalOffset(self) -> float:
        """The defocal offset in meters."""
        return self._defocalOffset  # type: ignore

    @defocalOffset.setter
    def defocalOffset(self, value: float) -> None:
        value = np.abs(float(value))
        self._defocalOffset = value

    @property
    def pupilOffset(self) -> float:
        """The pupil offset in meters."""
        return self.focalLength**2 / self.defocalOffset

    @property
    def pixelSize(self) -> float:
        """The pixel size in meters."""
        return self._pixelSize

    @pixelSize.setter
    def pixelSize(self, value: float) -> None:
        """Set the pixel size."""
        value = float(value)
        if value <= 0:
            raise ValueError("pixelSize must be positive.")
        self._pixelSize = value

    @property
    def donutRadius(self) -> float:
        """The expected donut radius in pixels."""
        rMeters = self.defocalOffset / np.sqrt(4 * self.focalRatio**2 - 1)
        rPixels = rMeters / self.pixelSize
        return rPixels

    @property
    def donutDiameter(self) -> float:
        """The expected donut diameter in pixels."""
        return 2 * self.donutRadius

    def createPupilGrid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create a grid for the pupil.

        The coordinates of the grid are in normalized pupil coordinates.
        These coordinates are defined such that u^2 + v^2 = 1 is the outer
        edge of the pupil, and u^2 + v^2 = obscuration^2 is the inner edge.

        The number of pixels is chosen to match the resolution of the image.

        Returns
        -------
        np.ndarray
            The 2D u-grid on the pupil plane
        np.ndarray
            The 2D v-grid on the pupil plane
        """
        # Set the resolution equal to the resolution of the image
        nPixels = np.ceil(self.donutDiameter).astype(int)

        # Create a 1D array with the correct number of pixels
        grid = np.linspace(-1.05, 1.05, nPixels)

        # Create u and v grids
        uPupil, vPupil = np.meshgrid(grid, grid)

        return uPupil, vPupil

    def createImageGrid(self, nPixels: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create an (nPixel x nPixel) grid for the image.

        The coordinates of the grid are in normalized image coordinates.
        These coordinates are defined such that u^2 + v^2 = 1 is the outer
        edge of the unaberrated donut, and u^2 + v^2 = obscuration^2 is the
        inner edge.

        Parameters
        ----------
        nPixels : int
            The number of pixels on a side.

        Returns
        -------
        np.ndarray
            The 2D u-grid on the image plane
        np.ndarray
            The 2D v-grid on the image plane
        """
        # Create a 1D array with the correct number of pixels
        grid = np.arange(nPixels, dtype=float)

        # Center the grid
        grid -= grid.mean()

        # Convert to pupil normalized coordinates
        grid /= self.donutRadius

        # Create u and v grids
        uImage, vImage = np.meshgrid(grid, grid)

        return uImage, vImage

    @property
    def wavelength(self) -> Union[float, dict]:
        """Return the effective wavelength(s) in meters."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: Union[float, dict]) -> None:
        if not isinstance(value, float) and not isinstance(value, dict):
            raise TypeError("wavelength must be a float or a dictionary.")
        if isinstance(value, dict):
            # Convert the wavelength dictionary to use BandLabels and floats
            value = {BandLabel(key): float(val) for key, val in value.items()}
        self._wavelength = value

        # Also clear the caches for the functions that use this value
        self.getIntrinsicZernikes.cache_clear()
        self.getOffAxisCoeff.cache_clear()

    @property
    def batoidModelName(self) -> Union[str, None]:
        return self._batoidModelName

    @batoidModelName.setter
    def batoidModelName(self, value: Optional[str]) -> None:
        if not isinstance(value, str) and value is not None:
            raise ValueError("batoidModelName must be a string, or None.")
        self._batoidModelName = value

        # Also clear the caches for the functions that use this value
        self.getBatoidModel.cache_clear()
        self.getIntrinsicZernikes.cache_clear()
        self.getOffAxisCoeff.cache_clear()

    @lru_cache(10)
    def getBatoidModel(
        self, band: Union[BandLabel, str] = BandLabel.REF
    ) -> batoid.Optic:
        """Return the batoid optical model for the instrument and the requested band.

        Parameters
        ----------
        band : BandLabel or str, optional
            The BandLabel Enum or corresponding string, specifying which batoid
            model to load. Only relevant if self.batoidModelName contains "{band}".
            (the default is BandLabel.REF)
        """
        if self.batoidModelName is None:
            return None
        
        # Get the band string from the Enum
        band = BandLabel(band).value

        # Fill any occurrence of "{band}" with the band string
        batoidModelName = self.batoidModelName.format(band=band)

        # Load the Batoid model
        return batoid.Optic.fromYaml(batoidModelName)

    @lru_cache(100)
    def getIntrinsicZernikes(
        self,
        xAngle: float,
        yAngle: float,
        band: Union[BandLabel, str] = BandLabel.REF,
        jmax: int = 66,
    ) -> np.ndarray:
        """Return the intrinsic Zernikes associated with the optical design.

        Parameters
        ----------
        xAngle : float
            The x-component of the field angle in degrees.
        yAngle : float
            The y-component of the field angle in degrees.
        band : BandLabel or str, optional
            The BandLabel Enum or corresponding string, specifying which batoid
            model to load. Only relevant if self.batoidModelName contains "{band}".
            (the default is BandLabel.REF)
        jmax : int, optional
            The maximum Noll index of the intrinsic Zernikes.
            (the default is 66)

        Returns
        -------
        np.ndarray
            The Zernike coefficients in meters, for Noll indices >= 4
        """
        # Get the band enum
        band = BandLabel(band)

        # Get the batoid model
        batoidModel = self.getBatoidModel(band)

        # If there is no batoid model, just return zeros
        if batoidModel is None:
            return np.zeros(jmax - 3)

        # Get the wavelength
        if isinstance(self.wavelength, dict):
            wavelength = self.wavelength[band]
        else:
            wavelength = self.wavelength

        # Get the intrinsic Zernikes in wavelengths
        zkIntrinsic = batoid.zernike(
            batoidModel,
            *np.deg2rad([xAngle, yAngle]),
            wavelength,
            jmax=jmax,
            eps=batoidModel.pupilObscuration,
        )

        # Multiply by wavelength to get Zernikes in meters
        zkIntrinsic *= wavelength

        # Keep only Noll indices >= 4
        zkIntrinsic = zkIntrinsic[4:]

        return zkIntrinsic

    @lru_cache(100)
    def getOffAxisCoeff(
        self,
        xAngle: float,
        yAngle: float,
        defocalType: DefocalType,
        band: Union[BandLabel, str] = BandLabel.REF,
        jmax: int = 66,
    ) -> np.ndarray:
        """Return the Zernike coefficients associated with the off-axis model.

        Parameters
        ----------
        xAngle : float
            The x-component of the field angle in degrees.
        yAngle : float
            The y-component of the field angle in degrees.
        defocalType : DefocalType or str
            The DefocalType Enum or corresponding string, specifying which side
            of focus to model.
        band : BandLabel or str, optional
            The BandLabel Enum or corresponding string, specifying which batoid
            model to load. Only relevant if self.batoidModelName contains "{band}".
            (the default is BandLabel.REF)
        jmax : int, optional
            The maximum Noll index of the off-axis model Zernikes.
            (the default is 66)

        Returns
        -------
        np.ndarray
            The Zernike coefficients in meters, for Noll indices >= 4
        """
        # Get the band enum
        band = BandLabel(band)

        # Get the batoid model
        batoidModel = self.getBatoidModel(band)

        # If there is no batoid model, just return zeros
        if batoidModel is None:
            return np.zeros(jmax - 3)

        # Offset the focal plane
        defocalType = DefocalType(defocalType)
        defocalSign = +1 if defocalType == DefocalType.Extra else -1
        offset = defocalSign * self.defocalOffset
        batoidModel = batoidModel.withGloballyShiftedOptic("Detector", [0, 0, offset])

        # Get the wavelength
        if isinstance(self.wavelength, dict):
            wavelength = self.wavelength[band]
        else:
            wavelength = self.wavelength

        # Get the off-axis model Zernikes in wavelengths
        zkIntrinsic = batoid.zernikeTA(
            batoidModel,
            *np.deg2rad([xAngle, yAngle]),
            wavelength,
            jmax=jmax,
            eps=batoidModel.pupilObscuration,
        )

        # Multiply by wavelength to get Zernikes in meters
        zkIntrinsic *= wavelength

        # Keep only Noll indices >= 4
        zkIntrinsic = zkIntrinsic[4:]

        return zkIntrinsic
    
    @property
    def maskParams(self) -> dict:
        """Return the mask parameters."""
        # Get the parameters if they exist
        params = getattr(self, "_maskParams", None)

        if params is None:
            # If they don't exist, use the primary inner and outer radii
            params = {
                "pupilOuter": {
                    "thetaMin": 0,
                    "center": [0],
                    "radius": [self.radius],
                },
                "pupilInner": {
                    "thetaMin": 0,
                    "center": [0],
                    "radius": [self.obscuration],
                },
            }

        return params

    @maskParams.setter
    def maskParams(self, value: Optional[dict]) -> None:
        """Set the mask parameters."""
        if isinstance(value, dict) or value is None:
            self._maskParams = value
        else:
            raise TypeError("maskParams must be a dictionary or None.")
