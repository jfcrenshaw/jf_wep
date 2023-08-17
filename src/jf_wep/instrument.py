from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from jf_wep.utils.paramReaders import mergeParams


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
    ) -> None:
        self.config(
            configFile=configFile,
            name=name,
            diameter=diameter,
            obscuration=obscuration,
            focalLength=focalLength,
            defocalOffset=defocalOffset,
            pixelSize=pixelSize,
        )

    def config(
        self,
        configFile: Union[Path, str, None] = None,
        name: Optional[str] = None,
        diameter: Optional[float] = None,
        obscuration: Optional[float] = None,
        focalLength: Optional[float] = None,
        defocalOffset: Optional[float] = None,
        pixelSize: Optional[float] = None,
    ) -> None:
        """Configure the instrument.

        For details on the parameters, see the class docstring.
        """
        # Merge the keyword arguments with the default parameters
        params = mergeParams(
            configFile,
            name=name,
            diameter=diameter,
            obscuration=obscuration,
            focalLength=focalLength,
            defocalOffset=defocalOffset,
            pixelSize=pixelSize,
        )

        # Set the name
        name = params["name"]
        if name is not None:
            name = str(name)
            self._name = name

        # Set the diameter
        diameter = params["diameter"]
        if diameter is not None:
            diameter = float(diameter)
            if diameter <= 0:
                raise ValueError("diameter must be positive.")
            self._diameter = diameter

        # Set the fractional obscuration
        obscuration = params["obscuration"]
        if obscuration is not None:
            obscuration = float(obscuration)
            if obscuration < 0 or obscuration > 1:
                raise ValueError(
                    "The obscuration must be between 0 and 1 (inclusive)."
                )
            self._obscuration = obscuration

        # Set the focal length
        focalLength = params["focalLength"]
        if focalLength is not None:
            focalLength = float(focalLength)
            if focalLength <= 0:
                raise ValueError("focalLength must be positive.")
            self._focalLength = focalLength

        # Set the defocal offset
        defocalOffset = params["defocalOffset"]
        if defocalOffset is not None:
            defocalOffset = np.abs(float(defocalOffset))
            self._defocalOffset = defocalOffset

        # Set the pixel size
        pixelSize = params["pixelSize"]
        if pixelSize is not None:
            pixelSize = float(pixelSize)
            if pixelSize <= 0:
                raise ValueError("pixelSize must be positive.")
            self._pixelSize = pixelSize

    @property
    def name(self) -> str:
        """The name of the instrument."""
        return self._name

    @property
    def diameter(self) -> float:
        """The primary mirror diameter in meters."""
        return self._diameter

    @property
    def obscuration(self) -> float:
        """The fractional obscuration."""
        return self._obscuration

    @property
    def focalLength(self) -> float:
        """The focal length in meters."""
        return self._focalLength

    @property
    def defocalOffset(self) -> float:
        """The defocal offset in meters."""
        return self._defocalOffset  # type: ignore

    @property
    def pixelSize(self) -> float:
        """The pixel size in meters."""
        return self._pixelSize

    @property
    def radius(self) -> float:
        """The primary mirror radius in meters."""
        return self.diameter / 2

    @property
    def area(self) -> float:
        """The primary mirror area in square meters."""
        return np.pi * self.radius**2 * (1 - self.obscuration**2)

    @property
    def pupilOffset(self) -> float:
        """The pupil offset in meters."""
        return self.focalLength**2 / self.defocalOffset

    @property
    def pupilMag(self) -> float:
        """The magnification of the pupil onto the image plane.
        
        Note this does not take into account any intrinsic aberrations
        inherent in the optical system.
        """
        return self.defocalOffset / self.focalLength

    @property
    def donutDiameter(self) -> float:
        """The expected donut diameter in pixels.
        
        Note this does not take into account any intrinsic aberrations
        inherent in the optical system.
        """
        return self.pupilMag * self.diameter / self.pixelSize

    @property
    def donutRadius(self) -> float:
        """The expected donut radius in pixels.
        
        Note this does not take into account any intrinsic aberrations
        inherent in the optical system.
        """
        return self.donutDiameter / 2

    def createPupilGrid(self, nPixels: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create an (nPixel x nPixel) grid for the pupil.

        The coordinates of the grid are in normalized pupil coordinates.
        These coordinates are defined such that u^2 + v^2 = 1 is the edge
        of the pupil, and u^2 + v^2 = obscuration^2 is the edge of the
        central obscuration.

        Parameters
        ----------
        nPixels : int
            The number of pixels on a side.

        Returns
        -------
        np.ndarray
            The 2D u-grid
        np.ndarray
            The 2D v-grid
        """
        # Create a 1D array with the correct number of pixels
        grid = np.arange(nPixels, dtype=float)

        # Center the grid
        grid -= grid.mean()

        # Convert to pupil normalized coordinates
        grid /= self.donutRadius

        # Create u and v grids
        uGrid, vGrid = np.meshgrid(grid, grid)

        return uGrid, vGrid
    
    def createPupilMask(self, nPixels: int) -> np.ndarray:
        """Create an (nPixel x nPixel) mask for the pupil.

        The coordinates of the grid are in normalized pupil coordinates.
        These coordinates are defined such that u^2 + v^2 = 1 is the edge
        of the pupil, and u^2 + v^2 = obscuration^2 is the edge of the
        central obscuration.

        Parameters
        ----------
        nPixels : int
            The number of pixels on a side.

        Returns
        -------
        np.ndarray
            The 2D pupil mask
        """
        # Get the pupil grid
        uPupil, vPupil = self.createPupilGrid(nPixels)

        # Calculate distance from center
        rPupil = np.sqrt(uPupil**2 + vPupil**2)

        # Mask outside the pupil
        mask = (rPupil >= self.obscuration) & (rPupil <= 1)

        return mask
    
    def createMarkedPupilMask(self, nPixels: int) -> np.ndarray:
        """Create a pupil mask with the positive u and v axes marked.
        
        The positive u axis is marked with a circle, and the positive v axis
        is marked with a diamond. These marks are cutouts in the mask. This
        is helpful for tracking axis directions in different algorithms.

        Parameters
        ----------
        nPixels : int
            The number of pixels on a side.

        Returns
        -------
        np.ndarray
            The 2D marked pupil mask
        """
        # Get the pupil grid
        uPupil, vPupil = self.createPupilGrid(nPixels)

        # Get the regular pupil mask
        mask = self.createPupilMask(nPixels)

        # Determine center and radius of markers
        mCenter = (1 + self.obscuration) / 2
        mRadius = (1 - self.obscuration) / 4

        # Add circle marker to positive u axis
        mask = mask & (np.sqrt((uPupil - mCenter) ** 2 + vPupil**2) > mRadius)

        # Add diamond marker to positive v axis
        mask = mask & (np.abs(uPupil) + np.abs(vPupil - mCenter) > mRadius)

        return mask