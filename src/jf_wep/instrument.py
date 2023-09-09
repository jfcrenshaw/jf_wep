from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from jf_wep.utils import mergeParams


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
        # Merge keyword arguments with defaults from configFile
        params = mergeParams(
            configFile,
            name=name,
            diameter=diameter,
            obscuration=obscuration,
            focalLength=focalLength,
            defocalOffset=defocalOffset,
            pixelSize=pixelSize,
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
        grid = np.linspace(-1, 1, nPixels)

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
