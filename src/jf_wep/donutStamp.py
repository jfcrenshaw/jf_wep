"""Class to hold donut images along with metadata."""
from copy import deepcopy
from typing import Union, Optional

import numpy as np
from typing_extensions import Self

from jf_wep.utils import DefocalType, BandLabel, PlaneType


class DonutStamp:
    """Class to hold a donut image along with metadata.

    Parameters
    ----------
    image : np.ndarray
        The square numpy array containing the donut image.
    fieldAngle : np.ndarray or tuple or list
        The field angle of the donut, in degrees. The field angle
        is the angle to the source, measured from the optical axis.
    defocalType : DefocalType or str
        Whether the image is intra- or extra-focal.
        Can be specified using a DefocalType Enum or the corresponding string.
    planeType : PlaneType or str
        Whether the image is on the image plane or the pupil plane.
        Can be specified using a PlaneType Enum, or the corresponding string.
    bandLabel : BandLabel or str
        Photometric filter for the exposure.
        Can be specified using a BandLabel Enum or the corresponding string.
    blendOffsets : np.ndarray or tuple or list, optional
        Positions of blended donuts relative to location of center donut,
        in pixels. Must be provided in the format [dxList, dyList].
        The lengths of dxList and dyList must be the same.
        (the default is an empty array, i.e. no blends)
    mask : np.ndarray, optional
        The mask for the image. Mask creation is meant to be handled by the
        ImageMapper class.
    """

    def __init__(
        self,
        image: np.ndarray,
        fieldAngle: Union[np.ndarray, tuple, list],
        defocalType: Union[DefocalType, str],
        planeType: Union[PlaneType, str] = PlaneType.Image,
        bandLabel: Union[BandLabel, str] = BandLabel.REF,
        blendOffsets: Union[np.ndarray, tuple, list] = np.zeros((2, 0)),
        mask: Optional[np.ndarray] = None,
    ) -> None:
        self.image = image
        self.fieldAngle = fieldAngle  # type: ignore
        self.defocalType = defocalType  # type: ignore
        self.planeType = planeType  # type: ignore
        self.bandLabel = bandLabel  # type: ignore
        self.blendOffsets = blendOffsets  # type: ignore
        self.mask = mask

    @property
    def image(self) -> np.ndarray:
        """Return the donut image array.

        For details about this parameter, see the class docstring.
        """
        return self._image

    @image.setter
    def image(self, value) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("image must be a numpy array.")
        if len(value.shape) != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("The image array must be square.")
        self._image = value

    @property
    def fieldAngle(self) -> np.ndarray:
        """Return the field angle of the donut in degrees.

        For details about this parameter, see the class docstring.
        """
        return self._fieldAngle

    @fieldAngle.setter
    def fieldAngle(self, value: Union[np.ndarray, tuple, list]) -> None:
        value = np.array(value, dtype=float).squeeze()
        if value.shape != (2,):
            raise ValueError("Field angle must have shape (2,).")
        self._fieldAngle = value

    @property
    def defocalType(self) -> DefocalType:
        """Return the DefocalType Enum of the stamp.

        For details about this parameter, see the class docstring.
        """
        return self._defocalType

    @defocalType.setter
    def defocalType(self, value: Union[DefocalType, str]) -> None:
        if isinstance(value, str) or isinstance(value, DefocalType):
            self._defocalType = DefocalType(value)
        else:
            raise TypeError(
                "defocalType must be a DefocalType Enum, "
                "or one of the corresponding strings."
            )

    @property
    def planeType(self) -> PlaneType:
        """Return the PlaneType Enum of the stamp.

        For details about this parameter, see the class docstring.
        """
        return self._planeType

    @planeType.setter
    def planeType(self, value: Union[PlaneType, str]) -> None:
        if isinstance(value, str) or isinstance(value, PlaneType):
            self._planeType = PlaneType(value)
        else:
            raise TypeError(
                "planeType must be a PlaneType Enum, "
                "or one of the corresponding strings."
            )

    @property
    def bandLabel(self) -> BandLabel:
        """Return the BandLabel Enum of the image.

        For details about this parameter, see the class docstring.
        """
        return self._bandLabel

    @bandLabel.setter
    def bandLabel(self, value: Union[BandLabel, str]) -> None:
        if isinstance(value, str) or isinstance(value, BandLabel):
            self._bandLabel = BandLabel(value)
        else:
            raise TypeError(
                "bandLabel must be a BandLabel Enum, "
                "or one of the corresponding strings."
            )

    @property
    def blendOffsets(self) -> Union[np.ndarray, None]:
        """Return the blend offsets array for the image.

        For details about this parameter, see the class docstring.
        """
        return self._blendOffsets

    @blendOffsets.setter
    def blendOffsets(self, value: Union[np.ndarray, tuple, list]) -> None:
        value = np.array(value, dtype=float)
        if value.shape[0] != 2 or len(value.shape) != 2:
            raise ValueError(
                "blendOffsets must have shape (2, N), "
                "where N is the number of blends you wish to mask."
            )
        self._blendOffsets = value

    @property
    def mask(self) -> Union[np.ndarray, None]:
        """Return the image mask."""
        return self._mask

    @mask.setter
    def mask(self, value: Optional[np.ndarray]) -> None:
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("mask must be an array, or None.")
        elif isinstance(value, np.ndarray) and value.shape != self.image.shape:
            raise ValueError("mask must have the same shape as self.image.")
        self._mask = value

    def copy(self) -> Self:
        """Return a copy of the DonutImage object.

        Returns
        -------
        DonutImage
            A deep copy of self.
        """
        return deepcopy(self)
