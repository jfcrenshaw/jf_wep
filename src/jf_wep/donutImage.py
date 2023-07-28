"""Class to hold donut images along with metadata."""
from typing import Union

import numpy as np

from jf_wep.utils import DefocalType, FilterLabel


class DonutImage:
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
        Can be specified using a DefocalType Enum,
        or the corresponding string.
    filterLabel : FilterLabel or str
        Photometric filter for the exposure.
        Can be specified using a FilterLabel Enum,
        or the corresponding string.
    blendOffsets : np.ndarray or tuple or list, optional
        Positions of blended donuts relative to location of center donut,
        in pixels. Must be provided in the format [dxList, dyList].
        The lengths of dxList and dyList must be the same.
        (the default is None)

    Raises
    ------
    TypeError
        image is not a numpy array
    ValueError
        image array is not square
    ValueError
        fieldAngle cannot be cast to float array or has wrong shape
    TypeError
        defocalType is not a DefocalType Enum or a string
    TypeError
        filterLabel is not a FilterLabel Enum or a string
    """

    def __init__(
        self,
        image: np.ndarray,
        fieldAngle: Union[np.ndarray, tuple, list],
        defocalType: Union[DefocalType, str],
        filterLabel: Union[FilterLabel, str],
        blendOffsets: Union[np.ndarray, tuple, list, None],
    ) -> None:
        # Set the image
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy array.")
        if len(image.shape) != 2 or image.shape[0] != image.shape[1]:
            raise ValueError("The image array must be square.")
        self._image = image

        # Set the field angle
        fieldAngle = np.array(fieldAngle, dtype=float).squeeze()
        if fieldAngle.shape != (2,):
            raise ValueError("Field angle must have shape (2,).")
        self._fieldAngle = fieldAngle

        # Set the defocal type
        if isinstance(defocalType, str):
            self._defocalType = DefocalType(defocalType)
        elif isinstance(defocalType, DefocalType):
            self._defocalType = defocalType
        else:
            raise TypeError(
                "defocalType must be a DefocalType Enum, or "
                "one of the strings 'intra' or 'extra'."
            )

        # Set the filter label
        if isinstance(filterLabel, str):
            self._filterLabel = FilterLabel(defocalType)
        elif isinstance(filterLabel, FilterLabel):
            self._filterLabel = filterLabel
        else:
            raise TypeError(
                "filterLabel must be a FilterLabel Enum, or "
                "one of the corresponding strings."
            )

        # Set the blend offsets
        if blendOffsets is None:
            self._blendOffsets = blendOffsets
        else:
            blendOffsets = np.array(blendOffsets, dtype=float)
            if blendOffsets.shape[0] != 2 or len(blendOffsets.shape) != 2:
                raise ValueError(
                    "blendOffsets must have shape (2, N), "
                    "where N is the number of blends you wish to mask."
                )
            self._blendOffsets = blendOffsets

    @property
    def image(self) -> np.ndarray:
        """Return the donut image array.

        For details about this parameter, see the class docstring.
        """
        return self._image

    @property
    def fieldAngle(self) -> np.ndarray:
        """Return the field angle of the donut.

        For details about this parameter, see the class docstring.
        """
        return self._fieldAngle

    @property
    def defocalType(self) -> DefocalType:
        """Return the DefocalType Enum of the image.

        For details about this parameter, see the class docstring.
        """
        return self._defocalType

    @property
    def filterLabel(self) -> FilterLabel:
        """Return the FilterLabel Enum of the image.

        For details about this parameter, see the class docstring.
        """
        return self._filterLabel

    @property
    def blendOffsets(self) -> Union[np.ndarray, None]:
        """Return the blend offsets array for the image.

        For details about this parameter, see the class docstring.
        """
        return self._blendOffsets
