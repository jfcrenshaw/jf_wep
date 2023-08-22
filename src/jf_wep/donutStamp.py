"""Class to hold donut images along with metadata."""
from copy import deepcopy
from typing import Union

import numpy as np
from typing_extensions import Self

from jf_wep.utils import DefocalType, FilterLabel


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
        (the default is an empty array, i.e. no blends)
    """

    def __init__(
        self,
        image: np.ndarray,
        fieldAngle: Union[np.ndarray, tuple, list],
        defocalType: Union[DefocalType, str],
        filterLabel: Union[FilterLabel, str] = FilterLabel.REF,
        blendOffsets: Union[np.ndarray, tuple, list] = np.zeros((2, 0)),
    ) -> None:
        self.image = image
        self.fieldAngle = fieldAngle  # type: ignore
        self.defocalType = defocalType  # type: ignore
        self.filterLabel = filterLabel  # type: ignore
        self.blendOffsets = blendOffsets  # type: ignore

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
        """Return the DefocalType Enum of the image.

        For details about this parameter, see the class docstring.
        """
        return self._defocalType

    @defocalType.setter
    def defocalType(self, value: Union[DefocalType, str]) -> None:
        if isinstance(value, str):
            self._defocalType = DefocalType(value)
        elif isinstance(value, DefocalType):
            self._defocalType = value
        else:
            raise TypeError(
                "defocalType must be a DefocalType Enum, or "
                "one of the strings 'intra' or 'extra'."
            )

    @property
    def filterLabel(self) -> FilterLabel:
        """Return the FilterLabel Enum of the image.

        For details about this parameter, see the class docstring.
        """
        return self._filterLabel

    @filterLabel.setter
    def filterLabel(self, value: Union[FilterLabel, str]) -> None:
        if isinstance(value, str):
            self._filterLabel = FilterLabel(value)
        elif isinstance(value, FilterLabel):
            self._filterLabel = value
        else:
            raise TypeError(
                "filterLabel must be a FilterLabel Enum, or "
                "one of the corresponding strings."
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

    def copy(self) -> Self:
        """Return a copy of the DonutImage object.

        Returns
        -------
        DonutImage
            A deep copy of self.
        """
        return deepcopy(self)
