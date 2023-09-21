import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import shift
from scipy.optimize import minimize
from scipy.signal import correlate


def centerWithTemplate(
    image: np.ndarray,
    template: np.ndarray,
    subPixel: bool = True,
    rMax: float = 10,
    strict: bool = False,
) -> np.ndarray:
    """Center the image by correlating with the template.

    Parameters
    ----------
    image : np.ndarray
        The image to be centered
    template : np.ndarray
        The template to use for correlation
    subPixel : bool, optional
        Whether to center with sub-pixel resolution.
        (the default is True)
    rMax : float, optional
        The maximum distance the image can be shifted, in pixels.
        (the default is 10)
    strict : bool, optional
        Whether to raise an error if the subpixel optimizer fails.
        (the default is False)

    Returns
    -------
    np.ndarray
        The centered image
    """
    # Replace any NaNs with zeros, because the NaNs screw up the correlation
    image = np.nan_to_num(image)

    # Correlate the template with the image
    corr = correlate(image, template, mode="same")

    # Mask within rMax of the center. We will not allow larger shifts.
    grid = np.arange(image.shape[0])
    rGrid = np.sqrt(
        np.sum(np.square(np.meshgrid(grid, grid) - grid.mean()), axis=0)
    )
    mask = rGrid <= rMax

    # Get the index of maximum correlation
    idxMax = np.unravel_index(np.argmax(mask * corr), corr.shape)
    idxMax = np.array(idxMax)  # type: ignore

    # If we don't want subpixel resolution, we will just roll the image and return
    if not subPixel:
        # Find shift relative to the center
        centerShift = idxMax - image.shape[0] // 2  # type: ignore

        # Roll the image so that the pixel with max correlation is in the center
        image = np.roll(image, -centerShift, (0, 1))

        return image

    # Otherwise, we will use scipy.optimize to get sub-pixel resolution
    # First create a pixel grid, and a small bounding box around idxMax
    grid = np.arange(image.shape[0])
    bounds = idxMax[:, None] + [-5, 5]  # type: ignore
    ySlice = slice(*bounds[0] + [0, 1])
    xSlice = slice(*bounds[1] + [0, 1])

    # Create a function that interpolates (negative) correlation values
    def interpCorr(pixels: np.ndarray) -> float:
        return -interpn(
            (grid[ySlice], grid[xSlice]),
            corr[ySlice, xSlice],
            pixels,
            method="splinef2d",
        )

    try:
        # Find the maximum correlation with subpixel resolution
        result = minimize(interpCorr, idxMax, bounds=bounds).x

        # If the optimization wasn't successful raise an error
        if not result.success:
            raise Exception

        # If we had success, extract the optimized parameters
        idxMax = result.x

    except Exception as e:
        # If optimization failed...
        if strict:
            # If we are being strict, raise the error
            raise e

        # Otherwise just use subpixel precision...
        # Find shift relative to the center
        centerShift = idxMax - image.shape[0] // 2  # type: ignore

        # Roll the image so that the pixel with max correlation is in the center
        image = np.roll(image, -centerShift, (0, 1))

    else:
        # If optimization succeeded, proceed with subpixel shift
        # Find shift relative to the center
        centerShift = idxMax - image.shape[0] / 2  # type: ignore

        # Undo the shift
        image = shift(image, -centerShift)

    return image
