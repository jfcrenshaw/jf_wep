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

    Returns
    -------
    np.ndarray
        The centered image
    """
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

    if subPixel:
        # Now use scipy.optimize to get max with sub-pixel resolution

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

        # Find the maximum correlation with subpixel resolution
        idxMax = minimize(interpCorr, idxMax, bounds=bounds).x

        # Find shift relative to the center
        centerShift = idxMax - image.shape[0] / 2  # type: ignore

        # Undo the shift
        image = shift(image, -centerShift, cval=np.nan)

    else:
        # Find shift relative to the center
        centerShift = idxMax - image.shape[0] // 2  # type: ignore

        # Roll the image so that the pixel with max correlation is in the center
        image = np.roll(image, -centerShift, (0, 1))

    return image
