"""Functions to flag grid points as inside or outside a polygon.

Note that we compile these functions with numba so that they run faster.
For both functions, we provide an explicit signature so they are eagerly
compiled, i.e. on import instead of execution.
"""
import numba
import numpy as np


@numba.jit(
    "boolean[:](float64, float64[:], float64[:,:])",
    nopython=True,
    fastmath=True,
)
def _polygonContainsRow(
    y: float, row: np.ndarray, poly: np.ndarray
) -> np.ndarray:
    """Return mask indicating if each point in the row is inside the polygon.

    Parameters
    ----------
    y : float
        The y value of every point in the row
    row : np.ndarray
        A 1D array of x values for every point in the row
    poly : np.ndarray
        An Nx2 array of points specifying the polygon vertices. It is assumed
        that an edge connects every pair of adjacent points, and that the final
        point is identical to the first point.

    Returns
    -------
    np.ndarray
        A 1D array with the same shape as row, indicating whether each point 
        is inside the polygon
    """
    # If the row is totally outside the polygon, return False
    if row.max() < poly[:, 0].min() or row.min() > poly[:, 0].max():
        return np.full_like(row, False, dtype=np.bool_)

    # Determine which polygon edges cross our column
    dy0 = y - poly[:-1, 1]
    dy1 = y - poly[1:, 1]
    idx = np.where(dy0 * dy1 < 0)[0]

    # Solve for the x-values of these edges where they cross the row
    m = (poly[idx + 1, 0] - poly[idx, 0]) / (poly[idx + 1, 1] - poly[idx, 1])
    x = m * dy0[idx] + poly[idx, 0]

    # Count the number of edges to the right of each point in the row
    edgesToRight = np.sum(row[:, None] < x, axis=-1)

    # The point is inside if the number of edges to the right is odd
    inside = edgesToRight % 2 == 1

    return inside


@numba.jit(
    "boolean[:,:](float64[:,:], float64[:,:], float64[:,:])",
    nopython=True,
    fastmath=True,
)
def polygonContains(
    xGrid: np.ndarray,
    yGrid: np.ndarray,
    poly: np.ndarray,
) -> np.ndarray:
    """Return mask indicating if each point in the grid is inside the polygon.

    Note this function works only in 2D, and assumes that xGrid and yGrid are
    regular grids like will be returned from np.meshgrid().

    Parameters
    ----------
    xGrid : np.ndarray
        A 2D array of x values for the grid points
    yGrid : np.ndarray
        A 2D array of y values for the grid points
    poly : np.ndarray
        An Nx2 array of points specifying the polygon vertices. It is assumed
        that an edge connects every pair of adjacent points, and that the final
        point is identical to the first point.

    Returns
    -------
    np.ndarray
        A 2D array with the same shape as xGrid and yGrid, indicating whether
        each point is inside the polygon
    """
    # Get the array of unique y values
    y = yGrid[:, 0]

    # Create an array full of False
    inside = np.full_like(xGrid, False, dtype=np.bool_)

    # Determine which rows have y-values that fall within polygon limits
    idx = np.where((y > poly[:, 1].min()) & (y < poly[:, 1].max()))[0]

    # If none do, we can return all False
    if len(idx) == 0:
        return inside

    # Add some tiny shifts to the y-values of the polygon vertices
    # This helps avoid problems with horizontal lines
    polyShifted = poly.copy()
    shifts = np.arange(len(poly) - 2) % 11 - 5
    polyShifted[1:-1, 1] += 1e-12 * shifts

    # Loop over rows inside polygon limits
    for i in range(len(idx)):
        inside[idx[i]] = _polygonContainsRow(
            y[idx[i]],
            xGrid[idx[i]],
            polyShifted,
        )

    return inside