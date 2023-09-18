"""Class that maps the pupil to the image plane, and vice versa.

This is used in the TIE and Danish algorithms.
"""
from pathlib import Path
from typing import Optional, Tuple, Union

import galsim
import numpy as np
from scipy.interpolate import interpn

from jf_wep.donutStamp import DonutStamp
from jf_wep.instrument import Instrument
from jf_wep.utils import (
    DefocalType,
    PlaneType,
    centerWithTemplate,
    loadConfig,
    mergeParams,
    polygonContains,
    zernikeGradEval,
)


class ImageMapper:
    """Class for mapping the pupil to the image plane, and vice versa.

    This class also creates image masks.

    Parameters
    ----------
    configFile : Path or str, optional
        Path to file specifying values for the other parameters. If the
        path starts with "policy/", it will look in the policy directory.
        Any explicitly passed parameters override values found in this file
        (the default is policy/utils/imageMapper.yaml)
    instConfig : Path or str or dict or Instrument, optional
        Instrument configuration. If a Path or string, it is assumed this
        points to a config file, which is used to configure the Instrument.
        If a dictionary, it is assumed to hold keywords for configuration.
        If an Instrument object, that object is just used.
    opticalModel : str, optional
        The optical model to use for mapping between the image and pupil planes.
        Can be either "onAxis" or "offAxis". It is recommended you use offAxis,
        as this model can account for wide-field distortion effects, and so
        is appropriate for a wider range of field angles. However, the offAxis
        model requires a Batoid model of the telescope. If you do not have such
        a model, you can use the onAxis model, which is analytic, but is only
        appropriate near the optical axis. The field angle at which the onAxis
        model breaks down is telescope dependent.
    """

    def __init__(
        self,
        configFile: Union[Path, str, None] = "policy/imageMapper.yaml",
        instConfig: Union[Path, str, dict, Instrument, None] = None,
        opticalModel: Optional[str] = None,
    ) -> None:
        # Merge keyword arguments with defaults from configFile
        params = mergeParams(
            configFile,
            instConfig=instConfig,
            opticalModel=opticalModel,
        )

        # Configure the instrument
        self.configInstrument(params["instConfig"])

        # Set the optical model
        self.opticalModel = params["opticalModel"]  # type: ignore

    def configInstrument(
        self, instConfig: Union[Instrument, Path, str, dict]
    ) -> None:
        """Configure the instrument.

        For details about this parameter, see the class docstring.
        """
        self._instrument = loadConfig(instConfig, Instrument)

    @property
    def instrument(self) -> Instrument:
        """Return the instrument object.

        For details about this parameter, see the class docstring.
        """
        return self._instrument

    @property
    def opticalModel(self) -> str:
        return self._opticalModel

    @opticalModel.setter
    def opticalModel(self, value: str) -> None:
        allowedModels = ["onAxis", "offAxis"]
        if not isinstance(value, str) or value not in allowedModels:
            raise TypeError(
                f"opticalModel must be one of {str(allowedModels)[1:-1]}."
            )
        self._opticalModel = value

    def _constructForwardMap(
        self,
        uPupil: np.ndarray,
        vPupil: np.ndarray,
        zkCoeff: np.ndarray,
        donutStamp: DonutStamp,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct the forward mapping from the pupil to the image plane.

        Parameters
        ----------
        uPupil : np.ndarray
             Normalized x coordinates on the pupil plane
        vPupil : np.ndarray
             Normalized y coordinates on the image plane
        zkCoeff : np.ndarray
            The wavefront at the pupil, represented as Zernike coefficients
            in meters for Noll indices >= 4.
        donutStamp : DonutStamp
            A stamp object containing the metadata required for the mapping.

        Returns
        -------
        np.ndarray
            Normalized x coordinates on the image plane
        np.ndarray
            Normalized y coordinates on the image plane
        np.ndarray
            The Jacobian of the forward map
        np.ndarray
            The determinant of the Jacobian
        """
        # Get the reference Zernikes
        if self.opticalModel == "onAxis":
            zkRef = self.instrument.getIntrinsicZernikes(
                *donutStamp.fieldAngle,
                donutStamp.bandLabel,
            )
        else:
            zkRef = self.instrument.getOffAxisCoeff(
                *donutStamp.fieldAngle,
                donutStamp.defocalType,
                donutStamp.bandLabel,
            )

        # Create an array for the Zernike coefficients used for the mapping
        size = max(zkCoeff.size, zkRef.size)
        zkMap = np.zeros(size)

        # Add the initial and reference Zernikes
        zkMap[: zkCoeff.size] = zkCoeff
        zkMap[: zkRef.size] += zkRef

        # Calculate all 1st- and 2nd-order Zernike derivatives
        d1Wdu = zernikeGradEval(
            uPupil,
            vPupil,
            uOrder=1,
            vOrder=0,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )
        d1Wdv = zernikeGradEval(
            uPupil,
            vPupil,
            uOrder=0,
            vOrder=1,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )
        d2Wdudu = zernikeGradEval(
            uPupil,
            vPupil,
            uOrder=2,
            vOrder=0,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )
        d2Wdvdv = zernikeGradEval(
            uPupil,
            vPupil,
            uOrder=0,
            vOrder=2,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )
        d2Wdudv = zernikeGradEval(
            uPupil,
            vPupil,
            uOrder=1,
            vOrder=1,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )
        d2Wdvdu = d2Wdudv

        # Plus the first order derivatives at the center of the pupil
        d1Wdu0 = zernikeGradEval(
            np.zeros(1),
            np.zeros(1),
            uOrder=1,
            vOrder=0,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )
        d1Wdv0 = zernikeGradEval(
            np.zeros(1),
            np.zeros(1),
            uOrder=0,
            vOrder=1,
            zkCoeff=zkMap,
            obscuration=self.instrument.obscuration,
        )

        # Get the required info about the telescope geometry
        N = self.instrument.focalRatio
        l = self.instrument.defocalOffset  # noqa: E741

        # Calculate the mapping determined by the optical model
        if self.opticalModel == "onAxis":
            # The onAxis model is analytic and is intended for donuts near
            # the center of the focal plane. How far off-axis this model
            # works will be telescope specific.

            # The onAxis model does not require a batoid model, but if you
            # do have a good batoid model, there is no reason to ever use
            # the onAxis model

            # Determine defocal sign from the image plane at z = f +/- l
            # I.e., the extrafocal image at z = f + l is associated with +1,
            # and the intrafocal image at z = f - l is associated with -1.
            defocalSign = (
                +1 if donutStamp.defocalType == DefocalType.Extra else -1
            )

            # Calculate the prefactor
            rPupil = np.sqrt(uPupil**2 + vPupil**2)
            with np.errstate(invalid="ignore"):
                prefactor = np.sqrt(
                    (4 * N**2 - 1) / (4 * N**2 - rPupil**2)
                )

            # Map the pupil points onto the image plane
            uImage = prefactor * (
                -defocalSign * uPupil - 4 * N**2 / l * (d1Wdu - d1Wdu0)
            )
            vImage = prefactor * (
                -defocalSign * vPupil - 4 * N**2 / l * (d1Wdv - d1Wdv0)
            )

            # Calculate the elements of the Jacobian
            J00 = uPupil * uImage / (4 * N**2 - rPupil**2) - prefactor * (
                defocalSign + 4 * N**2 / l * d2Wdudu
            )
            J01 = (
                vPupil * uImage / (4 * N**2 - rPupil**2)
                - prefactor * 4 * N**2 / l * d2Wdvdu
            )
            J10 = (
                uPupil * vImage / (4 * N**2 - rPupil**2)
                - prefactor * 4 * N**2 / l * d2Wdudv
            )
            J11 = vPupil * vImage / (4 * N**2 - rPupil**2) - prefactor * (
                defocalSign + 4 * N**2 / l * d2Wdvdv
            )

        else:
            # The offAxis model uses a numerically-fit model from batoid
            # This model is able to account for widefield distortion effects
            # that are not captured by the intrinsic Zernikes of the telescope

            # Calculate the prefactor
            prefactor = -2 * N * np.sqrt(4 * N**2 - 1) / l

            # Map the pupil points onto the image plane
            uImage = prefactor * (d1Wdu - d1Wdu0)
            vImage = prefactor * (d1Wdv - d1Wdv0)

            # Calculate the elements of the Jacobian
            J00 = prefactor * d2Wdudu
            J01 = prefactor * d2Wdvdu
            J10 = prefactor * d2Wdudv
            J11 = prefactor * d2Wdvdv

        # Assemble the Jacobian
        jac = np.array(
            [
                [J00, J01],
                [J10, J11],
            ]
        )

        # Calculate the determinant
        jacDet = J00 * J11 - J01 * J10

        return uImage, vImage, jac, jacDet

    def _constructInverseMap(
        self,
        uImage: np.ndarray,
        vImage: np.ndarray,
        zkCoeff: np.ndarray,
        donutStamp: DonutStamp,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct the inverse mapping from the image plane to the pupil.

        Parameters
        ----------
        uImage : np.ndarray
            Normalized x coordinates on the image plane
        vImage : np.ndarray
            Normalized y coordinates on the image plane
        zkCoeff : np.ndarray
            The wavefront at the pupil, represented as Zernike coefficients
            in meters for Noll indices >= 4.
        donutStamp : DonutStamp
            A stamp object containing the metadata required for the mapping.

        Returns
        -------
        np.ndarray
            Normalized x coordinates on the pupil plane
        np.ndarray
            Normalized y coordinates on the pupil plane
        np.ndarray
            The Jacobian of the inverse mapping
        np.ndarray
            The determinant of the Jacobian
        """
        # Create a test grid on the pupil to pre-fit the image -> pupil mapping
        uPupilTest = np.linspace(-1, 1, 10)
        uPupilTest, vPupilTest = np.meshgrid(uPupilTest, uPupilTest)

        # Mask outside the pupil
        rPupilTest = np.sqrt(uPupilTest**2 + vPupilTest**2)
        pupilMask = rPupilTest <= 1
        pupilMask &= rPupilTest >= self.instrument.obscuration
        uPupilTest = uPupilTest[pupilMask]
        vPupilTest = vPupilTest[pupilMask]

        # Project the test pupil grid onto the image plane
        uImageTest, vImageTest, jac, jacDet = self._constructForwardMap(
            uPupilTest,
            vPupilTest,
            zkCoeff,
            donutStamp,
        )

        # Use test points to fit Zernike coefficients for image -> pupil mapping
        rImageMax = np.sqrt(uImageTest**2 + vImageTest**2).max()
        invCoeff, *_ = np.linalg.lstsq(
            galsim.zernike.zernikeBasis(
                6,
                uImageTest,
                vImageTest,
                R_outer=rImageMax,
            ).T,
            np.array([uPupilTest, vPupilTest]).T,
            rcond=None,
        )

        # Now we will map our image points to the pupil using the coefficients
        # we just fit, and then map them back to the image plane using the
        # analytic forward mapping
        # Ideally, this round-trip mapping will return the same image points
        # we started with, however our initial image -> pupil mapping will not
        # be perfect, so this will not be the case. We will iteratively apply
        # Newton's method to reduce the residuals, and thereby improve the
        # mapping

        # Map the image points to the pupil
        uPupil = galsim.zernike.Zernike(
            invCoeff[:, 0],
            R_outer=rImageMax,
        )(uImage, vImage)
        vPupil = galsim.zernike.Zernike(
            invCoeff[:, 1],
            R_outer=rImageMax,
        )(uImage, vImage)

        # Map these pupil points back to the image (RT = round-trip)
        uImageRT, vImageRT, jac, jacDet = self._constructForwardMap(
            uPupil,
            vPupil,
            zkCoeff,
            donutStamp,
        )

        # Calculate the residuals of the round-trip mapping
        duImage = uImageRT - uImage
        dvImage = vImageRT - vImage

        # Now iterate Newton's method to improve the mapping
        # (i.e. minimize the residuals)
        for _ in range(10):
            # Add corrections to the pupil coordinates using Newton's method
            uPupil -= (+jac[1, 1] * duImage - jac[0, 1] * dvImage) / jacDet
            vPupil -= (-jac[1, 0] * duImage + jac[0, 0] * dvImage) / jacDet

            # Map these new pupil points to the image plane
            uImageRT, vImageRT, jac, jacDet = self._constructForwardMap(
                uPupil,
                vPupil,
                zkCoeff,
                donutStamp,
            )

            # Calculate the new residuals
            duImage = uImageRT - uImage
            dvImage = vImageRT - vImage

            # If the residuals are small enough, stop iterating
            maxResiduals = np.max([np.abs(duImage), np.abs(dvImage)], axis=0)
            if np.all(maxResiduals <= 1e-5):
                break

        # Set not-converged points to NaN
        notConverged = maxResiduals > 1e-5
        uPupil[notConverged] = np.nan
        vPupil[notConverged] = np.nan
        jac[..., notConverged] = np.nan
        jacDet[notConverged] = np.nan

        # Invert the Jacobian
        jac = (
            np.array([[jac[1, 1], -jac[0, 1]], [-jac[1, 0], jac[0, 0]]])
            / jacDet
        )
        jacDet = 1 / jacDet

        return uPupil, vPupil, jac, jacDet

    def _getImageGridInsidePupil(
        self,
        zkCoeff: np.ndarray,
        donutStamp: DonutStamp,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the image grid and a mask for which pixels are inside the pupil.

        Note a pixel is considered inside the pupil if any fraction of the pixel
        is inside the pupil. In addition, the pupil considered is the pupil
        mapped to the image plane.

        Parameters
        ----------
        zkCoeff : np.ndarray
            The wavefront at the pupil, represented as Zernike coefficients
            in meters for Noll indices >= 4.
        donutStamp : DonutStamp
            A stamp object containing the metadata required for the mapping.

        Returns
        -------
        np.ndarray
            Normalized x coordinates on the image plane
        np.ndarray
            Normalized y coordinates on the image plane
        np.ndarray
            Binary mask array indicating whether each pixel is inside the pupil
        """
        # Map pupil edge to the image to determine edge of pupil on the image
        theta = np.linspace(0, 2 * np.pi, 100)
        uPupil, vPupil = np.cos(theta), np.sin(theta)
        uImageEdge, vImageEdge, *_ = self._constructForwardMap(
            uPupil,
            vPupil,
            zkCoeff,
            donutStamp,
        )
        imageEdge = np.array([uImageEdge, vImageEdge]).T

        # Create an image grid
        nPixels = donutStamp.image.shape[0]
        uImage, vImage = self.instrument.createImageGrid(nPixels)

        # Determine which image pixels have corners inside the pupil
        dPixel = uImage[0, 1] - uImage[0, 0]
        corners = np.append(uImage[0] - dPixel / 2, uImage[0, -1] + dPixel / 2)
        cornersIn = polygonContains(*np.meshgrid(corners, corners), imageEdge)

        # Select pixels that have at least one corner inside
        inside = (
            cornersIn[:-1, :-1]
            | cornersIn[1:, :-1]
            | cornersIn[:-1, 1:]
            | cornersIn[1:, 1:]
        )

        return uImage, vImage, inside

    def _maskWithCircle(
        self,
        uPupil: np.ndarray,
        vPupil: np.ndarray,
        uPupilCirc: float,
        vPupilCirc: float,
        rPupilCirc: float,
        fwdMap: Optional[tuple] = None,
    ) -> np.ndarray:
        """Return a fractional mask for a single circle.

        Parameters
        ----------
        uPupil : np.ndarray
            Normalized x coordinates on the pupil plane
        vPupil : np.ndarray
            Normalized y coordinates on the pupil plane
        uPupilCirc : float
            The u coordinate of the circle center
        vPupilCirc : float
            The v coordinate of the circle center
        rPupilCirc : float
            The normalized radius of the circle
        fwdMap : tuple
            A tuple containing (uImage, vImage, jac, jacDet), i.e.
            the output of self._constructForwardMap(uPupil, vPupil, ...)
            If not None, the mask is mapped to the image plane.
            (the default is None)

        Returns
        -------
        np.ndarray
            Fractional mask with the same shape as uPupil
        """
        # Center the pupil coordinates on the circle's center
        uPupilCen = uPupil - uPupilCirc
        vPupilCen = vPupil - vPupilCirc

        # Pixel scale in normalized coordinates is inverse of the donut radius
        pixelScale = 1 / self.instrument.donutRadius

        # If a forward map is provided, begin preparing for mapping the mask
        # to the image plane
        if fwdMap is not None:
            uImage, vImage, jac, jacDet = fwdMap

            # Calculate quantities for the forward map
            invJac = np.array(
                [
                    [+jac[1, 1], -jac[0, 1]],  # type: ignore
                    [-jac[1, 0], +jac[0, 0]],  # type: ignore
                ]
            )
            invJac /= jacDet

            # Use a local linear approximation to center the image coordinates
            uImageCen = (
                uImage - jac[0, 0] * uPupilCirc - jac[0, 1] * vPupilCirc
            )
            vImageCen = (
                vImage - jac[1, 0] * uPupilCirc - jac[1, 1] * vPupilCirc
            )

            # Calculate the diagonal distance across each pixel on the pupil
            diagL = np.sqrt(
                (invJac[0, 0] + invJac[0, 1]) ** 2  # type: ignore
                + (invJac[1, 0] + invJac[1, 1]) ** 2  # type: ignore
            )
            diagL *= pixelScale

        else:
            # Use the pupil coordinates as the image coordinates
            uImageCen = uPupilCen
            vImageCen = vPupilCen

            # Diagonal distance across a regular pixel
            diagL = np.sqrt(2) * pixelScale

        # Assign pixels to groups based on whether they're definitely
        # inside/outside the circle, or on the border
        rPupilCen = np.sqrt(uPupilCen**2 + vPupilCen**2)
        inside = rPupilCen < (rPupilCirc - diagL / 2)
        outside = rPupilCen > (rPupilCirc + diagL / 2)
        border = ~(inside | outside)

        # We can go ahead and assign fractional mask 1 (0) to pixels
        # totally inside (outside) the circle
        out = np.zeros_like(uPupil)
        out[inside] = 1
        out[outside] = 0

        # If nothing is on the border, go ahead and return the mask
        if not border.any():
            return out

        # Calculate coefficients for the line (y = m*x + b) that is tangent to
        # the circle where the ray that passes through each point intersects the
        # circle (in pupil coordinates)
        uPupilCen, vPupilCen = uPupilCen[border], vPupilCen[border]
        m = -uPupilCen / vPupilCen  # slope
        b = (
            np.sqrt(uPupilCen**2 + vPupilCen**2) * rPupilCirc / vPupilCen
        )  # intercept

        # Select the border image coordinates
        uImageCen, vImageCen = uImageCen[border], vImageCen[border]

        if fwdMap is not None:
            # Transform the slope and intercept to image coordinates
            invJac = invJac[..., border]  # type: ignore
            a1 = m * invJac[0, 0] - invJac[1, 0]
            a2 = m * uPupilCen + b - vPupilCen
            a3 = -m * invJac[0, 1] + invJac[1, 1]
            m = a1 / a3
            b = (a2 - a1 * uImageCen) / a3 + vImageCen

        # Use symmetry to map everything onto the situation where -1 <= mImage <= 0
        mask = m > 0
        uImageCen[mask] = -uImageCen[mask]
        m[mask] = -m[mask]

        mask = m < -1
        uImageCen[mask], vImageCen[mask] = vImageCen[mask], uImageCen[mask]
        m[mask], b[mask] = 1 / m[mask], -b[mask] / m[mask]

        # Calculate the v intercept on the right side of the pixel
        vStar = m * (uImageCen + pixelScale / 2) + b

        # Calculate the fractional distance of intercept from the top of the pixel
        gamma = (vImageCen + pixelScale / 2 - vStar) / pixelScale

        # Now determine illumination for border pixels
        borderOut = np.zeros_like(uPupilCen)

        # Pixels that are totally inside the circle
        mask = gamma < 0
        borderOut[mask] = 1

        # Pixels that are totally outside the circle
        mask = gamma > (1 - m)
        borderOut[mask] = 0

        # Pixels for which the circle crosses the left and bottom sides
        mask = (1 < gamma) & (gamma < (1 - m))
        borderOut[mask] = -0.5 / m[mask] * (1 - (gamma[mask] + m[mask])) ** 2

        # Pixels for which the circle crosses the left and right sides
        mask = (-m < gamma) & (gamma < 1)
        borderOut[mask] = 1 - gamma[mask] - m[mask] / 2

        # Pixels for which the circle crosses the top and right
        mask = (0 < gamma) & (gamma < -m)
        borderOut[mask] = 1 + 0.5 * gamma[mask] ** 2 / m[mask]

        # Values below the (centered) u axis need to be flipped
        mask = vImageCen < 0
        borderOut[mask] = 1 - borderOut[mask]

        # Put the border values into the global output array
        out[border] = borderOut

        return out

    def _maskLoop(
        self,
        donutStamp: DonutStamp,
        uPupil: np.ndarray,
        vPupil: np.ndarray,
        fwdMap: Optional[tuple] = None,
        binary: bool = False,
    ) -> np.ndarray:
        """Loop through mask elements to create the mask.

        Parameters
        ----------
        donutStamp : DonutStamp
            A stamp object containing the metadata required for constructing the mask.
        uPupil : np.ndarray
            Normalized x coordinates on the pupil plane
        vPupil : np.ndarray
            Normalized y coordinates on the pupil plane
        fwdMap : tuple
            A tuple containing (uImage, vImage, jac, jacDet), i.e.
            the output of self._constructForwardMap(uPupil, vPupil, ...)
            If not None, the mask is mapped to the image plane.
            (the default is None)
        binary : bool, optional
            Whether to return a binary mask. If False, a fractional mask is returned
            instead. (the default is False)

        Returns
        -------
        np.ndarray
            A flattened mask array
        """
        # Get the field angle
        angle = donutStamp.fieldAngle

        # Get the angle radius
        rTheta = np.clip(np.sqrt(np.sum(np.square(angle))), 1e-16, None)

        # Flatten the pupil arrays
        uPupil, vPupil = uPupil.ravel(), vPupil.ravel()

        # If a forward map is provided, flatten those arrays too
        if fwdMap is not None:
            uImage, vImage, jac, jacDet = fwdMap
            uImage, vImage = uImage.ravel(), vImage.ravel()
            jac = jac.reshape(2, 2, -1)
            jacDet = jacDet.ravel()

        # Get the mask parameters from the instrument
        maskParams = self.instrument.maskParams

        # Start with a full mask
        mask = np.ones_like(uPupil)

        # Loop over each mask element
        for key, val in maskParams.items():
            # Get the indices of non-zero pixels
            idx = np.nonzero(mask)[0]

            # If all the pixels are zero, stop here
            if not idx.any():
                break

            # Only apply this mask if we're past thetaMin
            if rTheta < val["thetaMin"]:
                continue

            # Calculate the radius and center of the mask
            radius = np.polyval(val["radius"], rTheta)
            rCenter = np.polyval(val["center"], rTheta)

            uCenter = rCenter * angle[0] / rTheta
            vCenter = rCenter * angle[1] / rTheta

            # Calculate the mask values
            maskVals = self._maskWithCircle(
                uPupil=uPupil[idx],
                vPupil=vPupil[idx],
                uPupilCirc=uCenter,
                vPupilCirc=vCenter,
                rPupilCirc=radius,  # type: ignore
                fwdMap=None
                if fwdMap is None
                else (uImage[idx], vImage[idx], jac[..., idx], jacDet[idx]),
            )

            # Assign the mask values
            if key.endswith("Inner"):
                mask[idx] = np.minimum(mask[idx], 1 - maskVals)
            else:
                mask[idx] = np.minimum(mask[idx], maskVals)

        if binary:
            mask = mask >= 0.5

        return mask

    def createPupilMask(
        self,
        donutStamp: DonutStamp,
        *,
        binary: bool = False,
    ) -> np.ndarray:
        """Create the pupil mask for the stamp.

        Parameters
        ----------
        donutStamp : DonutStamp
            A stamp object containing the metadata required for constructing the mask.
        binary : bool, optional
            Whether to return a binary mask. If False, a fractional mask is returned
            instead. (the default is False)

        Returns
        -------
        np.ndarray
            The pupil mask
        """
        # Get the pupil grid
        uPupil, vPupil = self.instrument.createPupilGrid()

        # Get the mask by looping over the mask elements
        mask = self._maskLoop(
            donutStamp=donutStamp,
            uPupil=uPupil,
            vPupil=vPupil,
            fwdMap=None,
            binary=binary,
        )

        # Restore the mask shape
        mask = mask.reshape(uPupil.shape)

        return mask

    def createImageMask(
        self,
        donutStamp: DonutStamp,
        zkCoeff: np.ndarray = np.zeros(1),
        *,
        binary: bool = False,
        _invMap: Optional[tuple] = None,
    ) -> np.ndarray:
        """Create the image mask for the stamp.

        Note the uImage and vImage arrays must be regular 2D grids, like what is
        be returned by np.meshgrid.

        Parameters
        ----------
        donutStamp : DonutStamp
            A stamp object containing the metadata required for constructing the mask.
        zkCoeff : np.ndarray
            The wavefront at the pupil, represented as Zernike coefficients
            in meters, for Noll indices >= 4.
        binary : bool, optional
            Whether to return a binary mask. If False, a fractional mask is returned
            instead. (the default is False)

        Returns
        -------
        np.ndarray
        """
        # Get the image grid inside the pupil
        uImage, vImage, inside = self._getImageGridInsidePupil(
            zkCoeff, donutStamp
        )

        # Get the inverse mapping from image plane to pupil plane
        if _invMap is None:
            # Construct the inverse mapping
            uPupil, vPupil, invJac, invJacDet = self._constructInverseMap(
                uImage[inside],
                vImage[inside],
                zkCoeff,
                donutStamp,
            )
        else:
            uPupil, vPupil, invJac, invJacDet = _invMap

        # Rearrange into the forward map
        jac = np.array(
            [
                [+invJac[1, 1], -invJac[0, 1]],  # type: ignore
                [-invJac[1, 0], +invJac[0, 0]],  # type: ignore
            ]
        )
        jac /= invJacDet
        jacDet = 1 / invJacDet

        # Package the forward mapping
        fwdMap = (uImage[inside], vImage[inside], jac, jacDet)

        # Get the mask by looping over the mask elements
        mask = np.zeros_like(inside, dtype=float)
        mask[inside] = self._maskLoop(
            donutStamp=donutStamp,
            uPupil=uPupil,
            vPupil=vPupil,
            fwdMap=fwdMap,
            binary=binary,
        )

        return mask

    def mapPupilToImage(
        self,
        donutStamp: DonutStamp,
        zkCoeff: np.ndarray = np.zeros(1),
    ) -> DonutStamp:
        """Map the pupil to the image plane.

        Parameters
        ----------
        donutStamp : DonutStamp
            A stamp object containing the metadata needed for the mapping.
            It is assumed that mapping the pupil to the image plane is meant
            to model the image contained in this stamp.
        zkCoeff : np.ndarray, optional
            The wavefront at the pupil, represented as Zernike coefficients
            in meters, for Noll indices >= 4.

        Returns
        -------
        DonutStamp
            The stamp object mapped to the image plane.
        """
        # Make a copy of the stamp
        stamp = donutStamp.copy()

        # Get the image grid inside the pupil
        uImage, vImage, inside = self._getImageGridInsidePupil(zkCoeff, stamp)

        # Construct the inverse mapping
        uPupil, vPupil, jac, jacDet = self._constructInverseMap(
            uImage[inside],
            vImage[inside],
            zkCoeff,
            stamp,
        )

        # Create the image mask
        mask = self.createImageMask(
            stamp,
            zkCoeff,
            _invMap=(uPupil, vPupil, jac, jacDet),
        )

        # Fill the image (this assumes that, except for vignetting,
        # the pupil is uniformly illuminated)
        stamp.image = np.zeros_like(stamp.image)
        stamp.image[inside] = mask[inside] * jacDet

        # Also save the mask
        stamp.mask = mask

        # And set the plane type
        stamp.planeType = PlaneType.Image

        return stamp

    def mapImageToPupil(
        self,
        donutStamp: DonutStamp,
        zkCoeff: np.ndarray = np.zeros(1),
    ) -> DonutStamp:
        """Map a stamp from the image to the pupil plane.

        Parameters
        ----------
        donutStamp : DonutStamp
            A stamp object containing the array to be mapped from the image
            to the pupil plane, plus the required metadata.
        zkCoeff : np.ndarray, optional
            The wavefront at the pupil, represented as Zernike coefficients
            in meters, for Noll indices >= 4.
            (the default is zero)

        Returns
        -------
        DonutStamp
            The stamp object mapped to the image plane.
        """
        # Make a copy of the stamp
        stamp = donutStamp.copy()

        # Create regular pupil and image grids
        uPupil, vPupil = self.instrument.createPupilGrid()
        uImage, vImage = self.instrument.createImageGrid(stamp.image.shape[0])

        template = self.createImageMask(stamp, zkCoeff, binary=True)
        stamp.image = centerWithTemplate(stamp.image, template)

        # Construct the forward mapping
        uImageMap, vImageMap, jac, jacDet = self._constructForwardMap(
            uPupil,
            vPupil,
            zkCoeff,
            stamp,
        )

        # Interpolate the array onto the pupil plane
        pupil = interpn(
            (vImage[:, 0], uImage[0, :]),
            stamp.image,
            (vImageMap, uImageMap),
            method="linear",
            bounds_error=False,
        )
        pupil *= jacDet

        # Set NaNs to zero
        pupil = np.nan_to_num(pupil)

        # Mask the pupil
        mask = self.createPupilMask(stamp, binary=True)
        pupil *= mask

        # Update the stamp with the new pupil image
        stamp.image = pupil

        # Also save the mask
        stamp.mask = mask

        # And set the plane type
        stamp.planeType = PlaneType.Pupil

        return stamp
