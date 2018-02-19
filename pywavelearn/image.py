"""
Functions to deal with image processing

Source code for image rectication was taken from
Flamingo [https://github.com/openearth/flamingo].
Only the docstrings were updated to match my style.
"""
import numpy as np

import cv2
import skimage.io
import skimage.util

from collections import OrderedDict


def pixel_window(a, i, j, iwin=8, jwin=8):
    """
    Get the surrouding pixels from a given pixel center.

    ----------
    Args:
        a (Mandatory [np.array]): input image array. Only used to figure out
                                  shapes.

        i (Mandatory [int]): pixel center in the i-direction.

        j (Mandatory [int]): pixel center in the j-direction.

        iwin (Optional [int]): window size in the i-direction. Default is 8.

        iwin (Optional [int]): window size in the j-direction. Default is 8.

    ----------
    Returns:
         I (Mandatory [np.array]): surrounding pixels in the i-direction.

         J (Mandatory [np.array]): surrounding pixels in the j-direction.
    """

    # compute domain
    i = np.arange(i-iwin, i+iwin+1, 1)
    j = np.arange(j-jwin, j+jwin+1, 1)

    # all pixels inside the domain
    Iimg, Jimg = np.meshgrid(i, j)

    # Remove pixels outside the borders

    # i-dimension
    Iimg = Iimg.flatten()
    Iimg[Iimg < 0] = -999
    Iimg[Iimg > a.shape[0]] = -999
    idx = np.where(Iimg == -999)

    # j-dimension
    Jimg = Jimg.flatten()
    Jimg[Jimg < 0] = -999
    Jimg[Jimg > a.shape[1]] = -999
    jdx = np.where(Jimg == -999)

    # finall array
    Ifinal = np.delete(Iimg, np.hstack([idx, jdx]))
    Jfinal = np.delete(Jimg, np.hstack([idx, jdx]))

    return Ifinal, Jfinal


def equalize(I, method="global"):
    """
    Equalize a low-contrast image.

    ----------
    Args:
        I [Mandatory (np.ndarray)]: image array. values must range from 0 to 1.
                                    ex: I = skimage.color.rgb2grey(img)

        method [Optional (str)]: Method to be used. Only global equalization is
                                 implemented by now.

    ----------
    Returns:
        IE [Mandatory (np.ndarray)]: equalized image.
    """
    from skimage import exposure
    from skimage.util import img_as_ubyte
    from skimage.morphology import disk, rectangle

    # Scale image to 0-255
    Iubyte = img_as_ubyte(I)

    if method == "global":
        IE = exposure.equalize_hist(I)
    else:
        raise NotImplementedError("Sorry, other methods are not \
                                  implemented yet.")

    return IE


def camera_parser(fname):
    """
    Read the camera matrix and distortion coeficients generated from
    "../scripts/calibrate.py"

    ----------
    Args:
        fname [Mandatory (str)]: Path to the camera calibration file.

    ----------
    Returns:
        K [ (np.ndarray)]: Camera matrix (OpenCV conventions)

        DC [ (np.ndarray)]: Vector with distortion coefficients
                            [k_1, k_2, p_1, p_2(, k_3(, k_4, k_5, k_6))]
                            of 4, 5, or 8 elements.
    """

    # open the file
    f = open(fname, "r")
    # read all lines
    lines = f.readlines()

    # build the camera matrix
    K = np.zeros([3, 3])
    for l, line in enumerate(lines):
        if "Camera Matrix:" in line:
            break
    K[0, 0] = float(lines[l+2].strip("\n").split(",")[0])
    K[0, 1] = float(lines[l+2].strip("\n").split(",")[1])
    K[0, 2] = float(lines[l+2].strip("\n").split(",")[2])
    K[1, 0] = float(lines[l+3].strip("\n").split(",")[0])
    K[1, 1] = float(lines[l+3].strip("\n").split(",")[1])
    K[1, 2] = float(lines[l+3].strip("\n").split(",")[2])
    K[2, 0] = float(lines[l+4].strip("\n").split(",")[0])
    K[2, 1] = float(lines[l+4].strip("\n").split(",")[1])
    K[2, 2] = float(lines[l+4].strip("\n").split(",")[2])

    # read the distortion coeficients
    DC = []
    for l, line in enumerate(lines):
        if "Distortion Coeficients:" in line:
            break
    DC.append(float(lines[l+2].split(":")[1]))  # K1
    DC.append(float(lines[l+3].split(":")[1]))  # K2
    DC.append(float(lines[l+4].split(":")[1]))  # P1
    DC.append(float(lines[l+5].split(":")[1]))  # P2
    DC.append(float(lines[l+6].split(":")[1]))  # K3

    return K, np.array(DC)


def metadata_parser(fname, output="metadata.txt"):
    """
    Extract metadata from a video file using exiftool.

    Use "apt-get install exiftool" to install, if not available in your system.

    ----------
    Args:
        fname [Mandatory (str)]: Full path to video file.
                                 example: /data/my-cool-video.avi

        output [Optional (str)]: Full path to the output file. Default is
                                 metadata.txt
    ----------
    Returns:
        metadata [Mandatory (collections.OrderedDict)]: A ordered dictonary
                                                        containing all
                                                        iformation exiftool
                                                        could get from
                                                        the input file
    """
    from subprocess import call

    # system call
    cmd = "exiftool {} -csv > {}".format(fname, output)
    call(cmd, shell=True)

    # Open the file
    f = open(output).readlines()

    # Get keys
    keys = []
    for key in f[0].split(","):
        keys.append(key)

    # Get values
    values = []
    for value in f[1].split(","):
        values.append(value)

    # Open a empty dictonary
    metadata = OrderedDict()

    # Populate the dictonary
    for key, value in zip(keys, values):
        metadata[key] = value

    return metadata


def find_homography(UV, XYZ, K, distortion=np.zeros((1, 4)), z=0):
    """
    Find homography based on ground control points

    Function uses the OpenCV image rectification workflow as described in
    http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    starting with solvePnP.

    This funciton was taken from Flamingo.
    ----------
    Args:
        UV [Mandatory (np.ndarray)]: Nx2 array of image coordinates of gcp's

        XYZ [Mandatory (np.ndarray)]: Nx3 array of real-world coordinates
                                      of gcp's

        K [Mandatory (np.ndarray)]: 3x3 array containing camera matrix

        distortion [Mandatory (np.ndarray)]: 1xP array with distortion
                                             coefficients with P = 4, 5 or 8

        z [Mandatory (float)]: Real-world elevation on which the image
                               should be projected.
    ----------
    Returns:
        H [Mandatory (np.ndarray)]: 3x3 homography matrix
    """

    UV = np.asarray(UV).astype(np.float32)
    XYZ = np.asarray(XYZ).astype(np.float32)
    K = np.asarray(K).astype(np.float32)

    # compute camera pose
    retval, rvec, tvec = cv2.solvePnP(XYZ, UV, K, distortion)

    # convert rotation vector to rotation matrix
    R = cv2.Rodrigues(rvec)[0]

    # assume height of projection plane
    R[:, 2] = R[:, 2] * z

    # add translation vector
    R[:, 2] = R[:, 2] + tvec.flatten()

    # compute homography
    H = np.linalg.inv(np.dot(K, R))

    # normalize homography
    H = H / H[-1, -1]

    return H


def get_pixel_coordinates(img):
    """
    Get pixel coordinates given an image.

    This funciton was taken from Flamingo.
    ----------
    Args:
        img [Mandatory (np.ndarray)]: NxMx1 or NxMx3 image matrix

    ----------
    Returns:
        U [Mandatory (np.ndarray)]: NxM matrix containing u-coordinates

        V [Mandatory (np.ndarray)]: NxM matrix containing v-coordinates
    """

    # get pixel coordinates
    U, V = np.meshgrid(range(img.shape[1]),
                       range(img.shape[0]))

    return U, V


def rectify_coordinates(U, V, H):
    """
    Get projection of image pixels in real-world coordinates given image
    coordinate matrices and  homography

    This funciton was taken from Flamingo.
    ----------
    Args:
        U [Mandatory (np.ndarray)]: NxM matrix containing u-coordinates

        V [Mandatory (np.ndarray)]: NxM matrix containing v-coordinates

        H [Mandatory (np.ndarray)]: 3x3 homography matrix
    ----------
    Returns:
        X [Mandatory (np.ndarray)]: NxM matrix containing real-world
                                    x-coordinates

        Y [Mandatory (np.ndarray)]: NxM matrix containing real-world
                                    y-coordinates
    """

    UV = np.vstack((U.flatten(),
                    V.flatten())).T

    # transform image using homography
    XY = cv2.perspectiveTransform(np.asarray([UV]).astype(np.float32), H)[0]

    # reshape pixel coordinates back to image size
    X = XY[:, 0].reshape(U.shape[:2])
    Y = XY[:, 1].reshape(V.shape[:2])

    return X, Y


def rectify_image(img, H):
    """
    Get projection of image pixels in real-world coordinates given
    an image and homography

    This funciton was taken from Flamingo.
    ----------
    Args:
    img [Mandatory (np.ndarray)]:  NxMx1 or NxMx3 image matrix

    H [Mandatory (np.ndarray)]: 3x3 homography matrix

    ----------
    Returns:
        X [Mandatory (np.ndarray)]: NxM matrix containing real-world
                                    x-coordinates

        Y [Mandatory (np.ndarray)]: NxM matrix containing real-world
                                    y-coordinates
    """

    U, V = get_pixel_coordinates(img)
    X, Y = rectify_coordinates(U, V, H)

    return X, Y


def rotate_translate(x, y, rotation=None, translation=None):
    """
    Rotate and/or translate coordinate system

    This funciton was taken from Flamingo.
    ----------
    Args:
        X [Mandatory (np.ndarray)]: NxM matrix containing real-world
                                    x-coordinates

        Y [Mandatory (np.ndarray)]: NxM matrix containing real-world
                                    y-coordinates

        rotation [Optional (np.ndarray)]: Rotation angle in degrees

        translation [list or tuple]: 2-tuple or list with x and y translation
                                     distances
    ----------
    Returns:
        X [Mandatory (np.ndarray)]: NxM matrix containing rotated/translated
                                    x-coordinates

        Y [Mandatory (np.ndarray)]: NxM matrix containing rotated/translated
                                    y-coordinates
    """

    if rotation is not None:
        shp = x.shape
        rotation = rotation / 180 * np.pi

        R = np.array([[np.cos(rotation), np.sin(rotation)],
                      [-np.sin(rotation), np.cos(rotation)]])

        xy = np.dot(np.hstack((x.reshape((-1, 1)),
                               y.reshape((-1, 1)))), R)

        x = xy[:, 0].reshape(shp)
        y = xy[:, 1].reshape(shp)

    if translation is not None:
        x += translation[0]
        y += translation[1]

    return x, y


def find_horizon_offset(x, y, max_distance=1e4):
    """
    Find minimum number of pixels to crop to guarantee all pixels are
    within specified distance

    This funciton was taken from Flamingo.
    ----------
    Args:
        X [Mandatory (np.ndarray)]: NxM matrix containing real-world
                                    x-coordinates

        Y [Mandatory (np.ndarray)]: NxM matrix containing real-world
                                    y-coordinates

        max_distance [Optional (float)]: Maximum distance from origin to be
                                         included in the plot. Larger numbers
                                         are considered to be beyond the
                                         horizon.
    ----------
    Returns:
        offset [Optional (float)]: Minimum crop distance in pixels
                                   (from the top of the image)
    """
    offset = 0
    if max_distance is not None:
        try:
            th = (np.abs(x) > max_distance) | (np.abs(y) > max_distance)
            offset = np.max(np.where(np.any(th, axis=1))) + 1
        except Exception:
            pass

    return offset


def construct_rgba_vector(img, n_alpha=0):
    """
    Construct RGBA vector to be used to color faces of pcolormesh

    This funciton was taken from Flamingo.
    ----------
    Args:
        img [Mandatory (np.ndarray)]: NxMx3 RGB image matrix

        n_alpha [Mandatory (float)]: Number of border pixels
                                     to use to increase alpha
    ----------
    Returns:
        rgba [Mandatory (np.ndarray)]: (N*M)x4 RGBA image vector
    """

    alpha = np.ones(img.shape[:2])

    if n_alpha > 0:
        for i, a in enumerate(np.linspace(0, 1, n_alpha)):
            alpha[:, [i, -2-i]] = a

    rgb = img[:, :-1, :].reshape((-1, 3))  # we have 1 less faces than grid
    rgba = np.concatenate((rgb, alpha[:, :-1].reshape((-1, 1))), axis=1)

    if np.any(img > 1):
        rgba[:, :3] /= 255.0

    return rgba


def _construct_rgba_vector(img, n_alpha=0):
    """
    Construct RGBA vector to be used to color faces of pcolormesh

    This funciton was taken from Flamingo.
    ----------
    Args:
        img [Mandatory (np.ndarray)]: NxMx3 RGB image matrix

        n_alpha [Mandatory (float)]: Number of border pixels
                                     to use to increase alpha
    ----------
    Returns:
        rgba [Mandatory (np.ndarray)]: (N*M)x4 RGBA image vector
    """

    alpha = np.ones(img.shape[:2])

    if n_alpha > 0:
        for i, a in enumerate(np.linspace(0, 1, n_alpha)):
            alpha[:, [i, -2-i]] = a

    rgb = img[:, :-1, :].reshape((-1, 3))  # we have 1 less faces than grid
    rgba = np.concatenate((rgb, alpha[:, :-1].reshape((-1, 1))), axis=1)

    if np.any(img > 1):
        rgba[:, :3] /= 255.0

    return rgba


def crop(I, cropfile='crop.txt'):
    """
    Crop image based on a "crop" file.

    The crop file is a text file with 3 lines:
    first line is True or False, second indicates crop in U and third the
    crop in V.

    For example: crop.txt

    True
    400:1200
    300:900

    ----------
    Args:
        I [Mandatory (np.ndarray)]: image array. Use cv or skimage to read the
                                    file

        cropfile [Optional (str)]: filename of the crop file
    ----------
    Returns:
        I [ (np.ndarray)]: Croped array
    """

    # Crop the image
    f = open(cropfile, "r").readlines()
    for line in f:
        line = line.strip("\n")
        if line == "True":
            crp = True
            break
        else:
            crp = False
    if crp:
        u1 = int(f[1].split(":")[0])
        u2 = int(f[1].split(":")[1])
        v1 = int(f[2].split(":")[0])
        v2 = int(f[2].split(":")[1])
        Iimg = I[v1:v2, u1:u2]
    return Iimg
