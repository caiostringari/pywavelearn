# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : build_geometry.py
# POURPOSE : build a gemoetry file with camera and rectification information
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# V1.0     : 06/09/2017 [Caio Stringari]
# V2.0     : 22/01/2018 [Caio Stringari]
#          : - add pywavelearn imports
#
# OBSERVATIOS:
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


# OS
import os
import sys

# dates
import datetime

# arguments
import argparse

# numpy, pandas and xarray
import numpy as np
import pandas as pd
import xarray as xr

# scipy
import scipy.spatial

# Image I/O
import cv2
import skimage.io

# errors
from sklearn.metrics import mean_squared_error

# Matplotlib
import matplotlib.pyplot as plt

# videotools
from pywavelearn.image import (find_homography, rectify_image,
                               find_horizon_offset, rotate_translate,
                               construct_rgba_vector, camera_parser)


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
        return I[v1:v2, u1:u2]
    else:
        return I


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()

    # input Frame
    parser.add_argument('--frame', '-i',
                        nargs=1,
                        action='store',
                        dest='frame',
                        required=True,
                        help="Input frame.",)
    # output netcdf
    parser.add_argument('--output', '-o',
                        nargs=1,
                        action='store',
                        dest='output',
                        required=False,
                        default=["geom.nc"],
                        help="Output netCDF.",)
    # crop file
    parser.add_argument('--crop', '-crop',
                        nargs=1,
                        action='store',
                        dest='crop',
                        required=True,
                        help="crop file.",)
    # geometry
    parser.add_argument('--gcpxyz', '-gcpxyz', '--xyz', '-xyz',
                        nargs=1,
                        action='store',
                        dest='xyzfile',
                        required=True,
                        help="GCP XYZ file.",)
    parser.add_argument('--gcpuv', '-gcpuv', '--uv', '-uv',
                        nargs=1,
                        action='store',
                        dest='uvfile',
                        required=True,
                        help="""GCP UV file. Use get_gcp_uvcoords.py to generate
a valid file.""",)
    # camera matrix
    parser.add_argument('--camera-matrix', '-cm',
                        nargs=1,
                        action='store',
                        dest='camera',
                        required=False,
                        help="""Camera matrix file. Only used if undistort is
True. Please use calibrate.py to generate a valid file.""",)
    # horizon
    parser.add_argument('--horizon', '--hor', '-horizon', '-hor',
                        nargs=1,
                        action='store',
                        dest='horizon',
                        required=False,
                        default=[1000],
                        help="""Maximum distance from origin to be
included in the plot.""",)
    # Projeciton height
    parser.add_argument('--Z', '--z', '-Z', '-z',
                        nargs=1,
                        action='store',
                        dest='Z',
                        required=False,
                        default=[0],
                        help="""Real-world elevation on which the image
should be projected.""",)
    # Rotation
    parser.add_argument('--theta', '-theta',
                        nargs=1,
                        action='store',
                        dest='theta',
                        required=False,
                        default=[0],
                        help="Rotation angle. Default is 0.0.",)
    # Translation
    parser.add_argument('--X', '--x', '-X', '-x',
                        nargs=1,
                        action='store',
                        dest='X',
                        required=False,
                        default=[0],
                        help="Translation in the x-direction",)
    parser.add_argument('--Y', '--y', '-Y', '-y',
                        nargs=1,
                        action='store',
                        dest='Y',
                        required=False,
                        default=[0],
                        help="Translation in the x-direction",)
    # Pixel window
    parser.add_argument('--pixel-window', '--pxwin', '-pxwin', '-win',
                        nargs=1,
                        action='store',
                        dest='pxwin',
                        required=False,
                        default=[2],
                        help="Pixel window size. Default is 2.",)

    # parse all the arguments
    args = parser.parse_args()

    img = os.path.isfile(args.frame[0])
    if img:
        Ir = skimage.io.imread(args.frame[0])
        h,  w = Ir.shape[:2]
    else:
        raise ValueError("Could not find input frame.")

    # camera matrix
    K, DC = camera_parser(args.camera[0])

    # read XYZ coords
    dfxyz = pd.read_csv(args.xyzfile[0])
    XYZ = dfxyz[["x", "y", "z"]] .values

    gcp_x = XYZ[0, 0]
    gcp_y = XYZ[0, 1]

    # read UV coords
    dfuv = pd.read_csv(args.uvfile[0])
    UV = dfuv[["u", "v"]].values

    # horizon
    hor = float(args.horizon[0])

    # rotation Angle
    theta = float(args.theta[0])

    # translations
    xt = float(args.X[0])
    yt = float(args.Y[0])

    # projection height
    z = float(args.Z[0])

    # undistort image
    Kn, roi = cv2.getOptimalNewCameraMatrix(K, DC, (w, h), 1, (w, h))
    Ir = cv2.undistort(Ir, K, DC, None, Kn)

    # read crop file and crop frame
    Ir = crop(Ir, os.path.abspath(args.crop[0]))

    # homography
    H = find_homography(UV, XYZ, K, z=z, distortion=0)

    # rectify coordinates
    X, Y = rectify_image(Ir, H)

    # find the horizon limits
    horizon = find_horizon_offset(X, Y, max_distance=hor)

    # rotate and translate
    Xr, Yr = rotate_translate(X, Y, rotation=theta, translation=[xt, yt])

    # final arrays
    Xc = Xr  # [horizon:,:]
    Yc = Yr  # [horizon:,:]
    Ic = Ir  # [horizon:,:,:]

    # new image dimensions
    hc, wc = Ic.shape[:2]

    # flattened coordinates
    XYc = np.dstack([Xc.flatten(), Yc.flatten()])[0]

    # build the output data model
    ds = xr.Dataset()
    # write rgb variable
    ds['rgb'] = (('u', 'v', 'bands'),  Ir)  # ALl bandas
    # write camera matrix
    ds["camera_matrix"] = K.flatten()
    # write distortions
    ds["distortion_coeffs"] = DC
    # write homography
    ds["homography"] = H.flatten()
    # write shapes
    # ds["k_h_shapes"] = H.shape
    # write positional cooridinates
    ds.coords['X'] = (('u', 'v'), Xr)  # record X coordinate
    ds.coords['Y'] = (('u', 'v'), Yr)  # record Y coordinate
    # # auxiliary variables
    ds["bands"] = (('bands'), ["red", "green", "blue"])
    # write to file
    ds.to_netcdf(args.output[0])

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    # pixel coords
    ax1.imshow(Ic)
    # metric coords
    im = ax2.pcolormesh(Xc, Yc, Ic.mean(axis=2))
    im.set_array(None)
    im.set_edgecolor('none')
    im.set_facecolor(construct_rgba_vector(Ic, n_alpha=0))
    ax2.set_aspect("equal")
    # show
    plt.show()
