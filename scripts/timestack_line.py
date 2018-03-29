# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : timestack_line.py
# POURPOSE : Plot the timestack line
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# V1.0     : 02/08/2016 [Caio Stringari]
# V1.0     : 12/10/2016 [Caio Stringari]
#
# OBSERVATIOS:
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# Os
import sys
import os

import datetime

# Arguments
import argparse

# Files
from glob import glob

# Numerical
import numpy as np
import scipy.spatial

# Data IO
import xarray
import pandas as pd

# Image processing
import cv2
import skimage.io
import pywavelearn.image as ipwl

# Matplotlib
import matplotlib.pyplot as plt


def main():

    # read frame
    img = os.path.isfile(args.frame[0])
    if img:
        Ir = skimage.io.imread(args.frame[0])
        h, w = Ir.shape[:2]
    else:
        raise ValueError("Could not find input frame.")

    # camera matrix
    K, DC = ipwl.camera_parser(args.camera[0])

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

    # homography
    H = ipwl.find_homography(UV, XYZ, K, z=z, distortion=0)

    # rectify coordinates
    X, Y = ipwl.rectify_image(Ir, H)

    # find the horizon limits
    horizon = ipwl.find_horizon_offset(X, Y, max_distance=hor)

    # rotate and translate
    Xr, Yr = ipwl.rotate_translate(X, Y, rotation=theta, translation=[xt, yt])

    # final arrays
    if hor == -999:
        Xc = Xr
        Yc = Yr
        Ic = Ir
    else:
        Xc = Xr[horizon:, :]
        Yc = Yr[horizon:, :]
        Ic = Ir[horizon:, :, :]

    # new image dimensions
    hc, wc = Ic.shape[:2]

    # flattened coordinates
    XYc = np.dstack([Xc.flatten(), Yc.flatten()])[0]

    # build timestack line without GUI
    if args.x1 and args.x2 and args.y1 and args.y2:
        x1 = float(args.x1[0])
        x2 = float(args.x2[0])
        y1 = float(args.y1[0])
        y2 = float(args.y2[0])
    # start GUI
    else:
        # coords=[]
        plt.ion()
        fig, ax = plt.subplots()
        im = plt.pcolormesh(Xc, Yc, np.mean(Ic, axis=2))
        rgba = ipwl.construct_rgba_vector(Ic, n_alpha=0)
        im.set_array(None)  # remove the array
        im.set_edgecolor('none')
        im.set_facecolor(rgba)
        ax.set_aspect("equal")
        points = plt.ginput(n=2, timeout=100, show_clicks=True)
        # cid = fig.canvas.mpl_connect('button_press_event', _onclick)
        plt.show()
        plt.close()
        # organizing data
        x1 = points[0][0]
        x2 = points[1][0]
        y1 = points[0][1]
        y2 = points[1][1]

    # build the timestack line
    npoints = np.int(args.stackpoints[0])
    xline = np.linspace(x1, x2, npoints)
    yline = np.linspace(y1, y2, npoints)
    points = np.vstack([xline, yline]).T

    # do KDTree in the cliped array
    _, IDXc = scipy.spatial.KDTree(XYc).query(points)

    # points in metric coordinates
    xc = XYc[IDXc, 0]
    yc = XYc[IDXc, 1]

    # create a line of pixel centers
    i_stack_center = np.unravel_index(IDXc, Xr.shape)[0]
    j_stack_center = np.unravel_index(IDXc, Yr.shape)[1]

    # pixel centes in metric coordinates
    x_stack_center = Xc[i_stack_center, j_stack_center]
    y_stack_center = Yc[i_stack_center, j_stack_center]

    # loop over pixel centers and get all pixels insed the window
    Ipx = []
    Jpx = []
    win = int(args.window[0])
    for i, j in zip(i_stack_center, j_stack_center):
        # find surrounding pixels
        isurrounding, jsurrounding = ipwl.pixel_window(Ic, i, j, win, win)
        # final pixel arrays
        iall = np.hstack([isurrounding, i])
        jall = np.hstack([jsurrounding, j])
        # apppend for plotting
        for val in isurrounding:
            Ipx.append(val)
        for val in jsurrounding:
            Jpx.append(val)

    # translate to metric coordinates
    Xpixels = Xc[Ipx, Jpx]
    Ypixels = Yc[Ipx, Jpx]

    # plot
    if args.show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        # pixel coords
        ax1.imshow(Ic)
        ax1.scatter(Jpx, Ipx, 20, facecolor="darkgreen", edgecolor="k", lw=1)
        ax1.scatter(
            j_stack_center,
            i_stack_center,
            50,
            facecolor="deepskyblue",
            edgecolor="w",
            lw=1)
        # metric coords
        im = ax2.pcolormesh(Xc, Yc, Ic.mean(axis=2))
        im.set_array(None)
        im.set_edgecolor('none')
        im.set_facecolor(ipwl.construct_rgba_vector(Ic, n_alpha=0))
        ax2.scatter(
            Xpixels,
            Ypixels,
            20,
            facecolor="darkgreen",
            edgecolor="k",
            lw=1)
        ax2.scatter(
            x_stack_center,
            y_stack_center,
            50,
            facecolor="deepskyblue",
            edgecolor="w",
            lw=1)
        ax2.set_xlim(-200, 200)
        ax2.set_ylim(-200, 200)
        ax2.set_aspect("equal")
        # ax2.legend()
        plt.show()


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser()

    # Input Frame
    parser.add_argument(
        '--frame', '-i',
        nargs=1,
        action='store',
        dest='frame',
        required=True,
        help="Input frame.",)
    # Geometry
    parser.add_argument(
        '--gcpxyz', '-gcpxyz', '--xyz', '-xyz',
        nargs=1,
        action='store',
        dest='xyzfile',
        required=True,
        help="GCP XYZ file.",)
    parser.add_argument(
        '--gcpuv',
        '-gcpuv',
        '--uv',
        '-uv',
        nargs=1,
        action='store',
        dest='uvfile',
        required=True,
        help="GCP UV file. Use get_gcp_uvcoords.py to generate a valid file.",)
    # Camera matrix
    parser.add_argument(
        '--camera-matrix',
        '-cm',
        nargs=1,
        action='store',
        dest='camera',
        required=False,
        help="Camera matrix file. Only used if undistort is True."
             "Please use calibrate.py to generate a valid file.")
    # Horizon
    parser.add_argument(
        '--horizon',
        '--hor',
        '-horizon',
        '-hor',
        nargs=1,
        action='store',
        dest='horizon',
        required=False,
        default=[1000],
        help="Maximum distance from origin to be included in the plot."
             "Use -990 to ignore explicitly not use it.",)
    # Projeciton height
    parser.add_argument(
        '--Z',
        '--z',
        '-Z',
        '-z',
        nargs=1,
        action='store',
        dest='Z',
        required=False,
        default=[0],
        help="Real-world elevation on which the image should be projected",)
    # Rotation
    parser.add_argument(
        '--theta', '-theta',
        nargs=1,
        action='store',
        dest='theta',
        required=False,
        default=[0],
        help="Rotation angle. Default is 0.0.",)
    # Translation
    parser.add_argument(
        '--X', '--x', '-X', '-x',
        nargs=1,
        action='store',
        dest='X',
        required=False,
        default=[0],
        help="Translation in the x-direction",)
    parser.add_argument(
        '--Y', '--y', '-Y', '-y',
        nargs=1,
        action='store',
        dest='Y',
        required=False,
        default=[0],
        help="Translation in the x-direction",)
    # number of points in the stack
    parser.add_argument(
        '--stack-points', '-stkp',
        nargs=1,
        action='store',
        default=[np.int(256)],
        dest='stackpoints',
        required=False,
        help="Number of points in the timestack.",)
    # number of points in the window
    parser.add_argument(
        '--window',
        '-win',
        nargs=1,
        action='store',
        default=[0],
        dest='window',
        required=False,
        help="Number of points in the timestack point window.",
    )
    # pixel line coordinates
    parser.add_argument(
        '-x1',
        nargs=1,
        action='store',
        dest='x1',
        required=False,
        help="First x-coordinate for the timestack."
             "If passed WILL NOT use matplotlib GUI.")
    parser.add_argument(
        '-x2',
        nargs=1,
        action='store',
        dest='x2',
        required=False,
        help="Final x-coordinate for the timestack."
             "If passed WILL NOT use matplotlib GUI.")
    parser.add_argument(
        '-y1',
        nargs=1,
        action='store',
        dest='y1',
        required=False,
        help="Start y-coordinate for the timestack."
             "If passed WILL NOT use matplotlib GUI.")
    parser.add_argument(
        '-y2',
        nargs=1,
        action='store',
        dest='y2',
        required=False,
        help="Final y-coordinate for the timestack."
             "If passed WILL NOT use matplotlib GUI..")
    # Show results
    parser.add_argument(
        '--show', '--show-results', '-show', '-show-results',
        action='store_true',
        default=["show"],
        help="Show the results. Requires a valid $DISPLAY.",)
    # parse all the arguments
    args = parser.parse_args()
    # main call
    main()
