#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
#
# SCRIPT   : windowed_stack.py
# POURPOSE : Created a "windowed" timestack array
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# V1.0     : 06/09/2017 [Caio Stringari]
#
# OBSERVATIOS:
#
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
#
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from __future__ import print_function,division

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
from skimage.morphology import disk

# Matplotlib
import matplotlib.pyplot as plt

# videotools
from videotools import (camera_parser,
                        find_homography,
                        rectify_image,
                        find_horizon_offset,
                        rotate_translate,
                        _construct_rgba_vector,
                        crop,autocrop)

from collections import OrderedDict

#### Defintions

def kdtree(A,pt):
    _,indexes = scipy.spatial.KDTree(A).query(pt)
    return indexes

# def _onclick(event):
#     global ix, iy
#     ix, iy = event.xdata, event.ydata
#     # assign global variable to access outside of function
#     global coords
#     coords.append((ix, iy))
#     # Disconnect after 2 clicks
#     if len(coords) == 2:
#         fig.canvas.mpl_disconnect(cid)
#         plt.close(1)
#     return

def surrounding_pixels(i,j,k=1):

    i1,j1 = i-k,j+k
    i2,j2 = i,j+k
    i3,j3 = i+k,j+k
    i4,j4 = i+k,j
    i5,j5 = i+k,j-k
    i6,j6 = i,j-k
    i7,j7 = i-k,j-k
    i8,j8 = i-k,j

    I = [i1,i2,i3,i4,i5,i6,i7,i8]
    J = [j1,j2,j3,j4,j5,j6,j7,j8]

    return I,J

def pixel_window(a,i,j,s=8):
    
    # compute domain
    i = np.arange(i-s,i+s+1,1)
    j = np.arange(j-s,j+s+1,1)

    # all pixels inside the domain
    I,J = np.meshgrid(i,j)

    # Remove pixels outside the borders
    
    # i-dimension
    I = I.flatten()
    I[I<0] = -999
    I[I>a.shape[0]] = -999
    idx = np.where(I==-999)
    
    # j-dimension
    J = J.flatten()
    J[J<0] = -999
    J[J>a.shape[1]] = -999
    jdx = np.where(J==-999)

    Ifinal = np.delete(I,np.hstack([idx,jdx]))
    Jfinal = np.delete(J,np.hstack([idx,jdx]))


    return Ifinal,Jfinal


if __name__ == '__main__':

    ### Argument parser
    parser = argparse.ArgumentParser()

    # input Frame
    parser.add_argument('--frame','-i',
                        nargs = 1,
                        action = 'store',
                        dest = 'frame',
                        required = True,
                        help = "Input frame.",)
    # crop file
    parser.add_argument('--crop','-crop',
                        nargs = 1,
                        action = 'store',
                        dest= 'crop',
                        required = True,
                        help = "crop file.",)
    # geometry
    parser.add_argument('--gcpxyz','-gcpxyz','--xyz','-xyz',
                        nargs = 1,
                        action = 'store',
                        dest= 'xyzfile',
                        required = True,
                        help = "GCP XYZ file.",)
    parser.add_argument('--gcpuv','-gcpuv','--uv','-uv',
                        nargs = 1,
                        action = 'store',
                        dest= 'uvfile',
                        required = True,
                        help = "GCP UV file. Use get_gcp_uvcoords.py to generate a valid file.",)
    # pressure transducer line
    parser.add_argument('--survey','-survey',
                        nargs = 1,
                        action = 'store',
                        dest= 'ptfile',
                        required = False,
                        default = [None],
                        help = "Pressure transducer survey. Use include_pts.py to generate a valid file.",)
    # number of points in the stack
    parser.add_argument('--stack-points','-stkp',
                        nargs = 1,
                        action = 'store',
                        default = [np.int(300)],
                        dest = 'stackpoints',
                        required = False,
                        help = "Number of points in the timestack.",)
    # camera matrix
    parser.add_argument('--camera-matrix','-cm',
                        nargs = 1,
                        action = 'store',
                        dest = 'camera',
                        required = False,
                        help = "Camera matrix file. Only used if undistort is True. Please use calibrate.py to generate a valid file.",)
    # pixel line coordinates
    parser.add_argument('-x1',
                        nargs = 1,
                        action = 'store',
                        dest = 'x1',
                        required = False,
                        help = "First x-coordinate for the timestack. If passed WILL NOT use matplotlib GUI.",)
    parser.add_argument('-x2',
                        nargs = 1,
                        action = 'store',
                        dest = 'x2',
                        required = False,
                        help = "Final x-coordinate for the timestack. If passed WILL NOT use matplotlib GUI.",)
    parser.add_argument('-y1',
                        nargs = 1,
                        action = 'store',
                        dest = 'y1',
                        required = False,
                        help = "Start y-coordinate for the timestack. If passed WILL NOT use matplotlib GUI.",)
    parser.add_argument('-y2',
                        nargs = 1,
                        action = 'store',
                        dest = 'y2',
                        required = False,
                        help = "Final y-coordinate for the timestack. If passed WILL NOT use matplotlib GUI..",)
    # horizon
    parser.add_argument('--horizon','--hor','-horizon','-hor',
                        nargs = 1,
                        action = 'store',
                        dest = 'horizon',
                        required = False,
                        default = [1000],
                        help = "Maximum distance from origin to be included in the plot.",)
    # Projeciton height
    parser.add_argument('--Z','--z','-Z','-z',
                        nargs = 1,
                        action = 'store',
                        dest = 'Z',
                        required = False,
                        default = [0],
                        help = "Real-world elevation on which the image should be projected",)
    # Rotation
    parser.add_argument('--theta','-theta',
                        nargs = 1,
                        action = 'store',
                        dest = 'theta',
                        required = False,
                        default = [0],
                        help = "Rotation angle. Default is 0.0.",)
    # Translation
    parser.add_argument('--X','--x','-X','-x',
                        nargs = 1,
                        action = 'store',
                        dest = 'X',
                        required = False,
                        default = [0],
                        help = "Translation in the x-direction",)
    parser.add_argument('--Y','--y','-Y','-y',
                        nargs = 1,
                        action = 'store',
                        dest = 'Y',
                        required = False,
                        default = [0],
                        help = "Translation in the x-direction",)
    # Pixel window
    parser.add_argument('--pixel-window','--pxwin','-pxwin','-win',
                        nargs = 1,
                        action = 'store',
                        dest = 'pxwin',
                        required = False,
                        default = [2],
                        help = "Pixel window size. Default is 2.",)

    # parse all the arguments
    args = parser.parse_args()

    img = os.path.isfile(args.frame[0])
    if img:
        Ir = skimage.io.imread(args.frame[0])
        h,  w = Ir.shape[:2]
    else:
        raise ValueError("Could not find input frame.")
  
    # camera matrix
    K,DC = camera_parser(args.camera[0])
    
    # read XYZ coords
    dfxyz = pd.read_csv(args.xyzfile[0])
    XYZ = dfxyz[["x","y","z"]] .values

    gcp_x = XYZ[0,0]
    gcp_y = XYZ[0,1]

    # read UV coords
    dfuv = pd.read_csv(args.uvfile[0])
    UV = dfuv[["u","v"]].values

    # read pressure transducer coordinates
    if args.ptfile[0]:
        PTS = True
        df = pd.read_csv(args.ptfile[0]).dropna(subset=["x_rotated"])
        xy_pts = df[["x_rotated","y_rotated"]].values

    # horizon
    hor = float(args.horizon[0])

    # rotation Angle
    theta = float(args.theta[0])

    # translations
    xt = float(args.X[0])
    yt = float(args.Y[0])

    # projection height
    z = float(args.Z[0])

    # create a timestack line
    if args.x1 and args.x2 and args.y1 and args.y2:
        xstack = [np.float(args.x1[0]),np.float(args.x2[0])]
        ystack = [np.float(args.y1[0]),np.float(args.y2[0])]
        points = np.vstack([xstack,ystack]).T
    else:
        raise ValueError("Sorry, GUI was removed...")

    # pixel window size
    pxwin = np.int(args.pxwin[0])

    # undistort image
    Kn,roi = cv2.getOptimalNewCameraMatrix(K,DC,(w,h),1,(w,h))
    Ir = cv2.undistort(Ir, K, DC, None, Kn)

    # read crop file and crop frame
    Ir = crop(Ir,os.path.abspath(args.crop[0]))

    # homography
    H = find_homography(UV, XYZ, K, z=z, distortion=0)

    # rectify coordinates
    X,Y = rectify_image(Ir, H)

    # find the horizon limits
    horizon = find_horizon_offset(X, Y, max_distance=hor)

    # rotate and translate
    Xr, Yr = rotate_translate(X, Y, rotation=theta, translation=[xt,yt])

    # final arrays
    Xc = Xr[horizon:,:]
    Yc = Yr[horizon:,:]
    Ic = Ir[horizon:,:,:]

    # new image dimensions
    hc, wc = Ic.shape[:2]

    # flattened coordinates
    XYc = np.dstack([Xc.flatten(),Yc.flatten()])[0]

    interp = False
    # interpolate to a regular grid

    # do KDTree in the cliped array
    IDXc = kdtree(XYc,points)

    # points in metric coordinates - cliped array
    xc = XYc[IDXc,0]
    yc = XYc[IDXc,1]

    # points in pixel coordinates - cliped array
    ic = np.unravel_index(IDXc,Xr.shape)[0]
    jc = np.unravel_index(IDXc,Yr.shape)[1]

    # create a line of pixel centers
    i_stack_center = np.linspace(ic[0],ic[1],np.int(args.stackpoints[0])).astype(np.int).tolist()
    j_stack_center = np.linspace(jc[0],jc[1],np.int(args.stackpoints[0])).astype(np.int).tolist()

    # pixel centes in metric coordinates
    x_stack_center = Xc[i_stack_center,j_stack_center]
    y_stack_center = Yc[i_stack_center,j_stack_center]

    
    I = []
    J = []
    R = []
    G = []
    B = []
    # loop over pixel centers
    for i,j in zip(i_stack_center,j_stack_center):
        # find surrounding pixels
        isurrounding,jsurrounding = pixel_window(Ic,i,j,pxwin)
        # final pixel arrays
        iall = np.hstack([isurrounding,i])
        jall = np.hstack([jsurrounding,j])
        # extract pixel stats
        r = Ic[iall,jall,0].mean();R.append(r)
        g = Ic[iall,jall,1].mean();G.append(g)
        b = Ic[iall,jall,2].mean();B.append(b)
        # apppend for plotting
        for val in isurrounding: I.append(val)
        for val in jsurrounding: J.append(val)
    # Final RGB array
    RGB = np.vstack([R,G,B]).astype(np.int).T

    # find surrounding pixels in metric coordinates
    Xpixels = Xc[I,J]
    Ypixels = Yc[I,J]

    # Do a KDtree for surveyed locations
    IDXs = kdtree(XYc,xy_pts)
    # points in metric coordinates - cliped array
    xpts = XYc[IDXs,0]
    ypts = XYc[IDXs,1]
    # points in pixel coordinates - cliped array
    ipts = np.unravel_index(IDXs,Xr.shape)[0]
    jpts = np.unravel_index(IDXs,Yr.shape)[1]

    # # build the output data model
    # ds = xr.Dataset()
    # # write rgb variable
    # ds['rgb'] = (('points','bands'),  RGB) # All bandas
    # # write positional cooridinates
    # ds['x'] = ('points', x_stack_center) # x-coordinate
    # ds['y'] = ('points', y_stack_center) # y-coordinate
    # ds["i"] = ('points', i_stack_center) # central i-coordinate
    # ds["j"] = ('points', j_stack_center) # central j-coordinate
    # # write coordinates
    # ds.coords['time'] = datetime.datetime.now() # camera time
    # ds.coords["points"] = np.arange(0,len(i_stack_center),1)
    # # auxiliary variables
    # ds["bands"] = (('bands'),["red","green","blue"])
    # # ds["indexes"] = (('points'), IDXc)
    # # write to file
    # # ds.to_netcdf("{}/{}.nc".format(tmpfolder,now.strftime(fmt)))

    # plot
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))
    # pixel coords
    ax1.imshow(Ic)
    ax1.scatter(J,I,20,facecolor="darkgreen",edgecolor="k",lw=1)
    ax1.scatter(j_stack_center,i_stack_center,50,facecolor="deepskyblue",edgecolor="w",lw=1)
    ax1.scatter(jc,ic,50,facecolor="orangered",edgecolor="k",lw=1)
    ax1.scatter(jpts,ipts,50,color="red",marker="+",lw=2)
    # metric coords
    im = ax2.pcolormesh(Xc,Yc,Ic.mean(axis=2))
    im.set_array(None); im.set_edgecolor('none')
    im.set_facecolor(_construct_rgba_vector(Ic, n_alpha=0))
    ax2.scatter(Xpixels,Ypixels,20,facecolor="darkgreen",edgecolor="k",lw=1)
    ax2.scatter(x_stack_center,y_stack_center,50,facecolor="deepskyblue",edgecolor="w",lw=1)
    ax2.scatter(xstack,ystack,50,facecolor="orangered",edgecolor="k",lw=1,label="Input")
    ax2.scatter(xpts,ypts,50,color="red",marker="+",lw=2,label="Surveyd")
    ax2.set_xlim(xstack[0]-100,xstack[-1]+100)
    ax2.set_ylim(ystack[0]-100,ystack[-1]+100)
    ax2.legend()
    plt.show()