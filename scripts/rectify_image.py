#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
#
# SCRIPT   : rectify_image.py
# POURPOSE : rectify image CLI
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# V1.0     : 15/11/2017 [Caio Stringari]
#
# USAGE    : rectify_image.py -i '../data/Image/OMB.jpg' -xyz '../data/Image/xyz.csv' 
#                                                        -uv "../data/Image/uv.csv" 
#                                                        -cm '../data/Image/camera.txt' 
#                                                        -hor 150 --show --output '../data/Image/OMB.nc'
#
#------------------------------------------------------------------------
#------------------------------------------------------------------------
from __future__ import print_function,division

# OS
import os

# arguments
import argparse

# numpy, pandas and xarray
import numpy as np
import pandas as pd
import xarray as xr

# image processing
import cv2
import skimage.io
import pywavelearning.image as ipwl

# matplotlib
import matplotlib.pyplot as plt

def main():

    img = os.path.isfile(args.frame[0])
    if img:
        Ir = skimage.io.imread(args.frame[0])
        h,  w = Ir.shape[:2]
    else:
        raise ValueError("Could not find input frame.")
  
    # camera matrix
    K,DC = ipwl.camera_parser(args.camera[0])
    
    # read XYZ coords
    dfxyz = pd.read_csv(args.xyzfile[0])
    XYZ = dfxyz[["x","y","z"]] .values

    gcp_x = XYZ[0,0]
    gcp_y = XYZ[0,1]

    # read UV coords
    dfuv = pd.read_csv(args.uvfile[0])
    UV = dfuv[["u","v"]].values

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
    Kn,roi = cv2.getOptimalNewCameraMatrix(K,DC,(w,h),1,(w,h))
    Ir = cv2.undistort(Ir, K, DC, None, Kn)

    # homography
    H = ipwl.find_homography(UV, XYZ, K, z=z, distortion=0)

    # rectify coordinates
    X,Y = ipwl.rectify_image(Ir, H)

    # find the horizon limits
    horizon = ipwl.find_horizon_offset(X, Y, max_distance=hor)

    # rotate and translate
    Xr, Yr = ipwl.rotate_translate(X, Y, rotation=theta, translation=[xt,yt])

    # final arrays
    if hor == -999:
        Xc = Xr
        Yc = Yr
        Ic = Ir
    else:
        Xc = Xr[horizon:,:]
        Yc = Yr[horizon:,:]
        Ic = Ir[horizon:,:,:]

    # new image dimensions
    hc, wc = Ic.shape[:2]

    # flattened coordinates
    XYc = np.dstack([Xc.flatten(),Yc.flatten()])[0]

    # build the output data model
    ds = xr.Dataset()
    # write rgb variable
    ds['rgb'] = (('u', 'v','bands'),  Ic) # ALl bandas
    # write camera matrix
    ds["camera_matrix"] = K.flatten()
    # write distortions
    ds["distortion_coeffs"] = DC
    # write homography
    ds["homography"] = H.flatten()
    # write shapes
    # ds["k_h_shapes"] = H.shape
    # write positional cooridinates
    ds.coords['X'] = (('u', 'v'), Xc) # record X coordinate
    ds.coords['Y'] = (('u', 'v'), Yc) # record Y coordinate
    # # auxiliary variables
    ds["bands"] = (('bands'),["red","green","blue"])
    # write to file
    ds.to_netcdf(args.output[0])

    if args.show:
        # plot
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))
        # pixel coords
        ax1.imshow(Ic)
        # metric coords
        im = ax2.pcolormesh(Xc,Yc,Ic.mean(axis=2))
        im.set_array(None); im.set_edgecolor('none')
        im.set_facecolor(ipwl.construct_rgba_vector(Ic, n_alpha=0))
        plt.show()

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
    # camera matrix
    parser.add_argument('--camera-matrix','-cm',
                        nargs = 1,
                        action = 'store',
                        dest = 'camera',
                        required = False,
                        help = "Camera matrix file. Only used if undistort is True. Please use calibrate.py to generate a valid file.",)
    # horizon
    parser.add_argument('--horizon','--hor','-horizon','-hor',
                        nargs = 1,
                        action = 'store',
                        dest = 'horizon',
                        required = False,
                        default = [1000],
                        help = "Maximum distance from origin to be included in the plot. Use -990 to ignore explicitly not use it.",)
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
    # Output
    parser.add_argument('--output','--o','-o','-output',
                        nargs = 1,
                        action = 'store',
                        dest = 'output',
                        required = False,
                        default = ["rectified.nc"],
                        help = "Output netCDF file name.",)
    # Show results
    parser.add_argument('--show','--show-results','-show','-show-results',
                        action = 'store_true',
                        default = ["show"],
                        help = "Show the results. Requires a valid $DISPLAY.",)

    # parse all the arguments
    args = parser.parse_args()

    # call main script
    main()

