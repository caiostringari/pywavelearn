#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
#
# SCRIPT   : pixelline.py
# POURPOSE : Auxiliary script to verify geometries.
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# V1.0     : 02/08/2016 [Caio Stringari]
# V1.0     : 12/10/2016 [Caio Stringari]
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

# Os
import sys,os,subprocess

import datetime

# Arguments
import argparse

# files
from glob import glob

# numpy
import numpy as np
import pandas as pd
import scipy.spatial

# Image IO
import cv2
import skimage.io
import xarray

# errors
from sklearn.metrics import mean_squared_error

# Matplotlib
import matplotlib.pyplot as plt

# videotools
from videotools import (camera_parser,
                        find_homography,
                        rectify_image,
                        find_horizon_offset,
                        rotate_translate,
                        _construct_rgba_vector,
                        crop)

from collections import OrderedDict

#### Defintions

def kdtree(A,pt):
    _,indexes = scipy.spatial.KDTree(A).query(pt)
    return indexes

def _onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    # assign global variable to access outside of function
    global coords
    coords.append((ix, iy))
    # Disconnect after 2 clicks
    if len(coords) == 2:
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return

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
    parser.add_argument('--pressure-transducers','-pts',
                    nargs = 1,
                    action = 'store',
                    dest= 'ptfile',
                    required = False,
                    default = [None],
                    help = "Pressure transducer coordinates. Use include_pts.py to generate a valid file.",)
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

    # parse all the arguments
    args = parser.parse_args()

    img = os.path.isfile(args.frame[0])
    if img:
        Im = skimage.io.imread(args.frame[0])
        h,  w = Im.shape[:2]
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

    # sys.exit()


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
    Im = cv2.undistort(Im, K, DC, None, Kn)

    # read crop file and crop frame
    Im = crop(Im,os.path.abspath(args.crop[0]))

    # homography
    H = find_homography(UV, XYZ, K, z=z, distortion=0)

    # Rectify coordinates
    X,Y = rectify_image(Im, H)

    # find the horizon limits
    horizon = find_horizon_offset(X, Y, max_distance=hor)

    # rotate and translate
    xr, yr = rotate_translate(X, Y, rotation=theta, translation=[xt,yt])

    # final coordinates array
    Xf = xr[horizon:,:]
    Yf = yr[horizon:,:]
    If = Im[horizon:,:,:]

    # flatten coordiantes
    XY1 = np.dstack([Xf.flatten(),Yf.flatten()])[0]
    XY2 = np.dstack([xr.flatten(),yr.flatten()])[0]

    # build a netcdf model
    ds = xarray.Dataset()
    # all rgb bands
    ds['rgb'] = (('u', 'v','bands'),  If) 
    # coordianates
    ds.coords['X'] = (('u', 'v'), Xf) # record X coordinate
    ds.coords['Y'] = (('u', 'v'), Yf) # record Y coordinate
    ds.coords['time'] = datetime.datetime.now() # record time

    ds.to_netcdf(args.frame[0].replace(".jpg",".nc"))


    # build timestack line without GUI
    if args.x1 and args.x2 and args.y1 and args.y2:
        x1 = float(args.x1[0]);  x2 = float(args.x2[0]);
        y1 = float(args.y1[0]);  y2 = float(args.y2[0])
    # start GUI
    else:
        coords=[]
        fig,ax = plt.subplots(figsize=(20,20))
        im=plt.pcolormesh(Xf,Yf,np.mean(If,axis=2))
        plt.scatter(XYZ[:,0],XYZ[:,1],100,marker="x",color="r",lw=2)
        rgba = _construct_rgba_vector(If, n_alpha=0)
        im.set_array(None) # remove the array
        im.set_edgecolor('none')
        im.set_facecolor(rgba)
        ax.set_aspect("equal")
        cid = fig.canvas.mpl_connect('button_press_event', _onclick)
        plt.show(1)
        plt.close(1)
        # organizing data
        x1 = coords[0][0];  x2 = coords[1][0];
        y1 = coords[0][1];  y2 = coords[1][1]


    # organize timestack line
    xline = np.linspace(x1,x2,300)
    yline = np.linspace(y1,y2,300)
    points =  np.vstack([xline,yline]).T
    
    
    # # do a KDTree search for timestack points in XYZ space
    # idxs = kdtree(XY2,points)
    
    # # select the points in the UV space
    # ui = np.unravel_index(idxs,Xf.shape)[0]
    # vi = np.unravel_index(idxs,Yf.shape)[1]
    # # select the points in the uv space
    # dsp = ds.isel_points(u=ui,v=vi)


    # do a kdtree search for GCPs points
    dist, idx = scipy.spatial.KDTree(XY2).query(XYZ[:,0:2],k=1)
    # metric coordinates
    x_gcp = XY2[idx,0]
    y_gcp = XY2[idx,1]
    # pixel coordinates
    pxi_gcp = np.unravel_index(idx,xr.shape)[0]
    pxj_gcp = np.unravel_index(idx,xr.shape)[1]

    # do a kdtree search for timestack line points
    dist, idx = scipy.spatial.KDTree(XY2).query(points,k=1)
    # metric coordinates
    x_stk = XY2[idx,0]
    y_stk = XY2[idx,1]
    # pixel coordinates
    pxi_stk = np.unravel_index(idx,xr.shape)[0]
    pxj_stk = np.unravel_index(idx,xr.shape)[1]

    # do a kdtree search for PTS points
    if PTS:    
        dist, idx = scipy.spatial.KDTree(XY2).query(xy_pts,k=1)
        # metric coordinates
        x_pts = XY2[idx,0]
        y_pts = XY2[idx,1]
        # # pixel coordinates
        # pxi_gcp = np.unravel_index(idx,xr.shape)[0]
        # pxj_gcp = np.unravel_index(idx,xr.shape)[1]


    # Calculate errors
    xerror = np.round(np.sqrt(mean_squared_error(XYZ[:,0],x_gcp)),2)
    yerror = np.round(np.sqrt(mean_squared_error(XYZ[:,1],y_gcp)),2)

    print ("\n---------------------------------")
    print ("  Error report :\n")
    print ("   RMSE  x-coordinate : {} m".format(xerror))
    print ("   RMSE  y-coordinate : {} m".format(yerror))
    print ("---------------------------------\n")




    # Plot recfication in pixel in metric coordinates
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))

    
    # plot retification in metric coordinates    
    im = ax1.pcolormesh(ds["X"].values,ds["Y"].values,ds["rgb"].values.mean(axis=2))
    im.set_array(None); im.set_edgecolor('none'); im.set_facecolor(_construct_rgba_vector(ds["rgb"].values, n_alpha=0))
    
    # plot timestack line
    ax1.plot(xline,yline,color="k",lw=2,ls="-",label="Input Timestack line")
    ax1.plot(x_stk,y_stk,color="r",lw=2,ls="--",label="Predicted Timestack line")
    
    # plot input and predicted GCPs in metric coordinates
    ax1.scatter(XYZ[:,0],XYZ[:,1],50,marker="+",color="red",label="input")
    ax1.scatter(x_gcp,y_gcp,50,marker="+",color="cyan",label="predicted")

    # plot plots in metric coordinates
    if PTS:
        # surveyed pts
        ax1.scatter(xy_pts[:,0],xy_pts[:,1],50,marker="o",facecolor="skyblue",edgecolor="k",lw=1,label="Surveyed PTs")
        # predicted pts
        ax1.scatter(x_pts,y_pts,50,marker="o",facecolor="orangered",edgecolor="k",lw=1,label="Predicted PTs")


    # adjust axis
    ax1.grid(ls="--")
    ax1.legend(loc="best")
    ax1.set_xlabel(r"x $[m]$")
    ax1.set_ylabel(r"y $[m]$")
    ax1.set_xlim(xline[0]-100,xline[-1]+100)
    ax1.set_ylim(yline[0]-100,yline[-1]+100)


    ax2.imshow(Im)
    ax2.scatter(pxj_stk,pxi_stk,50,color="r")


    plt.show()



    # dist = np.sqrt(((x2-x1)*(x2-x1))+((y2-y1)*(y2-y1)))
    # x0=dsp.X.values.min()
    # y0=dsp.Y.values.min()
    # x1=dsp.X.values.max()
    # y1=dsp.Y.values.max()
    # # dist = np.sqrt(((x1-x0)*(x1-x0))+((y1-y0)*(y1-y0)))

    # print ("\n +++ Pixel line +++ \n")

    # print ("x1 : ",x1)
    # print ("x2 : ",x2)
    # print ("y1 : ",y1)
    # print ("y2 : ",y2)
    # print ("\nTotal distance : {} (m)".format(round(dist,2)))
