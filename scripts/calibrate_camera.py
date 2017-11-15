#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
#
# SCRIPT   : calibrate.py
# POURPOSE : Find Intrinsic and Extrinsic camera parameters.
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# V1.0     : 29/07/2016 [Caio Stringari]
#
#
# The algorithm is based on [Zhang2000] and [BouguetMCT]
#
# Zhang. A Flexible New Technique for Camera Calibration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11):1330-1334, 2000.
# Y.Bouguet. MATLAB calibration tool. http://www.vision.caltech.edu/bouguetj/calib_doc/
#
# OBSERVATIONS  : This script has python3 compatibility only
#
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
#
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# system
import os
import sys
import subprocess

# Arguments
import argparse

# files
from glob import glob

# Numpy
import numpy as np

# OpenCV
import cv2

# skimage
import skimage.io

# Matplotlib
import matplotlib.pyplot as plt

# I/O
from pandas import DataFrame
import pickle

import seaborn as sns
sns.set_context("paper", font_scale = 1.5, rc={"lines.linewidth":2.0})
sns.set_style("ticks",   {
                            "axes.linewidth"   : 0.5,
                            "axes.linestyle"   : u"--",
                            'legend.frameon'   : False,
                            'xtick.major.size' : 5.0,
                            'xtick.minor.size' : 0.0,
                            'ytick.major.size' : 5.0,
                            'ytick.minor.size' : 0.0,
                          })


def write_calibration(fname):
    """ Write the camera matrix to file.

    Args:
        fname [Mandatory (str)] Filename.

    Returns:

    """
    f = open('{}.txt'.format(fname),"w")
    f.write("### Camera Calibration Results ###\n")
    f.write("\n")
    f.write("Camera : {}\n".format(camera))
    f.write("\n")
    f.write("Camera Matrix:\n")
    f.write("\n")
    df = DataFrame(mtx)
    df.to_csv(f,mode="a",header=False,index=False,float_format="%.8f")
    f.write("\n")
    # f.write("Camera Matrix file: {}.npy \n".format(fname))
    # np.savez("{}.npy".format(fname),mtx)
    # f.write("\n")
    f.write("Pixel Centers:\n")
    f.write("\n")
    f.write("U0 : {}\n".format(u0))
    f.write("U0 : {}\n".format(v0))
    f.write("\n")
    f.write("Distortion Coeficients:\n")
    f.write("\n")
    f.write("k1 : {}\n".format(k1))
    f.write("k2 : {}\n".format(k2))
    f.write("p1 : {}\n".format(p1))
    f.write("p2 : {}\n".format(p2))
    f.write("k3 : {}\n".format(k3))


def plot_calibration_error(error):
    """ Plot calibration relative and absolute error.

    Args:
        error [Mandatory (np.ndarray, list, tuple)] calculated error.

    Returns:

    """

    # Create a image x-axis
    images = []
    for i,im in enumerate(error):
        images.append("Image {}".format(str(i).zfill(2)))

    # Open the figure
    fig,ax = plt.subplots(figsize=(20,8))
    # Seaborn barplot
    sns.barplot(x=images,y=error, palette="Set3",ax=ax)
    # Average error
    plt.plot(np.ones(len(error))*merror,"--r",label=r"Average Error : {} $[pixels]$".format(merror))
    # rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    # legend
    plt.legend()
    # get rid of some lines
    sns.despine()
    # adjust subplots to look nice
    plt.tight_layout()
    # save
    plt.savefig(camera+".png",dpi=100)
    # show
    plt.show()



def plot_calibration(I,Ic,method="chessboard"):

    fig, [ax1,ax2] = plt.subplots(1,2,figsize=(12,4))
    skimage.io.imshow(I,ax=ax1)
    skimage.io.imshow(Ic,ax=ax2)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Camera name
    parser.add_argument('--camera','-c',
                        nargs = 1,
                        action = 'store',
                        default = 1,
                        dest = 'camera',
                        required = False,
                        help = "Camera name.",)
    # Method to use
    parser.add_argument('--method','-m',
                        nargs = 1,
                        action = 'store',
                        dest = 'method',
                        required = True,
                        help = "Calibration method to be used. Only \'chessboard\' is implemented at the mooment.",)
    # Input images location
    parser.add_argument('--images','-i',
                        nargs = 1,
                        action = 'store',
                        dest = 'images',
                        required = True,
                        help = "Input images folder.",)
    # Input images location
    parser.add_argument('--format','-fmt',
                        nargs = 1,
                        action = 'store',
                        dest = 'format',
                        required = True,
                        help = "Input images format. Usually is \'jpg\', \'JPG\', or \'png\'.",)
# Define the pattern to search
    parser.add_argument('--pattern','-p',
                        nargs = 2,
                        action = 'store',
                        dest = 'pattern',
                        required = True,
                        help = "Number of rows and colums in the chessboard. Ususally 1-ncols and 1-nrows.",)
    # Handle inputs
    args = parser.parse_args()
    # camera name
    camera = args.camera[0]
    # method
    method = args.method[0]
    # input image path
    path = os.path.abspath(args.images[0])

    # pattern
    n1 = int(args.pattern[0])
    n2 = int(args.pattern[1])



    # Read the images according to the chosen method
    if method == "chessboard":
        # Read the imagess
        images =  sorted(glob(path+"/*."+args.format[0]))
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((n1*n2,3), np.float32)
        objp[:,:2] = np.mgrid[0:n2,0:n1].T.reshape(-1,2)
    elif method == "circles":
        images = np.sort(glob('board/*.JPG'))
        raise IOError("Method not Implemented yet.")
    else:
        raise IOError("Method \'{}\'not Implemented.".format(method))

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.0001)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    nrets = 0  # Counter
    # Loop over the images trying to identify the chessboard
    for k,fname in enumerate(images):
        print ("Loading image {} [{} of {}]".format(fname,k+1,len(images)))
        # Read the image as 8 bit
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # skimage.io.imshow(gray); plt.show()

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (n1,n2),None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print ("    Chessboard pattern found in image {}".format(fname))
            # Refine corners
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Append
            objpoints.append(objp)
            imgpoints.append(corners2)
            # Store the last processed images
            I = gray
            nrets+=1
            if nrets > 10: break

    # Calculating the calibration
    h,  w = I.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    # fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(mtx, I.shape, sensor_size_mm, sensor_size_mm)
    # nmtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,nmtx,(w,h),5)

    # Get the intrinsic camera variables [See Holman et al 1997] for terminology
    fx_px = mtx[0,0] # Focal lenght in pixels [x]
    fy_px = mtx[1,1] # Focal lenght in pixels [y]
    # fx_mm = mtx[0,0]*(sensor_size_mm/w) # Focal lenght in mm [x]
    # fy_mm = mtx[1,1]*(sensor_size_mm/h) # Focal lenght in mm [y]
    u0 = mtx[0,2]
    v0 = mtx[1,2]
    k1 = dist[0][0]
    k2 = dist[0][1]
    p1 = dist[0][2]
    p2 = dist[0][3]
    k3 = dist[0][4]
    ud = np.arange(1,w+1,1)
    vd = np.arange(1,h+1,1)

    # Write the parameters to file
    write_calibration(camera)

    # Calculating the error
    error = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error.append(cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2))
    error = np.array(error)
    merror = round(error.mean(),3)
    # plot the calibration error
    plot_calibration_error(error)
