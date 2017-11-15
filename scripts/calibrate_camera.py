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
#------------------------------------------------------------------------
#------------------------------------------------------------------------

# system
import os

# arguments
import argparse

# files
from glob import glob

# numpy
import numpy as np

# OpenCV
import cv2

# I/O
from pandas import DataFrame

# Matplotlib
import matplotlib.pyplot as plt

def main():
    # camera name
    camera = args.camera[0]
   
    # input image path
    path = os.path.abspath(args.images[0])

    # pattern
    n1 = int(args.pattern[0])
    n2 = int(args.pattern[1])

    images =  sorted(glob(path+"/*."+args.format[0]))
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((n1*n2,3), np.float32)
    objp[:,:2] = np.mgrid[0:n2,0:n1].T.reshape(-1,2)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.0001)

    # arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    nrets = 0  # counter
    # loop over the images trying to identify the chessboard
    for k,fname in enumerate(images):
        print ("Loading image {} of {}".format(k+1,len(images)))
        # read the image as 8 bit
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (n1,n2),None)
        # if found, add object points, image points (after refining them)
        if ret == True:
            print ("    Chessboard pattern found in image {} of {}".format(k+1,len(images)))
            # refine corners
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # append
            objpoints.append(objp)
            imgpoints.append(corners2)
            # store the last processed images
            I = gray
            nrets+=1
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (n1,n2), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
            if nrets > 10: break
    cv2.destroyAllWindows()

    # calculating the calibration
    h,  w = I.shape[:2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    # get the intrinsic camera variables [See Holman et al 1997] for terminology
    fx_px = mtx[0,0] # Focal lenght in pixels [x]
    fy_px = mtx[1,1] # Focal lenght in pixels [y]
    u0 = mtx[0,2]
    v0 = mtx[1,2]
    k1 = dist[0][0]
    k2 = dist[0][1]
    p1 = dist[0][2]
    p2 = dist[0][3]
    k3 = dist[0][4]
    ud = np.arange(1,w+1,1)
    vd = np.arange(1,h+1,1)

    # calculating the error
    error = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error.append(cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2))
    error = np.array(error)
    merror = round(error.mean(),3)

    print ("Mean error is {} pixels".format(merror))

    # write the parameters to file
    f = open(args.output[0],"w")
    f.write("### Camera Calibration Results ###\n")
    f.write("\n")
    f.write("Camera : {}\n".format(camera))
    f.write("\n")
    f.write("Camera Matrix:\n")
    f.write("\n")
    df = DataFrame(mtx)
    df.to_csv(f,mode="a",header=False,index=False,float_format="%.4f")
    f.write("\n")
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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Camera name
    parser.add_argument('--camera','-c',
                        nargs = 1,
                        action = 'store',
                        default = ["My Camera"],
                        dest = 'camera',
                        required = False,
                        help = "Camera name.",)
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
    # Output file
    parser.add_argument('--output','-o',
                        nargs = 1,
                        action = 'store',
                        dest = 'output',
                        required = True,
                        help = "Output file name.",)
    # handle inputs
    args = parser.parse_args()

    # main call
    main() 


   