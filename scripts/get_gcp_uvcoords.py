#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
#
# SCRIPT   : get_gcp_coords.py
# POURPOSE : Get GCP coordinates from a image based on mouse click.
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# V1.0     : 01/08/2016 [Caio Stringari]
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

# Arguments
import argparse

# Files
from glob import glob

# Numpy
import numpy as np

# skimage
import skimage.io
import cv2

# Matplotlib
import matplotlib.pyplot as plt

# I/O
from pandas import DataFrame

from videotools import camera_parser


def crop(I,cropfile='crop.txt'):
    """
    Crop image based on a "crop" file. The crop file is a text file with 3 lines:
    first line is True or False, second indicates crop in U and third the crop in V.
    For example: crop.txt
    True
    400:1200
    300:900

    ----------
    Args:
        I [Mandatory (np.ndarray)]: image array. Use cv or skimage to read the file

        cropfile [Optional (str)]: filename of the crop file
    ----------
        Returns:
        I [ (np.ndarray)]: Croped array
    """

    # Crop the image
    f = open(cropfile,"r").readlines()
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
        I = I[v1:v2, u1:u2]
    return I


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    # print 'x = %d, y = %d'%(
    #     ix, iy)

    # assign global variable to access outside of function
    global coords
    coords.append((ix, iy))

    # Disconnect after 6 clicks
    if len(coords) == ngcps:
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return

if __name__ == '__main__':

    print ("\n ### Extracting UV GCP Coordinates ### \n")

    ### Argument parser
    parser = argparse.ArgumentParser()

    # Add an argument to pass one good frame with the GCPS.
    parser.add_argument('--input','-i',
                        nargs = 1,
                        action = 'store',
                        dest = 'input',
            			help = "Filename or folder with frames with a good view of the GCPs.",
            			required = True)
    # Crop file
    parser.add_argument('--crop','-c',
                        nargs = 1,
                        action = 'store',
                        dest = 'crop',
            			help = "Text file containg the crop limits.",
            			required = True)
    # Undistort flag
    parser.add_argument('--undistort','-u',
                        action = 'store_true',
                        dest = 'undistort',
            			help = "Use this option to undistort images based on camera-matrix file.",
            			required = False)
    # Camera matrix
    parser.add_argument('--camera-matrix','-cm',
                        nargs = 1,
                        action = 'store',
                        dest = 'camera',
                        required = False,
                        help = "Camera matrix file. Only used if undistort is True. Please use calibrate.py to generate a valid file.",)
    # Mumber of GPCS to consider
    parser.add_argument('--gcps','-n',
                        nargs = 1,
                        action = 'store',
                        dest = 'gcps',
            			help = "Number of GCPs that can be viewd in a single frame.",
            			required = True)
    # output file
    parser.add_argument('--output','-o',
                        nargs = 1,
                        action = 'store',
                        dest = 'output',
            			help = "Output filename (CSV encoding).",
            			required = True)
    # output file
    parser.add_argument('--show-result','--show','-show',"-s","--s",
                        action = 'store_true',
                        dest = 'show_result',
            			help = "Show the final result.",
            			required = False)
    # Parser
    args = parser.parse_args()

    # Files and Folders
    isfolder = os.path.isdir(args.input[0])

    if isfolder:
        files =sorted(glob(args.input[0]+"/*.jpg"))
        if not files:
            raise IOError("There is no jpgs in {}".format(args.input[0]))
    else:
        jpg = args.input[0]

    # Number of GCPS
    ngcps = int(args.gcps[0])

    # Undistort
    undistort = args.undistort

    # Camera-Matrix
    if undistort:
        try:
            cm = args.camera[0]
            if os.path.isfile(cm):
                K,DC = camera_parser(cm)
        except:
            raise #IOError("If undistort is passed, the user must indicate the camera-matrix file.")

    # Crop file
    fcrop = os.path.abspath(args.crop[0])


    if isfolder:
        U = []
        V = []
        # coords = []
        for jpg in files:
            coords = []

            # read the frame
            I = skimage.io.imread(jpg)
            h, w = I.shape[:2]

            # if undistort is true, undistort
            if undistort:
                Kn,roi = cv2.getOptimalNewCameraMatrix(K,DC,(w,h),1,(w,h))
                I = cv2.undistort(I, K, DC, None, Kn)
            # crop
            I = crop(I,fcrop)

            # GUI
            fig,ax = plt.subplots(figsize=(20,20))
            ax.set_aspect('equal')
            skimage.io.imshow(I,ax=ax)
            # call click func
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            # show the figure
            plt.show()
            # close the figure for good
            # fig.canvas.mpl_disconnect(cid)
            plt.close()

            U.append(coords[0][0])
            V.append(coords[0][1])

        # DataFrame
        U = np.array(U).astype(int)
        V = np.array(V).astype(int)
        df = DataFrame(np.vstack([U,V]).T.astype(int),columns=["u","v"])

        # GCP names - Same order as the mouse clicks
        gcps = []
        for i in range(len(df.index.values)):
            gcps.append("GCP {}".format(str(i+1).zfill(2)))
        # Updated dataframe
        df.index = gcps
        df.index.name = "GCPs"

        # print the dataframe on screen
        print (df)

        # Save the dataframe
        df.to_csv(args.output[0])
    else:
        coords = []
        # read the frame
        I = skimage.io.imread(jpg)
        h, w = I.shape[:2]

        # if undistort is true, undistort
        if undistort:
            Kn,roi = cv2.getOptimalNewCameraMatrix(K,DC,(w,h),1,(w,h))
            I = cv2.undistort(I, K, DC, None, Kn)
        # crop
        I = crop(I,fcrop)

        # GUI
        fig,ax = plt.subplots(figsize=(20,20))
        ax.set_aspect('equal')
        skimage.io.imshow(I,ax=ax)
        # call click func
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        # show the figure
        plt.show()

        # GCP names
        gcps=[]
        for k in np.arange(len(coords))+1:
            gcps.append("GCP {}".format(str(k).zfill(2)))

        # DataFrame
        # U = np.array(U).astype(int)
        # V = np.array(V).astype(int)
        df = DataFrame(np.array(coords).astype(int),columns=["u","v"])
        df.index = gcps
        df.index.name = "GCPs"
        print (df)

        # GCP names - Same order as the mouse clicks
        gcps = []
        for i in range(len(df.index.values)):
            gcps.append("GCP {}".format(str(i+1).zfill(2)))
        # Updated dataframe
        df.index = gcps
        df.index.name = "GCPs"
        # Save the dataframe
        df.to_csv(args.output[0])

    # plot the final result
    if args.show_result:
        fig,ax = plt.subplots(figsize=(20,20))
        ax.set_aspect('equal')
        skimage.io.imshow(I)
        # Scatter the GCPS
        plt.scatter(df.u.values,df.v.values,s=40,c="r",marker="+",linewidths=2)
        # Plot the GCPS names
        for x,y,gcp in zip(df.u.values,df.v.values,gcps):
            t = plt.text(x+30,y,gcp,color="k",fontsize=10)
            t.set_bbox(dict(facecolor=".5",edgecolor="k",alpha=0.85,linewidth=1))
        # Axis
        plt.ylim(I.shape[0],0)
        plt.xlim(0,I.shape[1])
        plt.tight_layout
        # Show the plot
        plt.show()

    print ("\n{} completed, my work is done !\n".format(sys.argv[0]))
