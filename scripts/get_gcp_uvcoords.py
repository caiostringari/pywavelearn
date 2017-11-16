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
# V1.1     : 16/11/2017 [Caio Stringari]
#
#------------------------------------------------------------------------
#------------------------------------------------------------------------

# system
import os
import sys

# Arguments
import argparse

# Files
from glob import glob

# numpy
import numpy as np

# image processing
import cv2
import skimage.io

# Matplotlib
import matplotlib.pyplot as plt

# I/O
from pandas import DataFrame

from pywavelearning.image import camera_parser


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

    print ("\nExtracting UV GCP Coordinates\n")

    ### Argument parser
    parser = argparse.ArgumentParser()

    # Add an argument to pass one good frame with the GCPS.
    parser.add_argument('--input','-i',
                        nargs = 1,
                        action = 'store',
                        dest = 'input',
            			help = "Filename or folder with frames with a good view of the GCPs.",
            			required = True)
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

    # can't use "def main():" for this script, it does not work with on_click() =[

    # files and Folders
    isfolder = os.path.isdir(args.input[0])

    if isfolder:
        files =sorted(glob(args.input[0]+"/*.jpg"))
        if not files:
            raise IOError("There is no jpgs in {}".format(args.input[0]))
    else:
        jpg = args.input[0]

    # number of GCPS
    ngcps = int(args.gcps[0])

    # camera matrix
    cm = args.camera[0]
    if os.path.isfile(cm):
        K,DC = camera_parser(cm)
    else:
        raise IOError("Cannot fint the camera-matrix file.")

    # folder case
    if isfolder:

        # intiate
        U = []
        V = []
        for jpg in files:
            coords = []
            # read the frame
            I = skimage.io.imread(jpg)
            h, w = I.shape[:2]
            # if undistort is true, undistort
            Kn,roi = cv2.getOptimalNewCameraMatrix(K,DC,(w,h),1,(w,h))
            I = cv2.undistort(I, K, DC, None, Kn)
            # Start GUI
            fig,ax = plt.subplots(figsize=(20,20))
            ax.set_aspect('equal')
            skimage.io.imshow(I,ax=ax)
            # call click func
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            # show the figure
            plt.show()            
            plt.close()
            # append to output
            U.append(coords[0][0])
            V.append(coords[0][1])
        # output 
        U = np.array(U).astype(int)
        V = np.array(V).astype(int)
        df = DataFrame(np.vstack([U,V]).T.astype(int),columns=["u","v"])
        # update names in the same order as the clicks
        gcps = []
        for i in range(len(df.index.values)):
            gcps.append("GCP {}".format(str(i+1).zfill(2)))
        # updated dataframe
        df.index = gcps
        df.index.name = "GCPs"
        # print the dataframe on screen
        print ("\n")
        print (df)
        # dump to .csv
        df.to_csv(args.output[0])

    # single image case
    else:
        coords = []
        # read the frame
        I = skimage.io.imread(jpg)
        h, w = I.shape[:2]
        # undistort
        Kn,roi = cv2.getOptimalNewCameraMatrix(K,DC,(w,h),1,(w,h))
        I = cv2.undistort(I, K, DC, None, Kn)
        # start  GUI
        fig,ax = plt.subplots(figsize=(20,20))
        ax.set_aspect('equal')
        skimage.io.imshow(I,ax=ax)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        # show the figure
        plt.show()
        # GCP names
        gcps=[]
        for k in np.arange(len(coords))+1:
            gcps.append("GCP {}".format(str(k).zfill(2)))
        # create a dataframe
        df = DataFrame(np.array(coords).astype(int),columns=["u","v"])
        df.index = gcps
        df.index.name = "GCPs"
        # # update names in the same order as the clicks
        gcps = []
        for i in range(len(df.index.values)):
            gcps.append("GCP {}".format(str(i+1).zfill(2)))
        # updated dataframe
        df.index = gcps
        df.index.name = "GCPs"

        print ('\n')
        print (df)
        
        # dump to csv
        df.to_csv(args.output[0])

    # plot the final result
    if args.show_result:
        fig,ax = plt.subplots(figsize=(20,20))
        ax.set_aspect('equal')
        skimage.io.imshow(I)
        # scatter the GCPS
        plt.scatter(df.u.values,df.v.values,s=40,c="r",marker="+",linewidths=2)
        # plot the GCPS names
        for x,y,gcp in zip(df.u.values,df.v.values,gcps):
            t = plt.text(x+30,y,gcp,color="k",fontsize=10)
            t.set_bbox(dict(facecolor=".5",edgecolor="k",alpha=0.85,linewidth=1))
        # set axes
        plt.ylim(I.shape[0],0)
        plt.xlim(0,I.shape[1])
        plt.tight_layout
        # show the plot
        plt.show()
        plt.close()

    print ("\nMy work is done!\n")
