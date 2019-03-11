# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : extract_timestack.py
# POURPOSE : Extract a timestack considering pixel surroundings
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 06/09/2016 [Caio Stringari]
# v2.0     : 02/12/2016 [Caio Stringari]
#
# EXAMPLES:
#
# For usage help call:  python extract_timestack.py -h
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

from __future__ import print_function, division

# system
import os
import sys
import subprocess

# warnings
import warnings

# Files
from glob import glob

# Arguments
import argparse

# Dates
import datetime

# numpy
import numpy as np
import pandas as pd

# scipy
import scipy.spatial
from scipy.stats import mode


# multiprocess
import multiprocessing as mp
from multiprocessing import Queue

# random numbers
import random

# strings
import string

# netcdf IO
import xarray

# computer vision
import cv2
import skimage.io

# local tools
import pywavelearn.image as ipwl
from pywavelearn.utils import chunkify


def rectify_worker(num, frames):
    """
    Worker function passed to multiprocessing.Process(). This function:
     - reads a list of frames, loop over its elements;
     - undistort each frame using the camera matrix;
     - crop the edges;
     - rectify (project) the geometry to real-world coordinates;
     - calculate the horizon;
     - rotate the matrix;
     - extract the red, green and blue pixel arrays;
     - save the data in netcdf4 format.

    ----------
    Args:
        num [Mandatory (int)]: Number of the process being called.

        frames [Mandatory (list)]: List of frames to process.

    ----------
    Returns:

    """

    name = mp.current_process().name

    # print ("Worker",num)

    print("    + Worker ", num, ' starting')

    N = len(frames)

    # If

    # Loop over the files in the chunk
    k = 0
    for frame in frames:
        percent = round(((k * 100) / N), 2)
        print(
            "      --> Processing frame {} of {} ({} %) [Worker {}]".format(
                k +
                1,
                N,
                percent,
                num),
            " <--")

        # time
        fmt = "%Y%m%d_%H%M%S_%f"
        now = datetime.datetime.strptime(
            frame.split("/")[-1].strip(".jpg"), fmt)

        # Read image
        Ir = skimage.io.imread(frame)
        h, w = Ir.shape[:2]

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
        Xr, Yr = ipwl.rotate_translate(
            X, Y, rotation=theta, translation=[xt, yt])

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

        # interp = False
        # interpolate to a regular grid

        # do KDTree in the cliped array
        IDXc = kdtree(XYc, points)

        # points in metric coordinates - cliped array
        xc = XYc[IDXc, 0]
        yc = XYc[IDXc, 1]

        # points in pixel coordinates - cliped array
        ic = np.unravel_index(IDXc, Xr.shape)[0]
        jc = np.unravel_index(IDXc, Yr.shape)[1]

        # create a line of pixel centers
        i_stack_center = np.linspace(
            ic[0], ic[1], np.int(
                args.stackpoints[0])).astype(
            np.int).tolist()
        j_stack_center = np.linspace(
            jc[0], jc[1], np.int(
                args.stackpoints[0])).astype(
            np.int).tolist()

        # pixel centes in metric coordinates
        x_stack_center = Xc[i_stack_center, j_stack_center]
        y_stack_center = Yc[i_stack_center, j_stack_center]

        R = []
        G = []
        B = []
        # loop over pixel centers
        for i, j in zip(i_stack_center, j_stack_center):
            # find surrounding pixels
            isurrounding, jsurrounding = pixel_window(Ic, i, j, pxwin)
            # final pixel arrays
            iall = np.hstack([isurrounding, i])
            jall = np.hstack([jsurrounding, j])
            # extract pixel stats
            if pxstats == "mean":
                r = Ic[iall, jall, 0].mean()
                g = Ic[iall, jall, 1].mean()
                b = Ic[iall, jall, 2].mean()
            elif pxstats == "max":
                r = Ic[iall, jall, 0].max()
                g = Ic[iall, jall, 1].max()
                b = Ic[iall, jall, 2].max()
            elif pxstats == "min":
                r = Ic[iall, jall, 0].min()
                g = Ic[iall, jall, 1].min()
                b = Ic[iall, jall, 2].min()
            elif pxstats == "mode":
                r = mode(Ic[iall, jall, 0])[0][0]
                g = mode(Ic[iall, jall, 1])[0][0]
                b = mode(Ic[iall, jall, 2])[0][0]
            # Append to output
            R.append(r)
            G.append(g)
            B.append(b)
        # Final RGB array
        RGB = np.vstack([R, G, B]).astype(np.int).T

        # build the output data model
        ds = xarray.Dataset()
        # write rgb variable
        ds['rgb'] = (('points', 'bands'), RGB)  # All bandas
        # write positional cooridinates
        ds['x'] = x_stack_center  # x-coordinate
        ds['y'] = y_stack_center  # y-coordinate
        ds["i"] = i_stack_center  # central i-coordinate
        ds["j"] = j_stack_center  # central j-coordinate
        # write coordinates
        ds.coords['time'] = now  # camera time
        ds.coords["points"] = np.arange(0, len(i_stack_center), 1)
        # auxiliary variables
        ds["bands"] = ["red", "green", "blue"]
        # write to file
        units = 'days since 2000-01-01 00:00:00'
        calendar = 'gregorian'
        encoding = dict(time=dict(units=units, calendar=calendar))
        ds.to_netcdf("{}/{}.nc".format(tmpfolder, now.strftime(fmt)),
                     encoding=encoding)

        k += 1
        if fbreak and k == fbreak:
            print("++> Breaking loop")
            break
    print("    - Worker", num, ' finishing')

    return


def mergestacks(files, ncout='pts.nc', ramcut=3.0):
    """
    Merge extracted pixel lines in a timestack. Only will be called
    if parameter --timestack is True. Will always merge files in the
    temporal "time" dimension.

    ----------
    Args:
        files [Mandatory (list, np.ndarray)]: Sorted list of files to merge.

        ncout [Optional (str)]: Output filename. Defaul is timestack.nc

        ramcut [Optinal (float)]: Max fraction of memory to use. If the size of
        the expected merged file exceds this fraction will raise a MemoryError.

    ----------
    Returns:
    """
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xarray.open_dataset(path) as ds:
            # load dataset into memory
            ds.load()
            return ds

    # Memory check
    f_bytes = sum(os.path.getsize(f) for f in glob("tmp/*.nc"))
    m_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    if f_bytes >= m_bytes / ramcut:
        raise MemoryError("Not enough memory available to merge all files.")

    print("\n    + Merging timestacks, please wait...")

    # # progressbar
    # widgets = [ progressbar.FormatLabel(''),
    #             progressbar.Percentage(),
    #             ' ', progressbar.Bar(marker='#', left='[', right=']'),
    #             ' ', progressbar.ETA()]
    # bar = progressbar.ProgressBar(widgets=widgets,max_value=len(files))
    # bar.start()

    # Loop over files
    datasets = []
    for k, fname in enumerate(files):
        datasets.append(process_one_path(fname))
        # widgets[0] = progressbar.FormatLabel('      Stack {} of {}'.format(k+1,len(files)))
        # bar.update(k)
        # if k>100:break
    merged = xarray.concat(datasets, "time")
    merged.to_netcdf(ncout)
    merged.close()

    subprocess.call("rm -rf tmp/", shell=True)

    print("\n    - Stacks sucessfully merged")


def kdtree(A, pt):
    _, indexes = scipy.spatial.KDTree(A).query(pt)
    return indexes


def pixel_window(a, i, j, s=8):

    # compute domain
    i = np.arange(i - s, i + s + 1, 1)
    j = np.arange(j - s, j + s + 1, 1)

    # all pixels inside the domain
    I, J = np.meshgrid(i, j)

    # Remove pixels outside the borders

    # i-dimension
    I = I.flatten()
    I[I < 0] = -999
    I[I > a.shape[0]] = -999
    idx = np.where(I == -999)

    # j-dimension
    J = J.flatten()
    J[J < 0] = -999
    J[J > a.shape[1]] = -999
    jdx = np.where(J == -999)

    Ifinal = np.delete(I, np.hstack([idx, jdx]))
    Jfinal = np.delete(J, np.hstack([idx, jdx]))

    return Ifinal, Jfinal


def random_string(n):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(n))


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()

    # Number of cores to use
    parser.add_argument(
        '--nproc',
        '-n',
        nargs=1,
        action='store',
        default=[1],
        dest='nproc',
        required=False,
        help="Number of processors to use. Default is to use one.",)
    # Input frame folder
    parser.add_argument(
        '--input',
        '-i',
        nargs=1,
        action='store',
        dest='input',
        required=True,
        help="Folder with the extracted frames."
             "Please use extrac_frames.py before running this script.")
    # output
    parser.add_argument(
        '--output', '-o',
        nargs=1,
        action='store',
        default=['output.nc'],
        dest='output',
        required=False,
        help="Output file name.",)
    # Geometry
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
    parser.add_argument(
        '--gcpxyz',
        '-gcpxyz',
        '--xyz',
        '-xyz',
        nargs=1,
        action='store',
        dest='xyzfile',
        required=True,
        help="GCP XYZ file."
             "Similar to UV file, but with real-world coordinates instead",)
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
        default=[-999],
        help="Maximum distance from origin to be included in the plot.",
    )
    # Theta
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
    # Camera matrix
    parser.add_argument(
        '--camera-matrix',
        '-cm',
        nargs=1,
        action='store',
        dest='camera',
        required=True,
        help="Camera matrix file."
             "Please use calibrate.py to generate a valid file.",)
    # pixel line coordinates
    parser.add_argument(
        '-x1',
        nargs=1,
        action='store',
        dest='x1',
        required=False,
        help="First x-coordinate for the timestack."
             "If passed WILL NOT use matplotlib GUI.",)
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
             "If passed WILL NOT use matplotlib GUI.",
    )
    parser.add_argument(
        '-y2',
        nargs=1,
        action='store',
        dest='y2',
        required=False,
        help="Final y-coordinate for the timestack."
             "If passed WILL NOT use matplotlib GUI..",
    )
    # Number of points in the stack
    parser.add_argument('--stack-points', '-stkp',
                        nargs=1,
                        action='store',
                        default=[np.int(256)],
                        dest='stackpoints',
                        required=False,
                        help="Number of points in the timestack.",)
    # Pixel window size
    parser.add_argument(
        '--pixel-window', '--pxwin', '-pxwin', '-win',
        nargs=1,
        action='store',
        dest='pxwin',
        required=False,
        default=[0],
        help="Pixel window size. Default is 2.",)
    # Statistic
    parser.add_argument(
        '--pixel-statistic',
        '--statistic',
        '-stats',
        nargs=1,
        action='store',
        dest='pxstats',
        required=False,
        default=["mean"],
        help="Pixel statistics to use. Default is the average.",
    )
    # Whether to save frames or not
    parser.add_argument(
        '--save-frames',
        action='store_true',
        dest='save_frames',
        required=False,
        help="Save rectified frames in netcdf format. Default is false",
    )
    # Frame output path
    parser.add_argument(
        '--frame-output',
        nargs=1,
        action='store',
        default=['rectified/'],
        dest='frame_output',
        required=False,
        help="Output writing folder. Default is rectified/.",)
    # Compress
    parser.add_argument(
        '--compress',
        action='store_true',
        dest='compress',
        help="Compress the output folder to save disk space.")
    # Remove input frames to save space
    parser.add_argument(
        '--remove-frames',
        action='store_true',
        dest='remove_frames',
        help="Delete input frames folder.")
    # Force video duration
    parser.add_argument(
        '--force-break',
        nargs=1,
        action='store',
        dest='force_break',
        required=False,
        help="Force break after N frames.",)
    # parse all the arguments
    args = parser.parse_args()

    # ...I am not going to mess with this script moving things to main() ###

    # starting things up
    start = datetime.datetime.now()
    print("\nProcessing starting at : {} ###\n".format(
        datetime.datetime.now()))

    # number of processors
    nprocs = int(args.nproc[0])

    # force breake
    if args.force_break:
        fbreak = np.int(args.force_break[0])
    else:
        fbreak = False

    # input and output

    # input
    ipath = os.path.abspath(args.input[0])
    if os.path.isdir(ipath):
        pass
    else:
        raise IOError("Path {} not found !".format(ipath))

    # all frames
    files = np.sort(glob(ipath + "/*.jpg"))

    # output
    nc = args.output[0]

    # output
    if args.save_frames:
        keep = True
        opath = os.path.abspath(args.frame_output[0])
        subprocess.call("rm -rf {}".format(opath), shell=True)
        os.makedirs(opath)
    else:
        keep = False

    # temporary folder
    tmpfolder = 'tmp_' + random_string(8) + '/'
    subprocess.call("mkdir {}".format(tmpfolder), shell=True)

    # camera matrix
    K, DC = ipwl.camera_parser(args.camera[0])

    # read XYZ coords
    dfxyz = pd.read_csv(os.path.realpath(args.xyzfile[0]))
    XYZ = dfxyz[["x", "y", "z"]].values

    # read UV coords
    dfuv = pd.read_csv(os.path.realpath(args.uvfile[0]))
    UV = dfuv[["u", "v"]].values

    # horizon
    hor = float(args.horizon[0])

    # rotation Angle
    theta = float(args.theta[0])

    # projection height
    z = float(args.Z[0])

    # translation
    xt = float(args.X[0])
    yt = float(args.Y[0])

    # pixel window size
    pxwin = np.int(args.pxwin[0])

    # pixel stats
    pxstats = args.pxstats[0]

    # homography
    H = ipwl.find_homography(UV, XYZ, K, z=z)

    # reference homography
    # if args.

    # hhunkify
    fchunks = chunkify(files, nprocs)

    # create timestack line
    if args.x1 and args.x2 and args.y1 and args.y2:
        xstack = [np.float(args.x1[0]), np.float(args.x2[0])]
        ystack = [np.float(args.y1[0]), np.float(args.y2[0])]
        points = np.vstack([xstack, ystack]).T
        npoints = np.int(args.stackpoints[0])
    else:
        raise ValueError("Sorry, GUI was removed...")

    # loop over the number of processors
    Q = mp.Queue()
    procs = []
    for i, frames in zip(range(nprocs), fchunks):
        p = mp.Process(target=rectify_worker, args=(i + 1, frames))
        procs.append(p)
        p.start()
    # wait for all worker processes to finish
    for p in procs:
        p.join()

    # merge all extracted stacks to one netcdf
    mergestacks(np.sort(glob("{}*.nc".format(tmpfolder))), args.output[0])

    # remove temporary folder
    subprocess.call("rm -rf {}".format(tmpfolder), shell=True)

    # compress
    if keep:
        if args.compress:
            print("\nCompressing files, please wait...")
            tar = opath.split("/")[-1] + ".tar.gz"
            cmd = "tar -zcvf {} {}/*.nc > /dev/null 2>&1".format(
                tar, os.path.relpath(opath))
            subprocess.call(cmd, shell=True)
            subprocess.call("rm -rf {}".format(opath), shell=True)

    # delete input files
    if args.remove_frames:
        subprocess.call("rm -rf {}".format(ipath), shell=True)

    end = datetime.datetime.now()

    elapsed = (end - start).total_seconds()
    print("\nElapsed time: {} seconds [{} minutes] ({} hours) \n".format(
        elapsed, round(elapsed / 60, 2), round(elapsed / 3600., 2)))
    print("\nRectification finished at : {} \n".format(
        datetime.datetime.now()))
