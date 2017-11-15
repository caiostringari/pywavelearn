#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
#
# SCRIPT   : extract_frames.py
# POURPOSE : Extract frames from video data in parallel.
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 25/07/2016 [Caio Stringari]
# v1.1     : 05/08/2015 [Caio Stringari]
#
# OBSERVATIONS  : The script is smart enough to pull metadata out of the input file,
#                 however, it not always work. Be carefull !
#
# EXAMPLES:
#
# For usage help call:  python extract_frames.py -h
#
# 1 - Extract all frames in the video VIDEO.MP4 the folder "frames/" using 4 cores. Output frequency as 5Hz
# python extract_frames --nproc 4 -i VIDEO.MP4 -o frames/ --freq 5
# 2 - Same as example 1 but force a start date
# python extract_frames --nproc 4 -i VIDEO.MP4 -o frames/ --freq 5 --force-date 20160101-06:30:00
#
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
#
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# python 3
from __future__ import print_function,division

# system
import os
import sys
import subprocess

# Arguments
import argparse

# Dates
import datetime

# numpy
import numpy as np

# multiprocess
import multiprocessing as mp
from multiprocessing import Queue

# Ordered dict
from collections import OrderedDict

# User tools
from videotools import metadata_parser,chunkify


def image_worker(num,frames,times,seeks):
    """
    Worker function passed to multiprocessing.Process(). Will do most of
    the heavy lifting: open the video, extrac the frames and save the output.

    ----------
    Args:
        num [Mandatory (int)]: Number of the process being called.

        frames [Mandatory [np.ndarray, list]]: Sequential frame numbers list.

        times [Mandatory [np.ndarray, list]]: Timestamp list

        seeks [Mandatory [np.ndarray, list]]: ellapsed times to seek in the video.

    ----------
    Returns:

    """

    name = mp.current_process().name

    print ("Worker",num)

    print ("    +",name, ' starting')

    # Loop over the files in the chunk
    k = 0
    for frame,time,seek in zip(frames,times,seeks):

        print ("      --> Processing frame for :",time,"[Worker {}]".format(num)," <--")

        seek1 = seek.split(".")[0]
        seek2 = "00:00:00."+seek.split(".")[1]
        #print (seek1,seek2)

        fname = "{}/{}.jpg".format(fpath,time.strftime("%Y%m%d_%H%M%S_%f"))
        cmd = "ffmpeg -y -ss {} -i {} -ss {} -frames:v 1 {} > /dev/null 2>&1".format(seek1,fi,seek2,fname)
        subprocess.call(cmd,shell=True)

        k+=1
        if fbreak and k == fbreak:
            print ("++> Breaking loop")
            break

    print ("    -",name, ' finishing')

    return

if __name__ == '__main__':

    # Insert main working folder in the PATH
    #sys.path.insert(0,"/home/caio/Documents/PhD/Tools/")


    ### Argument parser
    parser = argparse.ArgumentParser()

    # Number of cores to use
    parser.add_argument('--nproc','-n',
                        nargs = 1,
                        action = 'store',
                        default = 1,
                        dest = 'nproc',
                        required = False,
                        help = "Number of processors to use. Default is to use one.",)
    # Input file name
    parser.add_argument('--input','-i',
                        nargs = 1,
                        action = 'store',
                        dest = 'input',
                        required = True,
                        help = "Video file full path. Must be a file format supported by ffmpeg.",)

    # Step
    parser.add_argument('--freq','-f',
                        nargs = 1,
                        action = 'store',
                        default = [5],
                        dest = 'freq',
                        required = False,
                        help = "Output writing frequency. Default is 5Hz.",)
    # Output path
    parser.add_argument('--output','-o',
                        nargs = 1,
                        action = 'store',
                        default = ['frames/'],
                        dest = 'output',
                		required = False,
                        help = "Output writing folder. Default is frames/.",)
    # Force start date
    parser.add_argument('--force-date',
                        nargs = 1,
                        action = 'store',
                        default = [False],
                        dest = 'force_date',
                        required = False,
                        help = "Force a start date, in case the metadata is wrong. Use YYYYMMDD-HH:MM:SS format.",)
    # Force FPS
    parser.add_argument('--force-fps',
                        nargs = 1,
                        action = 'store',
                        default = [False],
                        dest = 'force_fps',
                        required = False,
                        help = "Force the number of frames per second (FPS).",)
    # Force video duration
    parser.add_argument('--force-duration',
                        nargs = 1,
                        action = 'store',
                        default = [False],
                        dest = 'force_duration',
                        required = False,
                        help = "Force video duration lenght in HH:MM:SS format.",)
    # Force video duration
    parser.add_argument('--force-break',
                        nargs = 1,
                        action = 'store',
                        dest = 'force_break',
                        required = False,
                        help = "Force break after N frames.",)

    #TODO: Add options to extract only certain amoount of time or/and start/end dates.

    # parse all the arguments
    args = parser.parse_args()

    # starting things
    start = datetime.datetime.now()
    print ("\n #### Video processing starting at : {} ###\n".format(datetime.datetime.now()))

    ### Input and Output ###

    # force breaker
    if args.force_break:
        fbreak = np.int(args.force_break[0])
    else:
        fbreak = False

    # frames folder
    fpath = args.output[0]
    subprocess.call("rm -rf {}".format(fpath),shell=True)
    os.makedirs(fpath)

    # input file
    fi = args.input[0]
    if os.path.isfile(fi):
        pass
    else:
        raise IOError("File {} not found !".format(fi))

    ### Number of processors
    nprocs = int(args.nproc[0])

    ### Output frequency
    freq = int(args.freq[0]) # in HZ !

    ### Get metadata
    metadata = metadata_parser(fi)
    # print (metadata["Duration"])

    # frames per second
    if args.force_fps[0]:
        fps = int(args.force_fps[0])
    else:
        fps = int(metadata["ExposureTime"].split('/')[1])

    # Sampling frequencies
    fs = 1/fps # sample frequency in seconds
    fhz = 1/fs # sample frequency in hertz


    # Step between the recording frequency and the output frequency
    if freq > fhz:
        raise IOError("Output frequency cannot be greater than the aquisition frequency.")
    step = int(fhz/freq)

    # Total lenght in seconds
    if args.force_duration[0] == True:
        hours = np.int(args.force_duration[0].split(":")[0])
        minutes = np.int(args.force_duration[0].split(":")[1])
        seconds = np.int(args.force_duration[0].split(":")[2])
    else:
        hours = np.int(metadata["Duration"].split(":")[0])
        minutes = np.int(metadata["Duration"].split(":")[1])
        seconds = np.int(metadata["Duration"].split(":")[2])

    t0 = datetime.datetime(2000,1,1,0,0) # random reference time
    t1 = t0+datetime.timedelta(hours=hours,minutes=minutes,seconds=seconds)
    duration = np.int((t1-t0).total_seconds())+1

    # Total number of frames
    nframes = duration*fps

    # Video Original start date
    if args.force_date[0]:
        fmt = "%Y%m%d %H:%M:%S"
        vd_start = datetime.datetime.strptime(args.force_date[0],fmt)
    else:
        fmt = "%Y:%m:%d %H:%M:%S"
        vd_start = datetime.datetime.strptime(metadata["DateTimeOriginal"].split("+")[0],fmt)

    # Frames to be processed
    frames = np.arange(0,nframes,step)

    ### Get time of each frame
    times = []
    seeks = []
    for i in range(len(frames)):
        now = vd_start+datetime.timedelta(seconds=i*(duration/(len(frames))))
        seek = t0+datetime.timedelta(seconds=i*(duration/(len(frames))))
        # append to the arrays
        times.append(now)
        seeks.append(seek.strftime("%H:%M:%S.%f"))

    ### Create the groups
    gframes = chunkify(frames,nprocs)
    gtimes = chunkify(times,nprocs)
    gseeks = chunkify(seeks,nprocs)

    ### Loop over the number of processors
    Q = mp.Queue()
    procs = []
    for i,wframes,wtimes,wseeks in zip(range(nprocs),gframes,gtimes,gseeks):
        # print (i+1, gframes, gstarts)
        p = mp.Process(target=image_worker, args=(i+1,wframes,wtimes,wseeks))
        procs.append(p)
        p.start()

    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    end = datetime.datetime.now()

    elapsed = (end-start).total_seconds()
    print ("\n Elapsed time: {} seconds [{} minutes] ({} hours) \n".format(elapsed,round(elapsed/60,2),round(elapsed/3600.,2)))
    print ("\n #### Video processing finished at : {} ###\n".format(datetime.datetime.now()))
