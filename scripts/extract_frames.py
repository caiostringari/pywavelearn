# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : extract_frames.py
# POURPOSE : Extract frames from video data in parallel.
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 25/07/2016 [Caio Stringari]
# v1.1     : 05/08/2016 [Caio Stringari]
# v1.2     : 16/11/2017 [Caio Stringari]
# v1.3     : 16/11/2017 [Caio Stringari] - Fix PEP8 issues
#
# OBS      : The script is smart enough to pull metadata out of the input file,
#            however, it not always work. Be carefull !
#
# USAGE    : python extract_frames.py --help
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# python 3
from __future__ import print_function, division

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

from pywavelearn.utils import chunkify
from pywavelearn.image import metadata_parser


def image_worker(num, fi, fpath, frames, times, seeks, fbreak):
    """
    Worker function passed to multiprocessing.Process(). Will do most of
    the heavy lifting: open the video, extrac the frames and save the output.

    ----------
    Args:
        num [Mandatory (int)]: Number of the process being called.

        frames [Mandatory [np.ndarray, list]]: Sequential frame numbers list.

        times [Mandatory [np.ndarray, list]]: Timestamp list

        seeks [Mandatory [np.ndarray, list]]: ellapsed times to seek in the
                                              video.
    ----------
    Returns:
    """

    name = mp.current_process().name

    print("Worker", num)

    print("    +", name, ' starting')

    # loop over the files in the chunk
    k = 0
    for frame, time, seek in zip(frames, times, seeks):

        print("      --> Processing frame for :",
              time, "[Worker {}]".format(num))

        seek1 = seek.split(".")[0]
        seek2 = "00:00:00." + seek.split(".")[1]

        fname = "{}/{}.jpg".format(fpath, time.strftime("%Y%m%d_%H%M%S_%f"))
        cmd = "ffmpeg -y -ss {} -i {} -ss {} -frames:v 1 {} > /dev/null 2>&1".format(
            seek1, fi, seek2, fname)
        subprocess.call(cmd, shell=True)

        k += 1
        if fbreak and k == fbreak:
            print("++> Breaking loop")
            break

    print("    -", name, ' finishing')

    return


def main():

    # starting things
    start = datetime.datetime.now()
    print(
        "\nVideo processing starting at : {} \n".format(
            datetime.datetime.now()))

    # force breaker
    if args.force_break:
        fbreak = np.int(args.force_break[0])
    else:
        fbreak = False

    # frames folder
    fpath = args.output[0]
    subprocess.call("rm -rf {}".format(fpath), shell=True)
    os.makedirs(fpath)

    # input file
    fi = args.input[0]
    if os.path.isfile(fi):
        pass
    else:
        raise IOError("File {} not found !".format(fi))

    # number of processors
    nprocs = int(args.nproc[0])

    # output frequency
    freq = int(args.freq[0])  # in HZ !

    # get metadata
    metadata = metadata_parser(fi)

    # frames per second
    if args.force_fps[0]:
        fps = int(args.force_fps[0])
    else:
        fps = int(metadata["ExposureTime"].split('/')[1])

    # sampling frequencies
    fs = 1 / fps  # sample frequency in seconds
    fhz = 1 / fs  # sample frequency in hertz

    # step between the recording frequency and the output frequency
    if freq > fhz:
        raise IOError("Output frequency cannot be greater than the"
                      "aquisition frequency.")
    step = int(fhz / freq)

    # total lenght in seconds
    if args.force_duration[0]:
        hours = np.int(args.force_duration[0].split(":")[0])
        minutes = np.int(args.force_duration[0].split(":")[1])
        seconds = np.int(args.force_duration[0].split(":")[2])
    else:
        hours = np.int(metadata["Duration"].split(":")[0])
        minutes = np.int(metadata["Duration"].split(":")[1])
        seconds = np.int(metadata["Duration"].split(":")[2])

    t0 = datetime.datetime(2000, 1, 1, 0, 0)  # random reference time
    t1 = t0 + datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
    duration = np.int((t1 - t0).total_seconds()) + 1

    # total number of frames
    nframes = duration * fps

    # video original start date
    if args.force_date[0]:
        fmt = "%Y%m%d %H:%M:%S"
        vd_start = datetime.datetime.strptime(args.force_date[0], fmt)
    else:
        fmt = "%Y:%m:%d %H:%M:%S"
        vd_start = datetime.datetime.strptime(
            metadata["DateTimeOriginal"].split("+")[0], fmt)

    # frames to be processed
    frames = np.arange(0, nframes, step)

    # get time of each frame
    times = []
    seeks = []
    for i in range(len(frames)):
        now = vd_start + \
            datetime.timedelta(seconds=i * (duration / (len(frames))))
        seek = t0 + datetime.timedelta(seconds=i * (duration / (len(frames))))
        # append to the arrays
        times.append(now)
        seeks.append(seek.strftime("%H:%M:%S.%f"))

    # create the groups
    gframes = chunkify(frames, nprocs)
    gtimes = chunkify(times, nprocs)
    gseeks = chunkify(seeks, nprocs)

    # loop over the number of processors
    Q = mp.Queue()
    procs = []
    for i, wframes, wtimes, wseeks in zip(
            range(nprocs), gframes, gtimes, gseeks):
        p = mp.Process(
            target=image_worker,
            args=(
                i + 1,
                fi,
                fpath,
                wframes,
                wtimes,
                wseeks,
                fbreak))
        procs.append(p)
        p.start()

    # wait for all worker processes to finish
    for p in procs:
        p.join()

    end = datetime.datetime.now()

    elapsed = (end - start).total_seconds()
    print("\nElapsed time: {} seconds [{} minutes] ({} hours) \n".format(
        elapsed, round(elapsed / 60, 2), round(elapsed / 3600., 2)))
    print(
        "\nVideo processing finished at : {} ###\n".format(
            datetime.datetime.now()))


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()

    # Number of process to use
    parser.add_argument(
        '--nproc',
        '-n',
        nargs=1,
        action='store',
        default=1,
        dest='nproc',
        required=False,
        help="Number of processors to use. Default is to use one.")
    # Input file name
    parser.add_argument(
        '--input',
        '-i',
        nargs=1,
        action='store',
        dest='input',
        required=True,
        help="Video file full path."
             "Must be a file format supported by ffmpeg.")
    # Step
    parser.add_argument(
        '--freq', '-f',
        nargs=1,
        action='store',
        default=[10],
        dest='freq',
        required=False,
        help="Output writing frequency. Default is 5Hz.",)
    # Output path
    parser.add_argument(
        '--output', '-o',
        nargs=1,
        action='store',
        default=['frames/'],
        dest='output',
        required=False,
        help="Output writing folder. Default is frames/.",)
    # Force start date
    parser.add_argument(
        '--force-date',
        nargs=1,
        action='store',
        default=[False],
        dest='force_date',
        required=False,
        help="Force a start date, in case the metadata is wrong."
             "Use YYYYMMDD-HH:MM:SS format.",)
    # Force FPS
    parser.add_argument(
        '--force-fps',
        nargs=1,
        action='store',
        default=[False],
        dest='force_fps',
        required=False,
        help="Force the number of frames per second (FPS).",)
    # Force video duration
    parser.add_argument(
        '--force-duration',
        nargs=1,
        action='store',
        default=[False],
        dest='force_duration',
        required=False,
        help="Force video duration lenght in HH:MM:SS format.",)
    # Force video duration
    parser.add_argument(
        '--force-break',
        nargs=1,
        action='store',
        dest='force_break',
        required=False,
        help="Force break after N frames.",)

    # TODO: Add options to extract only certain amoount of time or/and
    # start/end dates.

    # parse all the arguments
    args = parser.parse_args()

    # main calls
    main()
