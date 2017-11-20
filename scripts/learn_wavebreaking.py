#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
# SCRIPT   : wavebreaking
# POURPOSE : identify wave breaking events in timestacks based on machine learn
#            algorithms.
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 14/09/2017 [Caio Stringari] - Base code forked from previous scripts.
# v2.0     : 05/10/2017 [Caio Stringari] - Major reformulation, included the ML steps
# v2.1     : 31/10/2017 [Caio Stringari] - Refine ML portions
# v2.2     : 20/11/2017 [Caio Stringari] - Major reformulation, add JSON input and
#                                          edge detection options.
#
#
#------------------------------------------------------------------------
#------------------------------------------------------------------------
from __future__ import print_function, division

# import matplotlib
# matplotlib.use('GTKAgg')

# System
import sys
import os
import subprocess

# Arguments
import argparse

# Dates
import time
import datetime
from matplotlib.dates import date2num, num2date

# Data
import json
import xarray as xr
import pandas as pd

# GIS
import geopandas as gpd
from shapely.geometry import Point

# Numpy
import numpy as np

# Peak detection
from peakutils import baseline, envelope, indexes

# Colourspaces
import colour
from colorspacious import deltaE

# distances
from  scipy.spatial.distance import __all__ as scipy_dists

# pywavelearning
from pywavelearning.utils import ellapsedseconds ,peaklocalextremas
from pywavelearning.colour import get_dominant_colour,classify_colour
from pywavelearning.image import construct_rgba_vector, pixel_window


# from wave_utils.waves import *
# from wavetools import (ellapsedseconds, Hrms, find_nearest, fixtime, align_signals, smoothpixelseries)
# from videotools import _construct_rgba_vector as construct_rgba_vector, chunkify

# Image processing
from skimage.io import imsave


# SK-Learn classifiers
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# from sklearn.model_selection import train_test_split

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sns.set_context("paper", font_scale = 1.5)
sns.set_style("ticks", {'axes.linewidth': 2.0})


# import warnings
# warnings.filterwarnings("ignore")


def get_analysis_domain(time, y, stack):
    """
    Get analysis spatial domain from mouse clicks.

    ----------
    Args:
        time (Mandatory [np.array]): Time array.

        y (Mandatory [np.array]): Space array [m].

        stk (Mandatory [np.array]): [time,space,3] timestack array.
        Used for plotting only.

    ----------
    Return:
        swash_zone (Mandatory [Float]): onshore limit of the swash zone

        surf_zone (Mandatory [Float]): offshore limit of the surf zone
    """
    # Get analysis start location from GUI
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 4))
    # plt.show()
    ax.set_title(
        "Click in the start of swash zone and in the end of the surf zone...")
    im = ax.pcolormesh(time, y, stack.mean(axis=2).T)
    # set timestack to  to true color
    rgba = construct_rgba_vector(np.rollaxis(stack, 1, 0), n_alpha=0)
    rgba[rgba > 1] = 1
    im.set_array(None)
    im.set_edgecolor('none')
    im.set_facecolor(rgba)
    plt.draw()
    # location from click
    point = plt.ginput(n=2,timeout=1000000)
    swash_zone = point[0][1]
    surf_zone = point[1][1]
    # plt.show()
    plt.close()

    return swash_zone, surf_zone

def get_analysis_locations(stk_dist, start, end, step=1):
    """
    Select analysis locations based on the analysis domain and user defined
    step.

    ----------
    Args:
        stk_dist (Mandatory [np.array]): Space array [m].

        start, end (Mandatory [float]): start and end of the spatial domain,
        use get_analysis_domain() to obtain this values.

        step (Optional [int]): skip step. Default is not to skip anything.

    ----------
    Return:
        Y (Mandatory [np.array]): Cross-shore locations in meters.

        Idx (Mandatory [np.array]): Cross-shore locations indexes.
    """
    space_step = step
    Y = []
    Idx = []
    for y in stk_dist[::step]:
        idx = np.argmin(np.abs(stk_dist - y))
        # append only values in the surf zone
        if y > start and y < end:
            Y.append(y)
            Idx.append(idx)
    Y = Y[::-1]
    Idx = Idx[::-1]

    return Y, Idx

def get_pixel_lines(stk_time, stk, Idx, resample_rule="100L", pxmtc="lightness"):
    """
    Extract pixel timeseries from a given timestack at the locations obtained
    from get_analysis_locations().

    step.

    ----------
    Args:
        stk_time (Mandatory [np.array]): array of datetimes.

        stk (Mandatory [np.array]): [time,space,3] timestack array.

        Idx (Mandatory [np.array]): cross-shore indexes obtained from
        get_analysis_locations()

        resample_rule (Optional [str]): Resample frequency. Default is "100L"

        pxmtc (Optional [str]): Pixel metric to use. Default is "lightness"

    ----------
    Returns:
        time (Mandatory [np.array]): Array of datetimes

        dEk (Mandatory [np.array]): [time,Idx] pixel metric array

        RGB (Mandatory [np.array]): [time,Idx,3] colour array
    """


    RGB = []
    pxint = []
    for idx in Idx:
        # pixel line at exact location
        pxline = stk[:, idx, :]
        # build a dataframe
        df = pd.DataFrame(
            pxline, columns=["red", "green","blue"], index=stk_time)
        # add grey values
        df["grey"] = pxline.mean(axis=1).astype(np.int)
        # adjust records in time upsampling to 8Hz
        df = df.resample(resample_rule).backfill()
        # df = smoothpixelseries(df, window=5, order=2)
        # compute color parameters
        if pxmtc == "lightness":
            # get rgb values
            rgb = df[["red", "green", "blue"]].values.astype(np.int)
            # exclude invalid values from interpolation
            rgb[rgb < 0] = 0
            # lightness
            pxint.append(deltaE(rgb, [0, 0, 0], input_space="sRGB255") / 100)
            RGB.append(rgb)
        elif pxmtc == "intensity":
            # get rgb values
            rgb = df[["red", "green", "blue"]].values.astype(np.int)
            # exclude invalid values from interpolation
            rgb[rgb < 0] = 0
            # intensity
            pxint.append(np.mean(rgb))
            RGB.append(rgb)
        else:
            raise NotImplementedError(
                "Colour metric {} is not valid.".format(pxmtc))
    # get times
    time = df.index.to_datetime().to_pydatetime()

    return time, pxint, RGB


def get_training_data(rgb, regions=3, region_names=["sand", "foam", "water"],
                      nclicks=3, iwin=2, jwin=2):
    """
    Get training color data based on user input.

    ----------
    Args:
        rgb (Mandatory [np.array]): [time,space,3] timestack array.

        regions (Optional [int]): Number of regions in the timestack.
        Default is 3

        region_names   (Optional [int]): NUmber of regions in the timestack.
        Default is ["sand", "foam", "water"].

        nclicks (Optional [int]): Number of clicks. Default is 3.

        iwin (Optional [int]): Window size in the u-direction. Default is 2.

        jwin (Optional [int]): Window size in the u-direction. Default is 2.

    ----------
    Returns:
        df_colour (Mandatory [pd.DataFrame]): Dataframe with training colour
        information.
    """

    # get pixel centers in the time-space domain
    i = 0
    I_center = []
    J_center = []
    Labels = []
    Regions = []
    for label, region in zip(range(regions), region_names):
        # GUI
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_title("Click in {} points in the {}".format(nclicks, region))
        im = plt.pcolormesh(np.mean(rgb, axis=2).T)
        # set timestack to  to true color
        rgba = construct_rgba_vector(np.rollaxis(rgb, 1, 0), n_alpha=0)
        rgba[rgba > 1] = 1
        im.set_array(None)
        im.set_edgecolor('none')
        im.set_facecolor(rgba)
        plt.draw()
        # get the i,j coordnates
        points = np.array(plt.ginput(n=nclicks, show_clicks=True,timeout=1000000))
        # append to output
        for k in range(points.shape[0]):
            I_center.append(points[k, 0])
            J_center.append(points[k, 1])
            Labels.append(label)
            Regions.append(region)
        # iterate over regions
        i += 1
        plt.close()

    # loop over pixel centers and build the pixel windows
    i_pixels_all = []
    j_pixels_all = []
    l_pixels_all = []
    r_pixels_all = []
    for x, y, region, label in zip(I_center, J_center, Regions, Labels):
        # get the pixel window
        i_all, j_all = pixel_window(rgb, x, y, iwin=iwin, jwin=jwin)
        # append to output
        for i in range(len(i_all)):
            i_pixels_all.append(np.int(i_all[i]))
            j_pixels_all.append(np.int(j_all[i]))
            l_pixels_all.append(label)
            r_pixels_all.append(region)

    # get RGB values
    rgb_training = rgb[i_pixels_all, j_pixels_all, :]

    # get individual tristimulus values
    r = rgb_training[:, 0]
    g = rgb_training[:, 1]
    b = rgb_training[:, 2]

    # compute color space XYZ
    # XYZ = colour.sRGB_to_XYZ(rgb_training / 255)
    # # get individual vales
    # X = XYZ[:, 0]
    # Y = XYZ[:, 1]
    # Z = XYZ[:, 2]

    # color distances to a black pixel
    # dEk = deltaE(rgb_training, [0, 0, 0], input_space="sRGB255") / 100

    # build training dataset
    df_colour = pd.DataFrame(np.vstack([r, g, b]).T,
                             columns=["r", "g", "b",])

    # add label and region names
    df_colour["label"] = l_pixels_all
    df_colour["region"] = r_pixels_all

    return df_colour

def detect_breaking_events(Time, Y, PxInts, RGB, algorithm = "peaks",
                                                 peak_detection = "local_maxima",
                                                 threshold = 0.15,
                                                 win = 20,
                                                 colours = None):
#FIXME
    """
    Detect wave breaking events. Two methods are implemented, the first use a local maxima detection
    algorithm and the second uses a differential methods. In both cases, the detected peaks
    can be fine-tuned using the colour comparison machine learning compariosn method.

    ----------
    Args:

        Time (Mandatory [np.array]): Array of datetimes.

        Y (Mandatory [np.array]): Array of cross-shore locations [m].

        PxInts (Mandatory [np.array]): Array RGB colours. Needs to have dimensions compatible
        with Time and Y arrays.

        win (Optional [int]): Number of data points to look ahead when searching for pixel peaks
        in the data. Default is 16.

        fac (Optional [int]): Threshold for pixel peak detection. Default is 0.15.

        method (Optional [str]): Which method to use. Options are "local_maxima" and "differential".
        Default is differential.


        colour_comparison (Optional [str]): Whether to use the colour machine learning step.
        Default is False

        dominant_colours (Optional [list of lists]): A list of RGB list (or tuples) of dominant colours
        for the colour comparion. Ex: [[0,0,0],[255,255,255]]

        colour_labels (Optional [list of strings]): A list unique labels corresponding to each color
        in the dominant colour list. Mandatory if color_comparison is set to True

        tartget_id (Optional [int or float]): Which unique label in the color label array corresponds
        to wave breaking. Mandatory if color_comparison is set to True


    ----------
    Return:
         x (Mandatory [np.array]): Array of datetimes of identified breaking events.

         s (Mandatory [np.array]): Array of ellapsed seconds  since the time origin
                                   of identified breaking events.

         y (Mandatory [np.array]): array of spatial locations of pixel peaks.

    """

    PeakTimes = []
    # loop over data rows
    for pxint, rgb in zip(PxInts, RGB):

        if algorithm in ["colour","peaks"]:
            # calculate baseline
            bline = baseline(pxint)
            # calculate pixel peaks
            if peak_detection == "local_maxima":
                _, max_idx = peaklocalextremas(pxint - bline,
                                               lookahead = win,
                                               delta = threshold * (pxint - bline).max())
            elif peak_detection == "differential":
                    # calculate first derivative
                    pxintdt = np.diff(pxint - bline)
                    # remove values below zero
                    pxintdt[pxintdt <= 0] = 0
                    # scale from 0 to 1
                    pxintdt = pxintdt / pxintdt.max()
                    # get indexes
                    max_idx = indexes(pxintdt, thres=fac, min_dist=win)
            # colour learning step
            if algorithm == "colour":
                # get colour data
                target = colours["target"]
                labels = colours["labels"]
                dom_colours = colours["rgb"]
                # classifiy pixels
                breaker_idxs = []
                for idx in max_idx:
                    y_pred = classify_colour(rgb[idx], dom_colours, labels)
                    if y_pred[0] == target:
                        breaker_idxs.append(idx)
            # peaks only method
            else:
                breaker_idxs = max_idx
            # append to output
        PeakTimes.append(Time[breaker_idxs])

    # Organize peaks and times
    Xpeaks = []
    Ypeaks = []
    for i, pxtimes in enumerate(PeakTimes):
        for v in pxtimes:
            Xpeaks.append(v)
        for v in np.ones(len(pxtimes)) * Y[i]:
            Ypeaks.append(v)

    # sort values in time
    y = np.array(Ypeaks)[np.argsort(date2num(Xpeaks))]
    x = np.array(Xpeaks)[np.argsort(Xpeaks)]
    s = ellapsedseconds(x)

    return s, y


def write_raster_geom(rst,dx,dy,x,y,i,j):
    # FIXME: docstrings

    f = open(rst.replace(".jpg",".wld"),"w")
    f.write('{}\n'.format(dx))
    f.write('0\n')
    f.write('0\n')
    f.write('-{}\n'.format(dy))
    f.write('0.0\n')
    f.write(str(y)+"\n")
    f.close()

    f = open(rst.replace(".jpg",".points"),"w")
    f.write('mapX,mapY,pixelX,pixelY,enable\n')
    f.write('0,0,0.00,-{},1\n'.format(i))
    f.write('{},{},{},-0.00,1\n'.format(x,y,j))
    f.close()


def dbscan(times,breakers):

    # scale data for learning
    xscaled = times / (breakers.max() - breakers.min())
    yscaled = breakers / (times.max() - times.min())
    X = np.vstack([xscaled, yscaled]).T

    # cluster
    db = DBSCAN(eps=dbs_eps, min_samples=dbs_msp,metric=dbs_mtc).fit(X)

    # inliers and outliers
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # dump results into a dataframe
    df_dbscan = pd.DataFrame(np.vstack([times, breakers, labels]).T,
                             columns=["time", "breaker", "wave"])

    return df_dbscan


if __name__ == '__main__':

    print("\nCreating wave database, please wait...\n")

    print("\n    Starting at : {}\n".format(datetime.datetime.now()))

    # Argument parser
    parser = argparse.ArgumentParser()

    # input data
    parser.add_argument('--input', '-i',
                        nargs=1,
                        action='store',
                        dest='input',
                        help="Input json with analysis parameters.",
                        required=True)
    # parser
    args = parser.parse_args()

    ## read parameters from JSON ##
    with open(args.input[0], 'r') as f:
        H = json.load(f)

    ## input and output ##

    # input timestack
    ncin = H["data"]["ncin"]

    # output
    outputto = os.path.abspath(H["data"]["dataout"])
    if not os.path.isdir(outputto):
        os.makedirs(outputto)

    ## get analysis parameters ##

    # which algorithm to use to detect breaking events
    brk_alg = H["parameters"]["breaking_algorithm"]
    if brk_alg not in ["colour","peaks","edge"]:
            raise ValueError("Breaking detection algorithm should be either \'colour\', \'peaks\', or \'edges\'.")
    # metric to use for pixel intensity
    px_mtrc = H["parameters"]["pixel_metric"]
    if px_mtrc not in ["lightness","intensity"]:
        raise ValueError("Pixel metric should be either \'lightness\' or \'intensity\'.")
    # peak detection method
    pxp_mtd = H["parameters"]["peak_detection_algorithm"]
    # threshold for peak detection
    px_trsh = H["parameters"]["pixel_threshold"]
    # minimum number of samples to be used for the DBSCAN clustering step
    dbs_msp = H["parameters"]["min_samples"]
    # minimum distance be used for the DBSCAN clustering step
    dbs_eps = H["parameters"]["eps"]
    # distance metric for the DBSCAN clustering step
    dbs_mtc = H["parameters"]["dbscan_metric"]
    scipy_dists.append(['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'])
    if dbs_mtc not in scipy_dists:
        raise ValueError("Distance \'{}\' is not valid. Please check scipy.spatial.distance for valid options.".format(dbs_mtc))
    # resample rule for timeseries
    rs_rule = H["parameters"]["resample_rule"]
    # surf zone limits from input parameters file, if False, will call GUI.
    sz_lims = H["parameters"]["surf_zone_lims"]
    if not sz_lims[0]:
        sz_gui = True
    else:
        sz_gui = False
    # sand colour from input parameter file, if False, will call GUI
    s_color = H["parameters"]["sand_colour"]
    if not s_color[0]:
        sand_colour_gui = True
    else:
        sand_colour_gui = False
        sand = np.array(s_color)
    # water colour from input parameter file, if False, will call GUI
    w_color = H["parameters"]["water_colour"]
    if not w_color[0]:
        water_colour_gui = True
    else:
        water_colour_gui = False
        water = np.array(w_color)
    # foam colour from input parameter file, if False, will call GUI
    f_color = H["parameters"]["foam_colour"]
    if not f_color[0]:
        foam_colour_gui = True
    else:
        foam_colour_gui = False
        foam = np.array(f_color)
    # number of clicks for GUI
    n_clks = H["parameters"]["nclicks"]
    # pixel window for GUI
    px_wind = H["parameters"]["nclicks"]
    # try to fix bad pixel values
    gap_val = H["parameters"]["gap_value"]


    ## process timestack ##

    # read timestack
    ds = xr.open_dataset(ncin)

    # load timestack variables #
    x = ds["x"].values
    y = ds["y"].values

    # compute distance from shore
    stk_len = np.sqrt(((x.max() - x.min()) * (x.max() - x.min())) + ((y.max() - y.min()) * (y.max() - y.min())))
    crx_dist = np.linspace(0, stk_len, len(x))

    # get timestack times
    stk_time = pd.to_datetime(ds["time"].values).to_pydatetime()
    stk_secs = ellapsedseconds(stk_time)

    # get RGB values
    rgb = ds["rgb"].values

    # try to fix vertical grey strips, if any
    ifix,jfix = np.where(np.all(rgb == gap_val, axis=-1))
    rgb[ifix,jfix,:] = rgb[ifix-2,jfix-2,:]


    ## get analysis limits
    if sz_gui:
        crx_start, crx_end = get_analysis_domain(stk_secs, crx_dist, rgb)
    else:
        crx_start = sz_lims[0]
        crx_end = sz_lims[1]


    ### run the analysis ####
    if brk_alg == "colour":
        # sand
        if sand_colour_gui:
            df_sand = get_training_data(rgb,regions = 1,
                                            region_names = ["sand"],
                                            iwin = px_wind,
                                            jwin = px_wind,
                                            nclicks = n_clks)
            _,_,sand = get_dominant_colour(df_sand, n_colours=8)
            sand = sand[0]
        # water
        if water_colour_gui:
            df_water = get_training_data(rgb,regions = 1,
                                             region_names = ["water"],
                                             iwin = px_wind,
                                             jwin = px_wind,
                                             nclicks = n_clks)
            _,_,water = get_dominant_colour(df_water, n_colours=8)
            water = water[0]
        # foam
        if foam_colour_gui:
            df_foam = get_training_data(rgb,regions = 1,
                                             region_names = ["foam"],
                                             iwin = px_wind,
                                             jwin = px_wind,
                                             nclicks = n_clks)
            _,_,foam = get_dominant_colour(df_foam, n_colours=8)
            foam = foam[0]

        # build colour structures
        colours = {'labels':[0,1,2],
                   'aliases':["sand","water","foam"],
                   'rgb':[sand,water,foam],
                   'target':2}

        # get pixel lines at analysis locations only
        crx_y, crx_idx = get_analysis_locations(crx_dist, crx_start, crx_end)
        time, pxint, crxrgb = get_pixel_lines(stk_time, rgb, crx_idx,
                                           resample_rule=rs_rule, pxmtc=px_mtrc)

        # get analysis frequency and a 2 sececonds time window
        fs = (time[1] - time[0]).total_seconds()
        fs_win = np.int(2 * (1 / fs))

        # detect breaking events
        times, breakers = detect_breaking_events(time, crx_y, pxint, crxrgb,
                                                       algorithm = "colour",
                                                       peak_detection = "local_maxima",
                                                       threshold = px_trsh,
                                                       win = fs_win,
                                                       colours = colours)

    ## DBSCAN
    df_dbscan = dbscan(times,breakers)


    ## plot final results
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.8])
    ax2 = fig.add_axes([0.85, 0.1, 0.1, 0.8])

    # plot machine learn colour
    colors = plt.cm.get_cmap("prism", len(np.unique(df_dbscan["wave"].values)))
    # for label, region, color in zip(colour_labels, colours_regions, dominant_colours):
    #     ax2.scatter(1, label, 1000, marker="s", color=color /
    #                 255, label=region, edgecolor="k", lw=2)
    # set axis
    # ax2.set_yticks(colour_labels)
    # ax2.set_yticklabels(colours_regions)
    # ax2.set_ylim(-0.25, len(colours_regions) - 1 + 0.25)
    # for tick in ax2.get_yticklabels():
    #     tick.set_rotation(90)
    # ax2.xaxis.set_major_locator(plt.NullLocator())
    # ax2.set_title("Dominant Colours")
    # sns.despine(ax=ax2, bottom=True)
    # plot stack
    im = ax1.pcolormesh(stk_secs, crx_dist, rgb.mean(axis=2).T)
    # swash zone limit
    ax1.axhline(y=crx_start, lw=3,color="darkred", ls="--")
    ax1.axhline(y=crx_end, lw=3, color="darkred", ls="--")
    # plot all identified breaking events
    ax1.scatter(times, breakers, 60, color="orangered",edgecolor="k", alpha=0.75, marker="+", lw=1)
    # DBSCAN plot
    k = 0
    for label, group in df_dbscan.groupby("wave"):
        # get x and y for regression
        xwave = group["time"].values
        ywave = group["breaker"].values
        # scatter points
        ax1.scatter(xwave, ywave, 40, label="wave " + str(label),
                    c=colors(k), alpha=0.5, marker="o", edgecolor="k")
        bbox = dict(boxstyle="square",
                    ec="k", fc="w", alpha=0.5)
        ax1.text(xwave.min(), ywave.max(), "wave " + str(k),
                 rotation=45, fontsize=8, zorder=10, bbox=bbox)
        k += 1
    # set timestack to  to true color
    rgba = construct_rgba_vector(np.rollaxis(rgb, 1, 0), n_alpha=0)
    rgba[rgba > 1] = 1
    im.set_array(None)
    im.set_edgecolor('none')
    im.set_facecolor(rgba)
    # set axes limits
    ax1.set_xlim(xmin=stk_secs.min(), xmax=stk_secs.max())
    # ax1.set_ylim(ymin=crx_dist.min(), ymax=crx_dist.max())
    # write times instead of seconds
    newlabels = []
    for tkc in ax1.get_xticks():
        now = stk_time[0]+datetime.timedelta(seconds=tkc)
        newlabels.append(now.strftime("%d/%m/%y\n%H:%M"))
    ax1.set_xticklabels(newlabels,fontsize=14)
    ax1.set_yticklabels(ax1.get_yticks(),fontsize=16)
    # set axes
    ax1.set_ylabel(r"Cross-shore Distance $[m]$",fontsize=16)
    sns.despine(ax=ax1)
    # plt.draw()
    # plt.savefig(path+'plots/breaking/'+t1.strftime("%Y%m%d-")+str(timeseries).zfill(3)+".png",bbox_inches='tight', pad_inches=0)
    plt.show()
    # plt.close()
