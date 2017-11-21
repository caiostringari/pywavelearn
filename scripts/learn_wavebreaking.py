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
# v2.2     : 20/11/2017 [Caio Stringari] - Major reformulation, add JSON I/O and
#                                          edge detection options.
#
#
#------------------------------------------------------------------------
#------------------------------------------------------------------------
from __future__ import print_function, division

# System
import os
import sys

# Arguments
import argparse

# Dates
import time
import datetime
from matplotlib.dates import date2num, num2date

# Numpy
import numpy as np

# Data
import json
import xarray as xr
import pandas as pd

# GIS
import geopandas as gpd
from shapely.geometry import Point

# Peak detection
from peakutils import baseline, envelope, indexes

# Colourspaces
import colour
from colorspacious import deltaE

# distances, fr checking only
from scipy.spatial.distance import __all__ as scipy_dists

# pywavelearning
from pywavelearning.utils import ellapsedseconds, peaklocalextremas
from pywavelearning.image import construct_rgba_vector, pixel_window
from pywavelearning.colour import get_dominant_colour,classify_colour,colour_quantization

# Image processing
from skimage.io import imsave
from skimage.color import rgb2grey
from skimage.filters import sobel_h
from skimage.restoration import denoise_bilateral

# Machine learning
from sklearn.cluster import DBSCAN

# quite some warnings
import warnings
warnings.filterwarnings("ignore")

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale = 1.5)
sns.set_style("ticks", {'axes.linewidth': 2.0})

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

    # build training dataset
    df_colour = pd.DataFrame(np.vstack([r, g, b]).T,
                             columns=["r", "g", "b",])

    # add label and region names
    df_colour["label"] = l_pixels_all
    df_colour["region"] = r_pixels_all

    return df_colour

def get_pixel_lines(stk_time, stk, Idx, resample_rule="100L", pxmtc="lightness"):
    """
    Extract pixel timeseries from a given timestack at the locations obtained
    from get_analysis_locations().

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

    #case 3D
    if len(stk.shape) == 3:
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
    # case 2D:
    elif len(stk.shape) == 2:
        RGB = []
        pxint = []
        for idx in Idx:
            # pixel line at exact location
            pxline = stk[:, idx]
            # build a dataframe
            df = pd.DataFrame(pxline, columns=["grey"], index=stk_time)
            # adjust records in time upsampling if needed
            df = df.resample(resample_rule).backfill()
            # compute color parameters
            pxint.append(pxline)
            RGB.append([pxline,pxline,pxline])
    else:
        raise ValueError("Image needs to be 2D or 3D.")

    # get times
    time = df.index.to_datetime().to_pydatetime()

    return time, pxint, RGB

def detect_breaking_events(time, crx_dist, rgb, crx_start = None,
                                                crx_end = None,
                                                px_mtrc = "lightness",
                                                resample_rule = "100L",
                                                algorithm = "peaks",
                                                peak_detection = "local_maxima",
                                                posterize = False,
                                                ncolours = 0,
                                                threshold = 0.1,
                                                tswindow = 11,
                                                colours = None,
                                                denoise = True,
                                                pxwindow = 3,
                                                mask_drysand = False):
    """
    Detect wave breaking events. Two main methods are implemented:

    1 - Peak detection: detect wave breaking as lightness peaks in the timestack

        Two peak detection methods are implemented:

        1-a Local maxima. Uses peaklocalextremas() function from PyWaveLearning to
                          detect local maximas corresponding to wave breaking

        1-b Differential. Uses the first temporal derivative of the pixel intensity
                          to detect sharp transitions in the timestack that
                          should correspond to wave breaking.

        For both cases, the user can tell to the script to classifiy the identified
        pixel peaks based on known colours. For exemple, water is usually blue,
        sand is brownish and breaking waves are whiteish. Only peaks corresponding
        to wave breakin are append to the output structure.
        This is step done using classifiy_colour() from PyWaveLearning.

    2 - Edge detection: detect wave breaking as sharp edges in the timestack

        Two-options are available:

        2-a Edges only. Wave breaking events are obtained applying a sobel filter
                        to the timestack. Edge locations (time,space) are obrained as:

                        - argument of the maxima (argmax) of a cross-shore
                          pixel intenstiy series obtained at every timestamp.

                        - local maximas of a cross-shore pixel intenstiy series
                          obtained at every timestamp.

        2-b Edges and colours. Wave breaking events are obtained applying a sobel
                               filter to the timestack and the detected Edges
                               are classified using the colour information as in 1-a.
                               Edge locations (time,space) are obrained as:

                               - argument of the maxima (argmax) of a cross-shore
                                 pixel intenstiy series obtained at every timestamp.

                               - local maximas of a cross-shore pixel intenstiy series
                                 obtained at every timestamp.
    ----------
    Args:
        time (Mandatory [np.array]): Array of datetimes.

        crx_dist (Mandatory [np.array]): Array of cross-shore locations.

        rgb (Mandatory [np.array]): timestack array. Shape is [time,crx_dist,3].

        crx_start (Optional [float]): where in the cross-shore orientation to
                                       start the analysis. Default is crx_dist.min().

        crx_start (Optional [float]): where in the cross-shore orientation to
                                       finish the analysis. Default is crx_dist.max().

        px_mtrc (Optional [float]): Which pixel intenstiy metric to use.
                                     Default is "lightness".

        resample_rule (Optional [str]): To which frequency interpolate timeseries
                                          Default is  "100L".

        algorithm (Optional [str]): Wave breaking detection algorithm.
                                     Default is "peaks".

        peak_detection (Optional [str]): Peak detection algorithm.
                                          Default is  "local_maxima".

        threshold (Optional [float]): Threshold for peak detection algorithm.
                                     Default is 0.1

        tswindow (Optional [int]): Window for peak detection algorithm.
                                    Default is 11.

        denoise (Optional [bool]): = Denoise timestack using denoise_bilateral
                                     Default is True.

        pxwindow (Optional [int]): Window for denoise_bilateral. Default is 3.

        posterize (Optional [bool]): If true will reduce the number of colours in
                                     the timestack. Default is False.

        ncolours (Optional [str]): Number of colours to posterize. Default is 16.

        colours (Optional [dict]): A dictionary for the colour learning step.
                                    Something like:
                                    train_colours = {'labels':[0,1,2],
                                                     'aliases':["sand","water","foam"],
                                                     'rgb':[[195,185,155],
                                                            [30,75,75],
                                                            [255,255,255]]
                                                     'target':2}
                                    Default is None.

        mask_drysand (Experimental [bool]) = Mask dry sand using a color-temperature
                                           relationship. Default is False.
    ----------
    Return:
         time (Mandatory [np.array]): time of occurance of wave breaking events.

         breakers (Mandatory [np.array]): cross-shore location of wave breaking events.
    """

    if not crx_start:
        crx_start = crx_dist.min()
        crx_end = crx_dist.max()

    if posterize:
        print ("  + >> posterizing")
        rgb = colour_quantization(rgb,ncolours=ncolours)

    # get colour data
    if algorithm == "colour" or algorithm == "edges_and_colour":
        target = colours["target"]
        labels = colours["labels"]
        dom_colours = colours["rgb"]

    # denoise a little bedore computing edges
    if denoise:
        rgb = denoise_bilateral(rgb,pxwindow,multichannel=True)
        # scale back to 0-255
        rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())*255

    # mask sand - Not fully tested
    if mask_drysand:
        # calculate colour temperature
        cct = colour.xy_to_CCT_Hernandez1999(colour.XYZ_to_xy(colour.sRGB_to_XYZ(rgb/255)))
        # scale back to 0-1
        cct = (cct-cct.min())/(cct.max()-cct.min())*255
        # mask
        i,j = np.where(cct==0)
        rgb[i,j,:] = 0

    # detect edges
    if algorithm == "edges" or algorithm == "edges_and_colour":
        print ("  + >> calculating edges")
        edges = sobel_h(rgb2grey(rgb))

    # get pixel lines and RGB values at selected locations only
    if algorithm == "peaks" or algorithm == "colour":
        print ("  + >> extracting cross-shore pixels")
        Y, crx_idx = get_analysis_locations(crx_dist, crx_start, crx_end)
        Time, PxInts, RGB = get_pixel_lines(time, rgb, crx_idx,
                                                       resample_rule = resample_rule,
                                                       pxmtc = px_mtrc)

    # get analysis frequency and a 1 sececond time window
    if not tswindow:
        fs = (time[1] - time[0]).total_seconds()
        win = np.int((1 / fs))
    else:
        win = tswindow

    print ("  + >> detecting breaking events")
    PeakTimes = []
    print_check = False
    if algorithm == "peaks" or algorithm == "colour":
        # loop over data rows
        for pxint, rgb in zip(PxInts, RGB):
                # calculate baseline
                bline = baseline(pxint,2)
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
                        max_idx = indexes(pxintdt, thres=threshold, min_dist=win)
                else:
                    raise ValueError
                # colour learning step
                if algorithm == "colour":
                    if not print_check:
                        print ("  + >> colour learning")
                        print_check = True
                    # classifiy pixels
                    breaker_idxs = []
                    for idx in max_idx:
                        y_pred = classify_colour(rgb[idx], dom_colours, labels)
                        if y_pred[0] == target:
                            breaker_idxs.append(idx)
                # peaks only
                else:
                    breaker_idxs = max_idx
                PeakTimes.append(Time[breaker_idxs])
        # organize peaks and times
        Xpeaks = []
        Ypeaks = []
        for i, pxtimes in enumerate(PeakTimes):
            for v in pxtimes:
                Xpeaks.append(v)
            for v in np.ones(len(pxtimes)) * Y[i]:
                Ypeaks.append(v)
    # edges case
    if algorithm == "edges":
        Xpeaks = []
        Ypeaks = []
        # loop in time
        for i,t in enumerate(time):
            # cross-shore line
            crx_line = edges[i,:]
            # peaks with robust peak detection
            if peak_detection == "differential" or peak_detection == "local_maxima":
                crx_line = (crx_line - crx_line.min()) / (crx_line.max() - crx_line.min())
                if not np.all(crx_line==0):
                    idx_peak = indexes(crx_line,thres=1-threshold, min_dist=win)
                # apped peaks
                for peak in idx_peak:
                    if crx_dist[peak] > crx_start and  crx_dist[peak] < crx_end:
                        Xpeaks.append(t)
                        Ypeaks.append(crx_dist[peak])
            # peaks with simple argmax - works better without colour learning
            else:
                peak = np.argmax(crx_line)
                if crx_dist[peak] > crx_start and  crx_dist[peak] < crx_end:
                    Xpeaks.append(t)
                    Ypeaks.append(crx_dist[peak])
    # edges + colour learning case
    if algorithm == "edges_and_colour":
        Ipeaks = []
        Jpeaks = []
        # loop in time
        for i,t in enumerate(time):
            # cross-shore line
            crx_line = edges[i,:]
            if peak_detection == "differential" or peak_detection == "local_maxima":
                crx_line = (crx_line - crx_line.min()) / (crx_line.max() - crx_line.min())
                # peaks
                if not np.all(crx_line==0):
                    idx_peak = indexes(crx_line,thres=1-threshold, min_dist=win)
                    if not np.all(crx_line==0):
                        idx_peak = indexes(crx_line,thres=1-threshold, min_dist=win)
                # apped peaks
                for peak in idx_peak:
                    if crx_dist[peak] > crx_start and  crx_dist[peak] < crx_end:
                        Ipeaks.append(i)
                        Jpeaks.append(peak)
            else:
                peak = np.argmax(crx_line)
                if crx_dist[peak] > crx_start and  crx_dist[peak] < crx_end:
                    Ipeaks.append(i)
                    Jpeaks.append(peak)
        # colour learning step
        Xpeaks = []
        Ypeaks = []
        for i,j in zip(Ipeaks,Jpeaks):
            if not print_check:
                print ("  + >> colour learning")
                print_check = True
            # classify colour
            y_pred = classify_colour(rgb[i,j,:], dom_colours, labels)
            if y_pred[0] == target:
                Xpeaks.append(time[i])
                Ypeaks.append(crx_dist[j])

    # sort values in time and outout
    y = np.array(Ypeaks)[np.argsort(date2num(Xpeaks))]
    x = np.array(Xpeaks)[np.argsort(Xpeaks)]

    return ellapsedseconds(x), y

def dbscan(times,breakers,dbs_eps=0.01,dbs_msp=20,dbs_mtc="sqeuclidean"):
    """
    Wrapper around sklearn.cluster.DBSCAN.

    ----------
    Args:
        times (Mandatory [np.array]): array of times (in seconds).

        breakers (Mandatory [np.array]): array of breaker locations (in meters).

        dbs_eps (Optional [float]): minimum distance for DBSCAN

        dbs_msp (Optional [int]): minimum number of samples for DBSCAN

        dbs_mtc (Optional [str]): which distance metric to use.
    ----------
    Returns:
        df_dbscan (Mandatory [pandas.DataFrame]): DBSCAN results
    """

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


def write_outputs(path,basename,stk_secs,crx_dist,rgb,df_dbscan):
    """
    Write oututs in several common formats. Implemented options are
    "csv", "raster", "shapefile" and "geojson".

    ----------
    Args:
        path (Mandatory [str]): Output path.

        basename (Mandatory [str]): Basename.

        stk_secs (Mandatory [np.array]): Array of times (in seconds).

        crx_dist (Mandatory [np.array]): Array of cross-shore locations (in meters).

        rgb (Mandatory [np.array]): Timestack array (shape is time x crx_dist x 3).

        df_dbscan (Mandatory [pd.DataFrame]): Output of dbscan()
    ----------
    Returns:

    """

    ## csv output ##
    df_dbscan.to_csv(os.path.join(path,basename+".csv"))

    ## shapefile output ##
    geometry = [Point(xy) for xy in zip(df_dbscan["time"], df_dbscan["breaker"])]
    crs = {'init': 'epsg:4326'}
    geo_df = gpd.GeoDataFrame(df_dbscan, crs=crs, geometry=geometry)
    geo_df.to_file(driver='ESRI Shapefile', filename=os.path.join(path,basename+".shp"))

    # geojson output
    if os.path.isfile(os.path.join(path,basename+".geojson")):
        os.remove(os.path.join(path,basename+".geojson"))
    geo_df.to_file(filename=os.path.join(path,basename+".geojson"), driver="GeoJSON")

    ## raster output ##
    # information for .points and .wld
    dx = np.diff(stk_secs).mean()
    dy = np.diff(crx_dist).mean()
    x = stk_secs.max()
    y = crx_dist.max()-crx_dist.min()
    i = rgb.shape[1]
    j = rgb.shape[0]
    # dump stack to a jpeg
    imsave(os.path.join(path,basename+".jpg"), np.flipud(np.rollaxis(rgb, 1, 0)))
    # write world file
    with open(os.path.join(path,basename+".wld"),"w") as f:
        f.write('{}\n'.format(dx))
        f.write('0\n')
        f.write('0\n')
        f.write('-{}\n'.format(dy))
        f.write('0.0\n')
        f.write(str(y)+"\n")
    f.close()
    # write points file
    with open(os.path.join(path,basename+".points"),"w") as f:
        f.write('mapX,mapY,pixelX,pixelY,enable\n')
        f.write('0,0,0.00,-{},1\n'.format(i))
        f.write('{},{},{},-0.00,1\n'.format(x,y,j))
    f.close()


def main():
    """ Run the main script """

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
    basename = H["data"]["basename"]

    ## get analysis parameters ##

    # which algorithm to use to detect breaking events
    brk_alg = H["parameters"]["breaking_algorithm"]
    if brk_alg not in ["colour","peaks","edges","edges_and_colour"]:
            raise ValueError("Breaking detection algorithm should be either \'colour\', \'peaks\', \'edges\' or \'edges_and_colour\'.")
    # metric to use for pixel intensity
    px_mtrc = H["parameters"]["pixel_metric"]
    if px_mtrc not in ["lightness","intensity"]:
        raise ValueError("Pixel metric should be either \'lightness\' or \'intensity\'.")
    # colour quantization
    qnt_cl = H["parameters"]["colour_quantization"]
    n_clrs = H["parameters"]["quantization_colours"]
    # peak detection method
    pxp_mtd = H["parameters"]["peak_detection_algorithm"]
    # threshold for peak detection
    px_trsh = H["parameters"]["pixel_threshold"]
    # pixel window for denoise_bilateral
    px_wndw = H["parameters"]["pixel_window"]
    # time window for peak detection
    ts_wndw = H["parameters"]["time_window"]
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
    px_wind = H["parameters"]["gui_window"]
    # try to fix bad pixel values
    gap_val = H["parameters"]["gap_value"]
    # plot?
    plot = H["parameters"]["plot"]

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

    ### run the analysis ###
    if brk_alg in ["colour","edges_and_colour"]:
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
        train_colours = {'labels':[0,1,2],
                         'aliases':["sand","water","foam"],
                         'rgb':[sand,water,foam],
                         'target':2}
    else:
        train_colours = None

    # detect breaking events
    times, breakers = detect_breaking_events(stk_time, crx_dist, rgb,
                                                        crx_start = crx_start,
                                                        crx_end = crx_end,
                                                        tswindow = ts_wndw,
                                                        pxwindow = px_wndw,
                                                        px_mtrc = px_mtrc,
                                                        resample_rule = rs_rule,
                                                        algorithm = brk_alg,
                                                        peak_detection = pxp_mtd,
                                                        posterize = qnt_cl,
                                                        ncolours = n_clrs,
                                                        threshold = px_trsh,
                                                        colours = train_colours)
    ## DBSCAN ##
    print ("  + >> clustering wave paths")
    df_dbscan = dbscan(times,breakers,dbs_eps,dbs_msp,dbs_mtc)

    ## Outputs ##
    print ("  + >> writting output")
    write_outputs(outputto,basename,stk_secs,crx_dist,rgb,df_dbscan)
    #
    # if qnt_cl:
    #     rgb = colour_quantization(rgb,ncolours=n_clrs)

    if plot:
        print ("  + >> plotting")
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.8])
        ax2 = fig.add_axes([0.85, 0.1, 0.1, 0.8])

        # get unique colours for each wave
        colors = plt.cm.get_cmap("Set1", len(np.unique(df_dbscan["wave"].values)))

        # get traning colours
        if train_colours:
            for label, region, color in zip(train_colours["labels"],
                                            train_colours["aliases"],
                                            train_colours["rgb"]):
                ax2.scatter(1, label, 1000, marker="s",
                                            color=color/255,
                                            label=region,
                                            edgecolor="k", lw=2)
            # set axis
            ax2.set_yticks(train_colours["labels"])
            ax2.set_yticklabels(train_colours["aliases"])
            ax2.set_ylim(-0.25,len(train_colours["aliases"])-1+0.25)
            for tick in ax2.get_yticklabels():
                tick.set_rotation(90)
            ax2.xaxis.set_major_locator(plt.NullLocator())
        else:
            for label, region, color in zip([0,1,2],
                                            ['N/A','N/A','N/A'],
                                            [[1,1,1],[1,1,1],[1,11]]):
                ax2.scatter(1, label, 1000, marker="s",
                                            color=[1,1,1],
                                            label=region,
                                            edgecolor="k", lw=2)
            # set axis
            ax2.set_yticks([0,1,2])
            ax2.set_yticklabels(['N/A','N/A','N/A'])
            ax2.set_ylim(-0.25,len([0,1,2])-1+0.25)
            for tick in ax2.get_yticklabels():
                tick.set_rotation(90)
            ax2.xaxis.set_major_locator(plt.NullLocator())
        ax2.set_title("Training colours")

        # plot timestack
        im = ax1.pcolormesh(stk_secs, crx_dist, rgb.mean(axis=2).T)
        # set timestack to  to true color
        rgba = construct_rgba_vector(np.rollaxis(rgb, 1, 0), n_alpha=0)
        rgba[rgba > 1] = 1
        im.set_array(None)
        im.set_edgecolor('none')
        im.set_facecolor(rgba)

        # plot analysis limits
        ax1.axhline(y=crx_start, lw=3,color="darkred", ls="--")
        ax1.axhline(y=crx_end, lw=3, color="darkred", ls="--")

        # plot all identified breaking events
        ax1.scatter(times, breakers, 20, color="k",alpha=0.5, marker="+", lw=1,zorder=10)

        # DBSCAN plot
        k = 0
        for label, group in df_dbscan.groupby("wave"):
            if label > -1:
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

        # set axes limits
        ax1.set_xlim(xmin=stk_secs.min(), xmax=stk_secs.max())
        ax1.set_ylim(ymin=crx_dist.min(), ymax=crx_dist.max())

        # set axes labels
        ax1.set_ylabel(r"Cross-shore distance $[m]$",fontsize=16)
        ax1.set_xlabel(r"Time $[s]$",fontsize=16)

        # seaborn despine
        sns.despine(ax=ax2, bottom=True)
        sns.despine(ax=ax1)

        fig.tight_layout()

        plt.show()

if __name__ == '__main__':

    print("\nDetecting wave breaking, please wait...\n")

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

    # main
    main()

    print("\nMy work is done!\n")
