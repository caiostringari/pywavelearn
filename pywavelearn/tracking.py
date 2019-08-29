import numpy as np

import pandas as pd

from scipy import interpolate

# Dates
import time
import datetime
from matplotlib.dates import date2num, num2date

# image tools
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.transform import resize

from scipy.constants import g

# Least-square fits
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (RANSACRegressor,
                                  LinearRegression)

from pywavelearn.utils import ellapsedseconds, intersection

# pywavelearn tools
from pywavelearn.utils import ellapsedseconds, peaklocalextremas
from pywavelearn.image import construct_rgba_vector, pixel_window
from pywavelearn.colour import (get_dominant_colour,
                                classify_colour, colour_quantization)

from peakutils import baseline, envelope, indexes

# Colourspaces
import colour
from colorspacious import deltaE

# distances, fr checking only
from scipy.spatial.distance import __all__ as scipy_dists

# Image processing
from skimage import exposure
from skimage.io import imsave
from skimage.color import rgb2grey
from skimage.filters import sobel_h
from skimage.restoration import denoise_bilateral

# Machine learning
from sklearn.cluster import DBSCAN


def get_analysis_domain(time, y, stack):
    """
    Get analysis spatial domain from mouse clicks.

    ----------
    Args:
        time (Mandatory [np.array]): Time array [s].

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
    point = plt.ginput(n=2, timeout=1000000)
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
        points = np.array(plt.ginput(n=nclicks, show_clicks=True,
                                     timeout=1000000))
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
                             columns=["r", "g", "b"])

    # add label and region names
    df_colour["label"] = l_pixels_all
    df_colour["region"] = r_pixels_all

    return df_colour


def get_pixel_lines(stk_time, stk, Idx, resample_rule="100L",
                    pxmtc="lightness"):
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

    # case 3D
    if len(stk.shape) == 3:
        RGB = []
        pxint = []
        for idx in Idx:
            # pixel line at exact location
            pxline = stk[:, idx, :]
            # build a dataframe
            df = pd.DataFrame(
                pxline, columns=["red", "green", "blue"], index=stk_time)
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
                pxint.append(deltaE(rgb, [0, 0, 0],
                                    input_space="sRGB255") / 100)
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
            RGB.append([pxline, pxline, pxline])
    else:
        raise ValueError("Image needs to be 2D or 3D.")

    # get times
    time = pd.to_datetime(df.index.values).to_pydatetime()

    return time, pxint, RGB


def detect_breaking_events(time, crx_dist, rgb, crx_start=None,
                           crx_end=None, px_mtrc="lightness", colours=None,
                           resample_rule="100L", algorithm="peaks",
                           peak_detection="local_maxima", posterize=False,
                           ncolours=0, threshold=0.1, tswindow=11,
                           denoise=True, pxwindow=3, mask_drysand=False,
                           fix_constrast=False):
    """
    Detect wave breaking events.

    Two main methods are implemented:

    1 - Peak detection: detect wave breaking as lightness peaks in the
                        timestack

        Two peak detection methods are implemented:

        1-a Local maxima. Uses peaklocalextremas() function from pywavelearn
                          to detect local maximas corresponding to wave
                          breaking.

        1-b Differential. Uses the first temporal derivative of the pixel
                          intensity to detect sharp transitions in the
                          timestack that should correspond to wave breaking.

        In both cases, the user can tell to the script to classifiy the
        identified pixel peaks based on known colours. For exemple, water is
        usually blue, sand is brownish and breaking waves are whiteish.
        Only peaks corresponding to wave breakin are append to the output
        structure. This is step done using classifiy_colour()
        from pywavelearn.

    2 - Edge detection: detect wave breaking as sharp edges in the timestack

        Two-options are available:

        2-a Edges only. Wave breaking events are obtained applying a sobel
                        filter to the timestack. Edge locations (time,space)
                        are obrained as:

                        - argument of the maxima (argmax) of a cross-shore
                          pixel intenstiy series obtained at every timestamp.

                        - local maximas of a cross-shore pixel intenstiy series
                          obtained at every timestamp.

        2-b Edges and colours. Wave breaking events are obtained applying a
                               Sobel filter to the timestack and the detected
                               Edges are classified using the colour
                               information as in 1-a. Edge locations
                               (time,space) are obrained as:

                               - argument of the maxima (argmax) of a
                                 cross-shore pixel intenstiy series obtained
                                 at every timestamp.


                               - local maximas of a cross-shore pixel intenstiy
                                 series obtained at every timestamp.
    ----------
    Args:
        time (Mandatory [np.array]): Array of datetimes.

        crx_dist (Mandatory [np.array]): Array of cross-shore locations.

        rgb (Mandatory [np.array]): timestack array.
                                    Shape is [time,crx_dist,3].

        crx_start (Optional [float]): where in the cross-shore orientation to
                                       start the analysis.
                                       Default is crx_dist.min().

        crx_start (Optional [float]): where in the cross-shore orientation to
                                       finish the analysis.
                                       Default is crx_dist.max().

        px_mtrc (Optional [float]): Which pixel intenstiy metric to use.
                                    Default is "lightness".

        resample_rule (Optional [str]): To which frequency interpolate
                                        timeseries Default is  "100L".

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

        posterize (Optional [bool]): If true will reduce the number of colours
                                     in the timestack. Default is False.

        ncolours (Optional [str]): Number of colours to posterize.
                                   Default is 16.

        colours (Optional [dict]): A dictionary for the colour learning step.
                                    Something like:
                                    train_colours = {'labels':[0,1,2],
                                                     'aliases':
                                                     ["sand","water","foam"],
                                                     'rgb':[[195,185,155],
                                                            [30,75,75],
                                                            [255,255,255]]
                                                     'target':2}
                                    Default is None.

        mask_drysand (Experimental [bool]) = Mask dry sand using a
                                             colour-temperature (CCT)
                                             relationship. Default is False.
    ----------
    Return:
         time (Mandatory [np.array]): time of occurance of wave breaking
                                      events.

         breakers (Mandatory [np.array]): cross-shore location of wave breaking
                                          events.
    """
    if not crx_start:
        crx_start = crx_dist.min()
        crx_end = crx_dist.max()

    if posterize:
        print("  + >> posterizing")
        rgb = colour_quantization(rgb, ncolours=ncolours)

    # get colour data
    if algorithm == "colour" or algorithm == "edges_and_colour":
        target = colours["target"]
        labels = colours["labels"]
        dom_colours = colours["rgb"]

    # denoise a little bedore computing edges
    if denoise:
        rgb = denoise_bilateral(rgb, pxwindow, multichannel=True)
        # scale back to 0-255
        rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())*255

    # mask sand - Not fully tested
    if mask_drysand:
        print("  + >> masking dry sand [Experimental]")
        # calculate colour temperature
        cct = colour.xy_to_CCT_Hernandez1999(
            colour.XYZ_to_xy(colour.sRGB_to_XYZ(rgb/255)))
        # scale back to 0-1
        cct = (cct-cct.min())/(cct.max()-cct.min())*255
        # mask
        i, j = np.where(cct == 0)
        rgb[i, j, :] = 0

    if fix_constrast:
        print("  + >> fixing contrast")
        rgb = exposure.equalize_hist(rgb)
        # rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())*255

    # detect edges
    if algorithm == "edges" or algorithm == "edges_and_colour":
        print("  + >> calculating edges")
        edges = sobel_h(rgb2grey(rgb))

    # get pixel lines and RGB values at selected locations only
    if algorithm == "peaks" or algorithm == "colour":
        print("  + >> extracting cross-shore pixels")
        # rescale
        rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())*255
        Y, crx_idx = get_analysis_locations(crx_dist, crx_start, crx_end)
        Time, PxInts, RGB = get_pixel_lines(time, rgb, crx_idx,
                                            resample_rule=resample_rule,
                                            pxmtc=px_mtrc)

    # get analysis frequency and a 1 sececond time window
    if not tswindow:
        fs = (time[1] - time[0]).total_seconds()
        win = np.int((1 / fs))
    else:
        win = tswindow

    print("  + >> detecting breaking events")
    PeakTimes = []
    print_check = False
    if algorithm == "peaks" or algorithm == "colour":
        if peak_detection == "argmax":
            peak_detection = "local_maxima"
            print("  - >> setting peak detection to local maxima")
        # loop over data rows
        for pxint, rgb in zip(PxInts, RGB):
                # calculate baseline
                bline = baseline(pxint, 2)
                # calculate pixel peaks
                if peak_detection == "local_maxima":
                    _, max_idx = peaklocalextremas(pxint - bline,
                                                   lookahead=win,
                                                   delta=threshold *
                                                   (pxint - bline).max())
                elif peak_detection == "differential":
                        # calculate first derivative
                        pxintdt = np.diff(pxint - bline)
                        # remove values below zero
                        pxintdt[pxintdt <= 0] = 0
                        # scale from 0 to 1
                        pxintdt = pxintdt / pxintdt.max()
                        # get indexes
                        max_idx = indexes(pxintdt, thres=threshold,
                                          min_dist=win)
                else:
                    raise ValueError
                # colour learning step
                if algorithm == "colour":
                    if not print_check:
                        print("  + >> colour learning")
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
        for i, t in enumerate(time):
            # cross-shore line
            crx_line = edges[i, :]
            # peaks with robust peak detection
            if peak_detection == "differential" or \
               peak_detection == "local_maxima":
                crx_line = (crx_line - crx_line.min()) / (crx_line.max()
                                                          - crx_line.min())
                if not np.all(crx_line == 0):
                    idx_peak = indexes(crx_line,
                                       thres=1-threshold, min_dist=win)
                # apped peaks
                for peak in idx_peak:
                    if crx_dist[peak] > crx_start and crx_dist[peak] < crx_end:
                        Xpeaks.append(t)
                        Ypeaks.append(crx_dist[peak])
            # peaks with simple argmax - works better without colour learning
            else:
                peak = np.argmax(crx_line)
                if crx_dist[peak] > crx_start and crx_dist[peak] < crx_end:
                    Xpeaks.append(t)
                    Ypeaks.append(crx_dist[peak])
    # edges + colour learning case
    if algorithm == "edges_and_colour":
        Ipeaks = []
        Jpeaks = []
        # loop in time
        for i, t in enumerate(time):
            # cross-shore line
            crx_line = edges[i, :]
            if peak_detection == "differential" or \
               peak_detection == "local_maxima":
                crx_line = (crx_line - crx_line.min()) / (crx_line.max()
                                                          - crx_line.min())
                # peaks
                if not np.all(crx_line == 0):
                    idx_peak = indexes(crx_line, thres=1-threshold,
                                       min_dist=win)
                    if not np.all(crx_line == 0):
                        idx_peak = indexes(crx_line, thres=1-threshold,
                                           min_dist=win)
                # apped peaks
                for peak in idx_peak:
                    if crx_dist[peak] > crx_start and crx_dist[peak] < crx_end:
                        Ipeaks.append(i)
                        Jpeaks.append(peak)
            else:
                peak = np.argmax(crx_line)
                if crx_dist[peak] > crx_start and crx_dist[peak] < crx_end:
                    Ipeaks.append(i)
                    Jpeaks.append(peak)
        # colour learning step
        Xpeaks = []
        Ypeaks = []
        for i, j in zip(Ipeaks, Jpeaks):
            if not print_check:
                print("  + >> colour learning")
                print_check = True
            # classify colour
            y_pred = classify_colour(rgb[i, j, :], dom_colours, labels)
            if y_pred[0] == target:
                Xpeaks.append(time[i])
                Ypeaks.append(crx_dist[j])

    # sort values in time and outout
    y = np.array(Ypeaks)[np.argsort(date2num(Xpeaks))]
    x = np.array(Xpeaks)[np.argsort(Xpeaks)]

    return ellapsedseconds(x), y


def dbscan(times, breakers, dbs_eps=0.01, dbs_msp=20, dbs_mtc="sqeuclidean"):
    """
    Wrapper around sklearn.cluster.DBSCAN.

    ----------
    Args:
        times (Mandatory [np.array]): array of times (in seconds).

        breakers (Mandatory [np.array]): array of breaker locations
                                         (in meters).

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
    db = DBSCAN(eps=dbs_eps, min_samples=dbs_msp, metric=dbs_mtc).fit(X)

    # inliers and outliers
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # dump results into a dataframe
    df_dbscan = pd.DataFrame(np.vstack([times, breakers, labels]).T,
                             columns=["time", "breaker", "wave"])

    return df_dbscan


def write_outputs(path, basename, stk_secs, crx_dist, rgb, df_dbscan):
    """
    Write oututs in several common formats. Implemented options are
    "csv", "raster", "shapefile" and "geojson".

    ----------
    Args:
        path (Mandatory [str]): Output path.

        basename (Mandatory [str]): Basename.

        stk_secs (Mandatory [np.array]): Array of times (in seconds).

        crx_dist (Mandatory [np.array]): Array of cross-shore locations
                                         (in meters).

        rgb (Mandatory [np.array]): Timestack array
                                    (shape is time x crx_dist x 3).

        df_dbscan (Mandatory [pd.DataFrame]): Output of dbscan()
    ----------
    Returns:

    """

    # csv output
    df_dbscan.to_csv(os.path.join(path, basename+".csv"))

    #
    # FIONA IS TOO UNSTABLE TO USE IN PRODUCION
    #
    # shapefile output
    # geometry = [Point(xy) for xy in zip(df_dbscan["time"],
    # df_dbscan["breaker"])]
    # crs = {'init': 'epsg:4326'}
    # geo_df = gpd.GeoDataFrame(df_dbscan, crs=crs, geometry=geometry)
    # geo_df.to_file(driver='ESRI Shapefile',
    #                filename=os.path.join(path, basename+".shp"))
    #
    # # geojson output
    # if os.path.isfile(os.path.join(path, basename+".geojson")):
    #     os.remove(os.path.join(path, basename+".geojson"))
    # geo_df.to_file(filename=os.path.join(path, basename+".geojson"),
    #                driver="GeoJSON")

    w = shapefile.Writer(os.path.join(path, basename+".shp"),
                         shapeType=shapefile.POINT)
    w.autoBalance = 1

    w.field("time", "C")
    w.field("breaker", "C")
    w.field("wave", "C")

    # loop over rows
    for i, row in df_dbscan.iterrows():
        # create the point geometry
        w.point(float(row["time"]), float(row["breaker"]))
        # add attribute data
        w.record(row["time"], row["breaker"], row["wave"])
    # dump to
    # w.save(os.path.join(path, basename+".shp"))

    # raster output
    # information for .points and .wld
    dx = np.diff(stk_secs).mean()
    dy = np.diff(crx_dist).mean()
    x = stk_secs.max()
    y = crx_dist.max()-crx_dist.min()
    i = rgb.shape[1]
    j = rgb.shape[0]
    # dump stack to a jpeg
    imsave(os.path.join(path, basename+".jpg"),
           np.flipud(np.rollaxis(rgb, 1, 0)))
    # write world file
    with open(os.path.join(path, basename+".wld"), "w") as f:
        f.write('{}\n'.format(dx))
        f.write('0\n')
        f.write('0\n')
        f.write('-{}\n'.format(dy))
        f.write('0.0\n')
        f.write(str(y)+"\n")
    f.close()
    # write points file
    with open(os.path.join(path, basename+".points"), "w") as f:
        f.write('mapX,mapY,pixelX,pixelY,enable\n')
        f.write('0,0,0.00,-{},1\n'.format(i))
        f.write('{},{},{},-0.00,1\n'.format(x, y, j))
    f.close()


def optimal_wavepaths(clusters, min_wave_period=1, N=50, order=2,
                      project=False, project_samples=0.3, proj_max=None,
                      min_slope=-1, max_proj_time=5, remove_duplicates=False,
                      duplicate_thresholds=[1, 0.5], t_weights=1,
                      nproj=50):
    """
    Compute optimal wavepaths from a DBSCAN output.

    Input dataframe must have to have 3 columns named "x", "t" and "wave"
    for this function to work.

    ----------
    Args:
        clusters (Mandatory [pandas.DataFrame]): input dataframe.

        min_wave_period (Mandatory [pandas.DataFrame]): input dataframe.

        N (Optional [int]): Number of points in a unique wavepath.

        project (Optional [str]): If true, project the wavepaths.
                                  False by default.

        project_samples (Optional [int]): Fraction  of samples to consider when
                                          calculating the slope to project the
                                          wavepaths back in time.

        proj_max (Optional [float]): The offshore limit for wave path
                                     projection. Defaults to the maximun
                                     observed in the input dataset.

        nproj (Optional [int]): maximun number of samples in a projected wave
                                path. Default is 50.

        min_slope (Optional [float]): Minimum slope to wave to be considered
                                      propagating shorewards. Default is -1.

        max_proj_time (Optional [float]): Maximun projection time in terms of
                                          of number of multiples of the wave
                                          period.

        t_weights (Optional [float]): Time threshold to consider when computing
                                      the wights from the run-up max.
                                      Default is 1 second.

        remove_duplicates (Optional [int]): NOT IMPLEMENTED.

        duplicate_thresholds (Optional [int]): NOT IMPLEMENTED.

    ----------
    if project
        Returns:
            T0 (Mandatory [list of np.array]): array of time (seconds)
                                               projected breaking events

            X0 (Mandatory [list of np.array]): array of cross-shore breaking
                                               projected locations

            T1 (Mandatory [list of np.array]): array of time (seconds)
                                               breaking events

            X1 (Mandatory [list of np.array]): array of cross-shore breaking
                                               locations.
    else
        Returns:
            T1 (Mandatory [list of np.array]): array of time (seconds)
                                               breaking events.

            X1 (Mandatory [list of np.array]): arrays of cross-shore breaking
                                               locations.
    """
    # defensive programing
    for v in ["t", "x", "wave"]:
        if v not in clusters.columns.values:
            raise ValueError("No columns named '{}' was found.".format(v))

    # if no proj_max, use the location of the offshore most wave path
    if not proj_max:
        proj_max = clusters["x"].max()

    # data output
    T0 = []  # offshore projections
    X0 = []  # offshore projections
    X1 = []  # optimal wave path
    T1 = []  # optimal wave path
    L = []  # wave labels
    Tstart = []  # auxiliary variables
    Sstart = []  # auxiliary variables

    # loop over each group
    for cluster, df in clusters.groupby("wave"):

        # first, build and OLS model to get the path slope:
        model = LinearRegression()
        model.fit(df["t"].values.reshape(-1, 1), df["x"].values)

        # get slope, if slope < min_slope, project
        slope = model.coef_

        # if slope < thrx, proceed
        if slope < min_slope:

            # build the higher-order model
            model = Pipeline([('poly', PolynomialFeatures(degree=order)),
                              ('ols', LinearRegression(fit_intercept=False))])

            # give the shore-most points of the wavepath more weight
            weights = np.ones(len(df["x"].values))
            weights[0] = 100  # fisrt seaward-point
            tsrc = df["t"].values
            tidx = df["t"].values.max()-t_weights
            idx = np.argmin(np.abs(tsrc-tidx))
            weights[idx:] = 100

            # fit
            model.fit(df["t"].values.reshape(-1, 1), df["x"].values,
                      **{'ols__sample_weight': weights})

            # predict offshore
            tpred = np.linspace(df["t"].min(), df["t"].max(), N)
            xpred = model.predict(tpred.reshape(-1, 1))

            X1.append(xpred)  # insiders
            T1.append(tpred)  # insider
            L.append([cluster]*len(X1))

            # process only waves with T >= 1s
            dt = xpred.max()-xpred.min()
            if dt > min_wave_period:

                # get the first X samples of the predicted wave path
                n_project_samples = int(project_samples*len(tpred))
                # print(type(n_project_samples))
                tproj = tpred[:n_project_samples]
                xproj = xpred[:n_project_samples]

                # build a robust linear model
                model = LinearRegression()
                model.fit(tproj.reshape(-1, 1), xproj)

                # get slope, if slope < min_slope, project
                slope = model.coef_

                if slope < min_slope:

                    # predict
                    tprojected = np.linspace(df["t"].min()-(max_proj_time*dt),
                                             tpred[n_project_samples], nproj)
                    xprojected = model.predict(tprojected.reshape(-1, 1))

                    # intesect with proj_max
                    zero_t = np.linspace(tprojected.min(),
                                         tprojected.max(), nproj)
                    zero_y = np.ones(zero_t.shape)*proj_max
                    ti, xi = intersection(tprojected, xprojected,
                                          zero_t, zero_y)
                    if xi.size > 1:

                        # slice
                        idx = np.argmin(np.abs(xprojected-xi))
                        xslice = xprojected[idx:]
                        tslice = tprojected[idx:]

                        # intert to N
                        f = interpolate.interp1d(tslice, xslice)
                        tprojected = np.linspace(tslice.min(),
                                                 tslice.max(), N)
                        xprojected = f(tprojected)

                        # try to remove duplicated wavepaths
                        # not working yet
                        Tstart.append(tprojected.min())
                        Sstart.append(slope[0])

                    # else, append zeros
                    else:
                        tprojected = np.zeros(N)
                        xprojected = np.zeros(N)
                        Tstart.append(np.nan)
                        Sstart.append(np.nan)

                # else, append zeros
                else:
                    tprojected = np.zeros(N)
                    xprojected = np.zeros(N)
                    Tstart.append(np.nan)
                    Sstart.append(np.nan)

                # append
                X0.append(xprojected)  # projections
                T0.append(tprojected)  # projections

    # try to remove duplicate projections
    if remove_duplicates:
        raise NotImplementedError("Duplicate removel not implemented yet.")

    if project:
        return T0, X0, T1, X1, L
    else:
        return T1, X1, L


def project_rundown(T, X, slope=0.1, timestep=0.1, k=1):
    """
    Project the run-down curves given the run-up curves.

    Uses the simple fact that the only force driving the run-down is gravity

    ----------
    Args:
        T (Mandatory [list of np.array]): arrays of time (seconds)
                                          of run-up events.

        X (Mandatory [list of np.array]): arrays of cross-shore run-up
                                          locations.

        slope (Optional [float]): beach slope. Default is 0.1.

        timestep (Optinal [float]): timestep for the temporal integration.

    ----------
    Return:
        Tproj (Mandatory [list of np.array]): arrays of time (seconds)
                                              run-down events.

        Xproj (Mandatory [list of np.array]): arrays of cross-shore locations
                                              of run-down events.
    """
    # project run-downs
    Tproj = []
    Xproj = []
    for x, t in zip(X, T):
        # projection time vectior
        tproj = np.arange(0, (t.max()-t.min())*k, timestep)
        # maximun runup
        max_runrup = x.max()
        # integrate
        rd = 0.5 * (g*(slope/np.sqrt(1+slope**2))) * tproj**2
        # shift to match max runup
        rd += x.min()
        # append
        Tproj.append(tproj+t.max())
        Xproj.append(rd)
    return Tproj, Xproj


def swash_edge(time, dist, T, X, sigma=0.5, shift=0):
    """
    Get and smooth and continuos swash excursion curve.

    Uses an wavelet filter to smooth things on the background.

    ----------
    Args:
        time (Mandatory [np.array]): target array of times (seconds).

        dist (Mandatory [np.array]): target array of cross-shore locations (m).

        T (Mandatory [list of np.array]): arrays of time (seconds)
                                          of merged run-ups and run-downs.

        X (Mandatory [list of np.array]): arrays of merged cross-shore run-up
                                          and rund-down locations.

    ----------
    Return:
        swash_time (Mandatory [np.array]): array of swash time events.

        swash_crxs (Mandatory [np.array]): array of swash locations events.
    """

    # rasterize
    try:
        S, xedges, yedges = np.histogram2d(T,
                                           X,
                                           bins=(time, dist),
                                           normed=True)
    except Exception:
        sorts = np.argsort(time)
        time = time[sorts]
        sorts = np.argsort(dist)
        dist = dist[sorts]
        S, xedges, yedges = np.histogram2d(T,
                                           X,
                                           bins=(time, dist),
                                           normed=True)

    # bring back to original dimension
    S = resize(S, (time.shape[0], dist.shape[0]))
    S[S > 0] = 1

    # get the edge
    S = canny(gaussian(S, sigma=sigma))
    S[S > 0] = 1

    # loop over time, get the edge(t, x) position
    swash_time = []
    swash_crxs = []
    for i, t in enumerate(time):
        strip = S[i, :]
        idx = np.where(strip == 1)[0]
        if idx.size > 0:
            swash_time.append(t)
            swash_crxs.append(dist[idx[0]+shift])

    # sort in time
    sorts = np.argsort(swash_time)

    return np.array(swash_time)[sorts], np.array(swash_crxs)[sorts]


def merge_runup_rundown(T1, X1, T2, X2):
    """
    Merge run-up and run-down curves.

    ----------
    Args:
        T1, T2 (Mandatory [list of np.array]): arrays of time (seconds)
                                               of run-up (T1) and run-down (T2)
                                               events.

        X1, X2 (Mandatory [list of np.array]): arrays of cross-shore run-up
                                               (X1) and rund-down (X2)
                                               locations.

    ----------
    Returns:
        Tm (Mandatory [list of np.array]): merged arrays of time (seconds)
                                           run-down events.

        Xm (Mandatory [list of np.array]): merged arrays of cross-shore
                                           locations of run-down events.
    """

    # merge
    Tm = []
    Xm = []
    for t1, x1 in zip(T1, X1):
        for t2, x2 in zip(T2, X2):
            for v1 in t1:
                Tm.append(v1)
            for v2 in t2:
                Tm.append(v2)
            for v1 in x1:
                Xm.append(v1)
            for v2 in x2:
                Xm.append(v2)
    Tm = np.array(Tm)
    Xm = np.array(Xm)

    # sort data
    sorts = np.argsort(Tm)

    return Tm[sorts], Xm[sorts]
