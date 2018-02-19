# iterative tools
import operator

# pandas and numpy
import numpy as np
import pandas as pd

# random numbers
import random

# strings
import string

# signal tools from scipy
from scipy import signal

# spatial tools from scipy
import scipy.spatial

# date helpers
import datetime
from matplotlib.dates import date2num, num2date


def peaklocalextremas(y, lookahead=16, delta=0.1, x=None):
    """
    Find local extremas (minimas and maximas) using a recursive algorithm

    ----------
    Args:
        x (Mandatory [tuple,np.ndarray]): 1xN data array.

        lookahead (Optional [int]): Distance to look ahead from a peak
        candidate to determine if it is the actual peak. (samples/period)/f
        where 4>=f>=1.25 might be a good value. Defalt is 10 (~ 1 sec)

        delta (Optional [float]): this specifies a minimum difference between a
        peak and the following points, before a peak may be considered a peak.
        Useful to hinder the function from picking up false peaks towards to
        end of the signal. To work well delta should be set to delta
        >= RMSnoise * 5. When omitted delta function causes a 20x decrease in
        speed. When used correctly it can double the speed of the function.
        Default is 0.05.

        y (Optional [tuple, np.ndarray]): time array. If empty will use
        range(len(wave)).

    ----------
    Returns:
        minimas, maximas (Mandatory [np.ndarray]) local mininas and maximas
        arrays.
    """

    max_peaks = []
    min_peaks = []

    dump = []   # Used to pop the first hit which almost always is false

    # input check
    if x is None:
        x = range(len(y))

    if len(x) != len(y):
        raise ValueError("Input vectors wave and time must have same length")

    # translate to be a numpy arrays
    X = np.array(x)
    Y = np.array(y)

    # store data length for later use
    length = len(X)

    # perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    # maxima and minima candidates are temporarily stored in
    # mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    # only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(X[:-lookahead],
                                       Y[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        # look for maximas
        if y < mx - delta and mx != np.Inf:
            # maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if Y[index:index + lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                # set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue
            # else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = time[np.where(wave[index:index+lookahead]==mx)]
        # look for minimas
        if y > mn + delta and mn != -np.Inf:
            # minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if Y[index:index + lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                # set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
            # else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = time[np.where(wave[index:index+lookahead]==mn)]
    # remove the false hit on the first value of the wave
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except Exception:
        raise IndexError()
        # pass
        # no peaks were found, should the function return empty lists?
    try:
        return np.array(min_peaks)[:, 0].astype(int), \
               np.array(max_peaks)[:, 0].astype(int)
    except Exception:
        return np.array([0, 0]), np.array([0, 0])


def ellapsedseconds(times):
    """
    Auxiliary function to count how many (fractions) of seconds have passed
    from the start of a stationary series.

    ----------
    Args:
        time [Mandatory (pandas.to_datetime(array)]: array of timestamps

    ----------
    Returns:
        seconds [Mandatory (np.ndarray)]: array of ellapsed seconds.

    """

    seconds = []
    for t in range(len(times)):
        dt = (times[t]-times[0]).total_seconds()
        seconds.append(round(dt, 3))

    return np.array(seconds)


def fixtime(times, timedelta=0):
    """
    Fix time based on offsets. Always return a array of datetimes no matter
    what kind of input.

    ----------
    Args:
        times [Mandatory (np.ndarray, tuple or list)]: time vector

        timedelta [Optional (float)]: time offset in seconds.

    ----------
    Returns:
        fixed [Mandatory (np.ndarray of datetimes)]: fixed times
    """

    fixed = []
    for time in pd.to_datetime(times):
        now = time.to_pydatetime()+datetime.timedelta(seconds=timedelta)
        fixed.append(now)

    return np.array(fixed)


def dffs(df):
    """
    Auxiliary function do get sample frequency give a DataFrame
    in which the indexses are temporal.

    ----------
    Args:
        df [Mandatory (pd.DataFrame)]: dataframe with time info as indexes

    ----------
    Returns:
        fs [Mandatory (float)]: sample frequency in Hz
    """

    times = fixtime(df.index.values)

    # TODO: raise an error if the indexes are not time values

    fs = 1./(times[1]-times[0]).total_seconds()

    return int(fs)


def timeindexes(times, t1, t2):
    """
    Nearest neighbour search in time vectors

    ----------
    Args:
        times [Mandatory (np.ndarray, tuple or list)]: vector of datetimes

        t1,t2 [Mandatory (datetime.datetime)]: start and end times for search

    ----------
    Returns:
        i2,i2 [Mandatory (in)]: start and end indexes
    """

    # variables
    t = date2num(times)
    i = np.arange(0, len(t), 1)
    t1 = date2num(t1)
    t2 = date2num(t2)

    # indexes
    i1 = np.abs(t - t1).argmin()
    i2 = np.abs(t - t2).argmin()

    return i1, i2


def cross_correlation_fft(a, b, mode='valid'):
    """
    Cross correlation between two 1D signals. Similar to np.correlate,
    but faster.

    ----------
    Args:
        a [Mandatory (np.array)]: signal "a" array (1D)

        b [Mandatory (np.array)]: signal "b" array (1D)

        mode [Optional (string)]: mode option for np.fft.convolve()

    ----------
    Returns:
        r [Madatory, (np.array)] Correlation coefficients.
                                 Shape depends on mode.
    """

    a = np.asarray(a)
    b = np.asarray(b)
    if np.prod(a.ndim) > 1 or np.prod(b.ndim) > 1:
        raise ValueError('Can only vectorize vectors')
    if len(b) > len(a):
        a, b = b, a
    n = len(a)

    # Pad vector
    c = np.hstack((np.zeros(int(n/2)), b,
                   np.zeros(int(n/2) + len(a) - len(b) + 1)))

    # Convolution of reverse signal:
    return signal.fftconvolve(c, a[::-1], mode=mode)


def align_signals(a, b):
    """
    Finds optimal delay to align two 1D signals maximizes
    hstack((zeros(shift), b)) = a

    ----------
    Args:
        a [Mandatory (np.array)]: signal "a" array (1D)

        b [Mandatory (np.array)]: signal "b" array (1D)

    ----------
    Returns:
        shift [Mandatory (np.array)]: Integer that maximizes
                                  hstack((zeros(shift), b)) - a = 0
    """
    # check inputs
    a = np.asarray(a)
    b = np.asarray(b)
    if np.prod(a.ndim) > 1 or np.prod(b.ndim) > 1:
        raise ValueError('Can only vectorize vectors')
    # longest first
    sign = 1
    if len(b) > len(a):
        sign = -1
        a, b = b, a
    r = cross_correlation_fft(a, b)
    shift = np.argmax(r) - len(a) + len(a) / 2
    # deal with odd / even lengths (b doubles in size by cross_correlation_fft)
    if len(a) % 2 and len(b) % 2:
        shift += 1
    if len(a) > len(b) and len(a) % 2 and not(len(b) % 2):
        shift += 1
    return sign * shift


def find_nearest(target, val):
    """ Archaic and brute force method to find the nearest indexes to a
        given target cvalue in a inpute vector.

        >> Use np.argmin() or scipy.KDTree() instead <<

    ----------
    Args:
        target [Mandatory (np.array)]: 1d array of input data

        val [Mandatory (float)]: value to search in "target"

    ----------
    Returns:
        min_index [Mandatory (int)]: index of nearest value

        value [Mandatory (float, int)]: valeu of target[min_index]
    """
    difs = abs(target-val)
    min_index, min_value = min(enumerate(difs), key=operator.itemgetter(1))
    value = target[min_index]
    return min_index, value


def normalize(data):
    """ Normalize a vector by its variance.

        >> use sklearn.preprocessing  functions instead <<

    ----------
    Args:
        data [Mandatory (np.array)]: 1d array of input data

    ----------
    Returns:
        data [Mandatory (np.array)]: normalized data
    """
    variance = np.var(data)
    data = (data - np.mean(data)) / (np.sqrt(variance))
    return data


def nextpow2(i):
    """ Gets the next power of 2 of a given number

    ----------
    Args:
        i [Mandatory (integer)]: any integer

    ----------
    Returns:
        n [Mandatory (interger)]: next power of 2 of i
    """
    n = 1
    while n < i:
        n *= 2
    return n


def zeropad(x):
    """
    Pads an input vector to the next power of two.
    Very useful to speed up FFTs.

    ----------
    Args:
        x [Mandatory (np.array]: 1D np.array

    ----------
    Returns:
        zeropad [Mandatory (np.array]: zero-paded array
    """
    N = len(x)
    zeropad = np.zeros(nextpow2(N))
    zeropad[0:N] = x
    return zeropad


def str2bool(v):
    """
    Translate strings into booleans.

    ----------
    Args:
        v [Mandatory (str]: any string

    ----------
    Returns:
        bool [Mandatory (bool]: the boolean version of "v"
    """
    return v.lower() in ("yes", "true", "t", "1")


def chunkify(lst, n):
    """
    Chunkify a list in to "n" sub-lists. Works with numpy arrays as well.

    ----------
    Args:
        lst [Mandatory (list)]: input list or array.

        n [Mandatory (in)]: number of parts to divide the input list

    ----------
    Returns:
        lst [Mandatory (list of lists)]:
    """
    return [lst[i::n] for i in range(n)]


def kdtree(A, pt):
    _, indexes = scipy.spatial.KDTree(A).query(pt)
    return indexes


def random_string(n):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(n))


def process_timestack(ds):
    """
    Read timestack variables from netCDF.

    ----------
    Args:
        ncin (Mandatory [str]): Input file path. Should also work with
        remote files out-of-the-box.
    ----------
    Return:
        stk_secs (Mandatory [np.array]): time array

        crx_dist (Mandatory [np.array]): space array

        rgb (Mandatory [array]): RGB representation shape [time,space,3].
    """

    # get coordinates
    x = ds["x"].values
    y = ds["y"].values

    # compute distance from shore
    stk_len = np.sqrt(((x.max() - x.min()) * (x.max() - x.min())) +
                      ((y.max() - y.min()) * (y.max() - y.min())))
    stk_cross_shore_offset = y.min()
    stk_dist = np.linspace(stk_cross_shore_offset,
                           stk_len+stk_cross_shore_offset, len(x))
    # get timestack times
    stk_time = pd.to_datetime(ds["time"].values).to_pydatetime()
    stk_secs = ellapsedseconds(stk_time)
    # get RGB values
    rgb = ds["rgb"].values
    # return data
    return stk_time, stk_dist, rgb


def intersection(x1, y1, x2, y2):
    """
    Intersections of curves.

    Computes the (x,y) locations where two curves intersect.  The curves
    can be broken with NaNs or have vertical segments.

    ----------
    Args:
    x1, x2, y1, y2 (Mandatory [np.array]): arrays defining the curves
                                           (x, y) locations.
    ----------
    Returns:
    x, y (Mandatory [np.array]): coordinate(s) of the curve intersections.

    """
    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except Exception:
            T[:, i] = np.NaN

    irn = (T[0, :] >= 0) & (T[1, :] >= 0) & (T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, irn]
    xy0 = xy0.T

    return xy0[:, 0], xy0[:, 1]


def _rect_inter_inner(x1, x2):
    """ axiliary function to intersection() """

    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1

    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]

    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))

    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    """ axiliary functio to intersection() """

    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)

    return ii, jj
