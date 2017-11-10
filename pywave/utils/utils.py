def peaklocalextremas(y, lookahead=16, delta=0.1, x=None):
    """
    Find local extremas (minimas and maximas) using a recursive algorithm

    ----------
    Args:
        x (Mandatory [tuple,np.ndarray]): 1xN data array.

        lookahead (Optional [int]): Distance to look ahead from a peak candidate
        to determine if it is the actual peak. (samples/period)/f where
        4>=f>=1.25 might be a good value. Defalt is 10 (~ 1 sec)

        delta (Optional [float]): this specifies a minimum difference between a
        peak and the following points, before a peak may be considered a peak.
        Useful to hinder the function from picking up false peaks towards to end
        of the signal. To work well delta should be set to delta >= RMSnoise * 5.
        When omitted delta function causes a 20x decrease in speed. When used
        correctly it can double the speed of the function. Default is 0.05.

        y (Optional [tuple, np.ndarray]): time array. If empty will use
        range(len(wave)).

    ----------
    Return:
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
    except:
        raise IndexError()
        # pass
        # no peaks were found, should the function return empty lists?
    try:
        return np.array(min_peaks)[:, 0].astype(int), np.array(max_peaks)[:, 0].astype(int)
    except:
        return np.array([0,1,2]),np.array([0,1,2])

def dffs(df):
    """ Auxiliary function do get sample frequency give a DataFrame """
    times = fixtime(df.index.values)
    fs = 1./(times[1]-times[0]).total_seconds()
    return int(fs)

def ellapsedseconds(times):
    """ Auxiliary function to count how many (fractions) of seconds have passed
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
        seconds.append(round(dt,3))

    return np.array(seconds)


def fixtime(times,timedelta=0):
    """
    Fix time based on offsets. Always return a array of datetimes no matter
    what kind of input.

    ----------
    Args:
        times [Mandatory (np.ndarray, tuple or list)]: time vector

        timedelta [Optional (float)]: temporal offset IN SECONDS.
    ----------
    Returns:
        fixed [Mandatory (np.ndarray of datetimes)]: fixed times
    """

    fixed = []
    for time in pd.to_datetime(times):
        now = time.to_pydatetime()+datetime.timedelta(seconds=timedelta)
        fixed.append(now)

    return np.array(fixed)


def timeindexes(times,t1,t2):
    """
    Nearest neighbour search in time vectors

    ----------
    Args:
        times [Mandatory (np.ndarray, tuple or list)]: time vector

        t1,t2 [Mandatory (datetime.datetime)]: start and end times for search
    ----------
    Returns:
        i2,i2 [Mandatory (in)]: start and end indexes
    """
    from matplotlib.dates import date2num

    # variables
    t = date2num(times)
    i = np.arange(0,len(t),1)
    t1 = date2num(t1)
    t2 = date2num(t2)

    i1 = np.abs(t - t1).argmin()
    i2 = np.abs(t - t2).argmin()

    return i1,i2

#TODO: Fix Docstrings
def cross_correlation_fft(a, b, mode='valid'):
    """Cross correlation between two 1D signals. Similar to np.correlate, but
    faster.

    Parameters
    ----------
    a : np.array, shape(n)
    b : np.array, shape(m)
        If len(b) > len(a), a, b = b, a

    Output
    ------
    r : np.array
        Correlation coefficients. Shape depends on mode.
    """
    from scipy import signal
    a = np.asarray(a)
    b = np.asarray(b)
    if np.prod(a.ndim) > 1 or np.prod(b.ndim) > 1:
        raise ValueError('Can only vectorize vectors')
    if len(b) > len(a):
        a, b = b, a
    n = len(a)
    # Pad vector
    c = np.hstack((np.zeros(int(n/2)), b, np.zeros(int(n/2) + len(a) - len(b) + 1)))
    # Convolution of reverse signal:
    return signal.fftconvolve(c, a[::-1], mode=mode)

def align_signals(a, b):
    """Finds optimal delay to align two 1D signals
    maximizes hstack((zeros(shift), b)) = a

    Parameters
    ----------
    a : np.array, shape(n)
    b : np.array, shape(m)

    Output
    ------
    shift : int
        Integer that maximizes hstack((zeros(shift), b)) - a = 0
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

def find_nearest(target,val):
    import operator
    Difs                 = abs(target-val)
    min_index, min_value = min(enumerate(Difs), key=operator.itemgetter(1))
    value                = target[min_index]
    return min_index,value

def normalize(data):
    variance = np.var(data)
    data = (data - np.mean(data)) / (np.sqrt(variance))
    return data

def kdtree(A,pt):
    from scipy.spatial import KDTree
    _,indexes = scipy.spatial.KDTree(A).query(pt)
    return indexes

def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n

def zeropad(x):
    N=len(x)
    zeropad = np.zeros(nextpow2(N))
    zeropad[0:N] = x
    return zeropad

