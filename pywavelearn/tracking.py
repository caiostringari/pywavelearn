import numpy as np

import pandas as pd

from scipy import interpolate

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
