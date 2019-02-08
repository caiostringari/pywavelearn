import numpy as np

import pandas as pd

# Least-square fits
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (RANSACRegressor,
                                  LinearRegression)

from scipy import interpolate

# Least-square fits
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (RANSACRegressor,
                                  LinearRegression)

from pywavelearn.utils import ellapsedseconds, intersection


def optimal_wavepaths(clusters, min_wave_period=1, N=50, order=2,
                      project=False, project_samples=10, proj_max=None,
                      min_slope=-1, max_proj_time=5, remove_duplicates=False,
                      duplicate_thresholds=[1, 0.5], t_weights=1):
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

        project_samples (Optional [int]): Number of samples to consider when
                                          calculating the slope to project the
                                          wavepaths back in time.

        proj_max (Optional [float]): The offshore limit for wave path
                                     projection. Defaults to the maximun
                                     observed in the input dataset.

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

            # process only waves with T >= 1s
            dt = xpred.max()-xpred.min()
            if dt > min_wave_period:

                # get the first X samples of the predicted wave path
                tproj = tpred[:project_samples]
                xproj = xpred[:project_samples]

                # build a robust linear model
                model = LinearRegression()
                model.fit(tproj.reshape(-1, 1), xproj)

                # get slope, if slope < min_slope, project
                slope = model.coef_

                if slope < min_slope:

                    # predict
                    tprojected = np.linspace(df["t"].min()-(max_proj_time*dt),
                                             tpred[project_samples], 1000)
                    xprojected = model.predict(tprojected.reshape(-1, 1))

                    # intesect with proj_max
                    zero_t = np.linspace(tprojected.min(),
                                         tprojected.max(), 10)
                    zero_y = np.ones(zero_t.shape)*proj_max
                    ti, xi = intersection(tprojected, xprojected,
                                          zero_t, zero_y)
                    if xi.size > 0:

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
        return T0, X0, T1, X1
    else:
        return T1, X1
