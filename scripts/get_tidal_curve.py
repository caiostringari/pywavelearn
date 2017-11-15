#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
#
# SCRIPT   : tidal_curve.py
# POURPOSE : Compute the tidal curv for a given pressure transducer dataset.
#            Use the pt2nc.py or get_uniquedays.py scripts to generate valid input netcdf files.
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 19/07/2017 [Caio Stringari]
#
# OBSERVATIONS  :
#
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
#
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import print_function,division

# System
import sys,os
# Dates
import datetime

# Data
import xarray as xr
import pandas as pd

# Arguments
import argparse

# Numpy
import numpy as np

# linear regression
from scipy.stats import linregress
from scipy import interpolate

# scipy ffts
import scipy.signal


from wavetools import ellapsedseconds,nextpow2,zeropad,wave_number,dffs
from matplotlib.dates import date2num,num2date

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper", font_scale = 2.0, rc={"lines.linewidth":2.0})
sns.set_style("ticks",{'axes.linewidth' : 2,
                       'axes.linestyle' : u"--",
                       'legend.frameon' : True,
                       'xtick.major.size' : 5.0,
                       'xtick.minor.size' : 5.0,
                       'ytick.major.size' : 5.0,
                       'ytick.minor.size' : 5.0})

### Functions ###


def tidal_timestamps(df,threshold=0.1):
    """ Compute tidal time stamps from raw pressure transducer data.
    Divide the timeseries in "dt" long segments. 
    Uses a linear regression to compute tidal tendency and a distribution test
    to only select segments with unimodal water depth distributions.


    ----------
    Args:
        df [Mandatory (pandas.DataFrame)]: Pandas DataFrame with pressure data.
                                           at least one of the columns must be
                                           called "pressure". Index must be date-like

        threshold [Optional (float)]: Threshold for the linear regression to be considered
                                      greater than 0.
    ----------
    Returns:
        Time [Mandatory (np.array): array of datetimes. Values correspond to the center
                                of the time interval analysed

        Tide [Mandatory (np.array): array of tidal water levels centered at "Time"
    """

    # total duration of the records
    duration = (df.index[-1]-df.index[0]).total_seconds()/60
    # number of segments
    nsegments = int(duration/dt)
    # start and end datetime of the recores
    start = df.index[0].to_pydatetime()
    end = df.index[-1].to_pydatetime()

    Tide = []
    Time = []
    
    # Loop over segments
    for c in range(nsegments):

        # Start and end of the segment time interval
        t1 = start+pd.Timedelta(minutes=dt*(c))
        t2 = t1+pd.Timedelta(minutes=dt)

        # DataFrame segment
        dfc = df[t1:t2].copy().reset_index()
        dfc["seconds"] = ellapsedseconds(pd.to_datetime(dfc.time.values))
        # dfc.index = dfc.time; dfc.drop("time",axis=1,inplace=True)


         # Two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero.
        slope,intercept,r,p,std = linregress(dfc["seconds"].values,dfc["pressure"].values)
        m = dfc["pressure"].values.mean()
        d = dfc["pressure"].values.max()-dfc["pressure"].values.min()

        
        # if p value, mean and amplitude > treshold, segment is good for analysis
        if p > 0 and m > 0 and d > threshold:

            # verify if data is unimodal
            xgrid = np.linspace(0,dfc["pressure"].max()*1.25, len(dfc),)
            multimodal, npeaks = ismultimodal(dfc["pressure"], xgrid, bandwidth=0.1,plot=False)

            # If data is unimodal, compute tides
            if multimodal == False:
                Tide.append(dfc["pressure"].mean())
                Time.append(pd.to_datetime(dfc["time"].values[int(dfc.count().values[0]/2)]).to_pydatetime())

    return np.array(Time), np.array(Tide)


def tidal_tendency(tides):
    """ Compute tidal tendency based on tidal curve slope.
    If slope greater than 0, tide is flooding, otherwise, tide is ebbing.

    ----------
    Args:
        tides [Mandatory (np.array)]: Array with tidal water levels

    ----------
    Returns:
        tendencies [Mandatory (np.array): string array with tendency values.
    """

    tendencies = []
    for slope in np.diff(tides):
        if slope >= 0:
            tendency = "flood"
        elif slope < 0:
            tendency = "ebb"
        tendencies.append(tendency)
    tendencies.append(tendency)
    return np.array(tendencies)

def ismultimodal(x, xgrid, bandwidth=0.1,threshold=0.1,plot=False,**kwargs):
    """ Compute if sample data is unimodal using gaussian kernel density funcions.

    ----------
    Args:
        x [Mandatory (np.array)]: Array with water levels

        xgrid [Mandatory (np.array)]: Array of values in which the KDE is computed.

        bandwidth [Mandatory (np.array)]: KDE bandwith. Note that scipy weights its
                                          bandwidth by the covariance of the
                                         input data.  To make the results comparable
                                         to the other methods, we divide the bandwidth 
                                         by the sample standard deviation.

        threshold [Optional (float)]: Threshold for peak detection. Default is 0.1

        plot [Optional (bool)]: If True, plot the results. Default is false

        **kwargs [Optional (dict)]: scipy.stats.gausian_kde kwargs

    ----------
    Returns:
        multimodal [Mandatory (bool): If True, distribution is multimodal.

        npeaks [Mandatory (bool)]: number of peaks in the distribution.
    """
    # 

    from scipy.stats import gaussian_kde

    # start multimodal as false
    multimodal = False

    # compute gaussian KDE
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs).evaluate(xgrid)
    # print (kde)

    # compute how many peaks in the distribution
    root_mean_square = np.sqrt(np.sum(np.square(kde) / len(kde)))
    # compute peak to average ratios
    ratios = np.array([pow(x / root_mean_square, 2) for x in kde])
    # apply first order logic
    peaks = (ratios > np.roll(ratios, 1)) & (ratios > np.roll(ratios, -1)) & (ratios > threshold)
    # optional: return peak indices
    peak_indexes = []
    for i in range(0, len(peaks)):
        if peaks[i]:
            peak_indexes.append(i)
    npeaks = len(peak_indexes)
    
    # if more than one peak, distribution is multimodal
    if npeaks>1:
        multimodal = True

    if plot:
        fig,ax = plt.subplots()
        plt.plot(xgrid,kde,"-k",lw=3,zorder=10)
        plt.scatter(xgrid[peaks],kde[peaks],120,"r",zorder=11)
        plt.hist(x,bins=20,normed=True,alpha=0.5)
        ax.grid()
        try:
            sns.despine()
        except:
            pass
        plt.show()

    return multimodal,npeaks




### Main calls ###
if __name__ == '__main__':

    print ("\nSpliting data in stationary intervals relative to tides, please wait...\n")

    ### Argument parser
    parser = argparse.ArgumentParser()

    # Input data
    parser.add_argument('--input','-i',
                        nargs = 1,
                        action = 'store',
                        dest = 'input',
                        help = "Input data path with converted netCDF files.",
                        required = True)
    # Output data
    parser.add_argument('--output','-o',
                        nargs = 1,
                        action = 'store',
                        dest = 'output',
                        help = "Output file name (with no extension).",
                        required = True)
    # Output frequency
    parser.add_argument('--output-frequency','-f',
                        nargs = 1,
                        action = 'store',
                        dest = 'ofreq',
                        default = [5],
                        help = "Output frequency in minutes. Default is 5 minutes",
                        required = False)
    # Time interval of the analysis
    parser.add_argument( '--dt','-dt',
                        nargs = 1,
                        action = 'store',
                        dest = 'dt',
                        default = [60],
                        help = "Time interval for tide computation. Default is 60 minutes.",
                        required = False)
    # plots
    parser.add_argument( '--plot','-plot',
                        action = 'store_true',
                        dest = 'plot',
                        help = "Plot all stationary timeseries.",
                        required = False)
    # parser
    args = parser.parse_args()


    ### Input ###
    inp = args.input[0]

    ### Output ###
    out = args.output[0]+".csv"
    figname = args.output[0]+".png"

    ### Output frequency ###
    freq = int(args.ofreq[0])

    ### Time interval ###
    dt = int(args.dt[0])

    ### Plot ###
    if args.plot:
        plot = True
    else:
        plot = False

    ### Read input netcdf file ###

    ds = xr.open_dataset(inp)

    ### Translate to a DataFrame and resample to 1Hz ###
    df = ds.to_dataframe().resample(rule='1S',closed="right").bfill()

    ### Compute tidal timestamps and water levels ###
    time, tide = tidal_timestamps(df)

    ### Interploate tidal curve ###
    times = pd.date_range(start=time[0], end=time[-1], freq='{}T'.format(freq)).to_pydatetime()
    x = date2num(time); y= tide
    xnew = date2num(times)
    f = interpolate.interp1d(x,y,kind='cubic')    
    tides = f(xnew)
    tendencies = tidal_tendency(tides)

    ### Output table ###
    dfo = pd.DataFrame(np.vstack([tides,tendencies]).T,index=times,columns=["tide","tendency"])
    dfo.to_csv(out+".csv")

    ### PLots ###
    if plot:
        # cut dataframe to date range
        dfp = df[time[0]:time[-1]].copy().reset_index()   
        # plot
        fig,ax = plt.subplots(figsize=(12,6))
        ax.plot(dfp["time"].values,dfp["pressure"].values,'o',color="k",alpha=0.5,label="Pressure")
        ax.plot(time,tide,"or",zorder=11,markersize=12,label="Tidal points")
        ax.plot(times,tides,'-',color='cornflowerblue',zorder=10,lw=4,label="Tidal Curve")
        ax.legend(loc="best",ncol=3)
        sns.despine(ax=ax)
        ax.grid(ls="--")
        ax.set_xlim(time[0]-datetime.timedelta(seconds=120),time[-1]+datetime.timedelta(seconds=120))
        # ax.set_ylim(tide.min()*0.75,tide.max()*1.25)
        ax.set_ylabel(r"Water depth $[m]$")
        # ax.set_title()
        fig.tight_layout()
        plt.savefig(figname,dpi=300)
        plt.show()




    print ("\nScript {} has just finished. My work is done!\n".format(sys.argv[0]))
