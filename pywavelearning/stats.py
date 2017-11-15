""" Useful functions to calculate some wave statistics """

import numpy as np
from . import spectral

def Hm0(f,psd):
    """
    Same as Tm01() but with data and frequency as input paramters

    ----------
    Args:
        data [Mandatory (np.array,tuple or list)]: Array of measured values

        fs [Mandatory (int)]: sample frequency

        cut [Optional (bool)]: Cut to a sub-set of the spectrum

        tmin [Optional (bool)]: lower cutoff period if cut is True (in Seconds)

        tmax [Optional (bool)]: upper cutoff period if cut is True (in Seconds)

    ----------
    Returns:
        Tm [Mandatory (float)] spectral peak period [s]
    """
    # total Energy
    E = np.trapz(psd,f)

    # Hm0
    hm0 = 4*np.sqrt(E)

    return hm0

def HM0(data,fs):
    """ Same as Hm0() but passing data as input parameter.

    Hm0 = 4*srt(E)

    E = Sum s(f)xd(f). Calculated using the trapezoidal rule.

    s(f) is the spectral density energy
    d(f) is the bandwidth

    ----------
    Args:
        data [Mandatory (np.array,tuple or list)]: Array of measured values

        fs [Mandatory (int)]: sample frequency

    ----------
    Returns:
        hm0 [Mandatory (float)] Hm0 wave height [m]
    """

    # power spectrum density
    f, psd = spectral.power_spectrum_density(data,fs)

    # Total Energy
    E = np.trapz(psd,f)

    # Hm0
    hm0 = 4*np.sqrt(E)

    return hm0

def Hbar(data,fs=1,method="komar"):
    """ Compute averaged wave height Hbar.

    If method = spectral, uses Holthijsen formulation:

        Hbar = sqrt(pi/8)*Hm0

    if method == komar, uses Komar's formulation:

        Hbar = sqrt(2pi)*std(data)

        This is the default method

    ----------
    Args:
        data [Mandatory (np.array,tuple or list)]: Surface elevation values [m]

        fs [Optional (float)]: Sampling frequency [Hz] if "spectral" method is used.

        method [Optional (str)]: Which method to use.

    ----------
    Returns:
        Hbar [Mandatory (float)] Hbar wave height [m]
    """

    if method.lower() == "spectral":
        f,psd = spectral.power_spectrum_density(data,fs)
        hm0 = Hm0(f,psd)
        Hbar = np.sqrt(np.pi/8)*hm0
    elif method.lower() == "komar":
        Hbar = np.sqrt(2*np.pi)*data.std()

    return Hbar

def Hrms(data,fs=1,method="komar"):
    """ Compute the Root Mean Square wave height (Hrms).

    ----------
    Args:
        heights [Mandatory (np.ndarray)]: Array of measued heights [m].

        method [Optional (str)]: Method to use. Available methos are:

                "stats":  Calculate Hrms using the statistical formulation

                "spectral": Calculate Hrms using the PSD.

                "komar": Calculate Hrms using the Komar's variance formula.
                         This is the default.

        fs [Optional (float)]: Sampling frequency [Hz] if "spectral" method is used.
    ----------
    Returns:
        Hrms [Mandatory (np.ndarray)] Root Mean Square wave height [m]
    """

    if method == "stats":
        N = len(data)
        Hrms = np.sqrt((1./N)*np.sum(data*data))
    elif method == "spectral":
        f,psd = spectral.power_spectrum_density(data,fs)
        hm0 = Hm0(f,psd)
        Hrms =  0.5*np.sqrt(2)*hm0
    elif method == "komar":
         Hrms = np.sqrt(8)*data.std()

    return Hrms

def significant_wave_height(heights):
    """ Compute the Significant Wave Height (Hs or Hsig).

    ----------
    Args:
        heights [Mandatory (np.ndarray)]: Array of wave heights [m].

    ----------
    Return:
        hs [Mandatory (float)] Significant wave height [m]
    """

    # number of waves
    nwaves = len(heights)
    # descending sorted wave heights
    sheights = np.sort(np.array(heights))[::-1]
    # significant wave height
    Hs = ((1.)/(nwaves/3.))*(np.sum(sheights[0:int(len(sheights)/3)]))

    return Hs

def Tm01(f,psd,cut=False,tmin=1,tmax=20):
    """
    Calculate the first spectral wave period (Tm01). Will use only a portion
    of the spetra if "cut" is set to True.

    T0M1 = sum(df.*Efb(14:120)./freq(14:120))/sum(df.*Efb(14:120))

    ----------
    Args:
        f [Mandatory (np.nparray)]: Frequencies [Hz]

        psd [Mandatory (np.ndarray)]: Power Spectrum Density (PSD) [m**2Hz]

        cut [Optional (bool)]: Cut to a sub-set of the spectrum

        tmin [Optional (bool)]: lower cutoff period if cut is True (in Seconds)

        tmax [Optional (bool)]: upper cutoff period if cut is True (in Seconds)
    ----------
    Returns:
        Tm [Mandatory (float)] spectral peak period [s]
    """
        
    # cut to the desired frequency range
    if cut:
        i2 = np.abs((1/f)-tmin).argmin()
        i1 = np.abs((1/f)-tmax).argmin()
        f = f[i1:i2]
        psd = psd[i1:i2]

    # calculate
    m0 = np.trapz(np.abs(psd),f)
    m1 = np.trapz(psd*f,f)
    Tm = m0/m1

    return Tm

def Tm02(f,psd,cut=False,tmin=1,tmax=20):
    """
    Calculate the second spectral wave period (Tm01). Will use only a portion
    of the spetra if "cut" is set to True.

    T0M1 = sum(df.*Efb(14:120)./freq(14:120))/sum(df.*Efb(14:120))

    ----------
    Args:
        f [Mandatory (np.nparray)]: Frequencies [Hz]

        psd [Mandatory (np.ndarray)]: Power Spectrum Density (PSD) [m**2Hz]

        cut [Optional (bool)]: Cut to a sub-set of the spectrum

        tmin [Optional (bool)]: lower cutoff period if cut is True (in Seconds)

        tmax [Optional (bool)]: upper cutoff period if cut is True (in Seconds)
    ----------
    Returns:
        Tm [Mandatory (float)] spectral peak period [s]
    """
    if cut:
        i2 = np.abs((1/f)-tmin).argmin()
        i1 = np.abs((1/f)-tmax).argmin()
        f = f[i1:i2]
        psd = psd[i1:i2]

    # calculate
    m0 = np.trapz(np.abs(psd),f)
    m2 = np.trapz(psd*(f*f),f)
    Tm = np.sqrt(m0/m2)

    return Tm

def TM01(data,fs,cut=False,tmin=1,tmax=20):
    """
    Same as Tm01() but with data and frequency as input paramters

    ----------
    Args:
        data [Mandatory (np.array,tuple or list)]: Array of measured values

        fs [Mandatory (int)]: sample frequency

        cut [Optional (bool)]: Cut to a sub-set of the spectrum

        tmin [Optional (bool)]: lower cutoff period if cut is True (in Seconds)

        tmax [Optional (bool)]: upper cutoff period if cut is True (in Seconds)

    ----------
    Returns:
        Tm [Mandatory (float)] spectral peak period [s]
    """
    
    # power spectrum density
    f, psd = spectral.power_spectrum_density(data,fs)

    if cut:
        i2 = np.abs((1/f)-tmin).argmin()
        i1 = np.abs((1/f)-tmax).argmin()
        f = f[i1:i2]
        psd = psd[i1:i2]

    m0 = np.trapz(np.abs(psd),f)
    m1 = np.trapz(psd*f,f)

    Tm = m0/m1

    return Tm

def TM02(data,fs,cut=False,tmin=1,tmax=20):
    """
    Same as Tm02() but with data and frequency as input paramters

    ----------
    Args:
        data [Mandatory (np.array,tuple or list)]: Array of measured values

        fs [Mandatory (int)]: sample frequency

        cut [Optional (bool)]: Cut to a sub-set of the spectrum

        tmin [Optional (bool)]: lower cutoff period if cut is True (in Seconds)

        tmax [Optional (bool)]: upper cutoff period if cut is True (in Seconds)

    ----------
    Returns:
        Tm [Mandatory (float)] spectral peak period [s]
    """
    # power spectrum density
    f, psd = spectral.power_spectrum_density(data,fs)

    if cut:
        i2 = np.abs((1/f)-tmin).argmin()
        i1 = np.abs((1/f)-tmax).argmin()
        f = f[i1:i2]
        psd = psd[i1:i2]

    m0 = np.trapz(np.abs(psd),f)
    m2 = np.trapz(psd*(f*f),f)

    Tm = np.sqrt(m0/m2)

    return Tm

def significant_wave_period(periods):
    """
    Calculate the Significant Wave Period (Ts or Tsig) as the averaged value of
    the highest one-third of recorded wave periods. See Holthijsen for details.

    ----------
    Args:
        periods [Mandatory (np.ndarray)]: Array of wave periods [s]

    ----------
    Return:
        Ts [Mandatory (float)] Significant wave period [s]
    """

    # number of waves
    nwaves = len(periods)
    # descending sorted wave periods
    speriods = np.sort(np.array(periods))[::-1]
    # significant wave period
    Ts = ((1.)/(nwaves/3.))*(np.sum(speriods[0:int(len(speriods)/3)]))

    return Ts