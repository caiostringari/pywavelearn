def HM0(data,fs):
    """ Same as Hm0() but with data parameter.

    Hm0 = 4*srt(E)

    E = Sum s(f)xd(f). Calculated using the trapezoidal rule.

    s(f) is the spectral density energy
    d(f) is the bandwidth

    ----------
    Args:
        data (Mandatory [np.array,tuple or list]): Array of measured values

        fs (Mandatory (int): sample frequency

    ----------
    Return:
        hm0 ([float]) Hm0  height [m]
    """

    # power spectrum density
    f, psd = power_spectrum_density(data,fs)

    # Total Energy
    E = np.trapz(psd,f)

    # Hm0
    hm0 = 4*np.sqrt(E)

    return hm0

def Hbar(data,fs=1,method="spectral"):
    """ Compute Hbar.

    If method = spectral, uses Holthijsen formulation:

        Hbar = sqrt(pi/8)*Hm0

    if method == komar, uses Komar's formulation:

        Hbar = sqrt(2pi)*std(data)

    ----------
    Args:
        data (Mandatory [np.array,tuple or list]): Surface elevation values [m]

        fs (Optional [float]): Sampling frequency [Hz] in the spectral method.

        method (Optional [str]): Which method to use. Holthijsen is the default.

    ----------
    Return:
        Hbar ([float]) Hbar  height [m]
    """

    if method.lower() == "spectral":
        f,psd = power_spectrum_density(data,fs)
        hm0 = Hm0(f,psd)
        Hbar = np.sqrt(np.pi/8)*hm0
    elif method.lower() == "komar":
        Hbar = np.sqrt(2*np.pi)*data.std()

    return Hbar

def Hrms(data,fs=1,method="stats"):
    """ Compute the Root Mean Square wave heights (Hrms).

    ----------
    Args:
        heights (Mandatory [np.ndarray]): Array of wave heights [m].

        method (Optional [str]): Method to use. Available methos are:

                "stats":  Calculate Hrms as

                "spectral":
    ----------
    Return:
        Hrms ([float]) Root Mean Square wave height [m]
    """

    if method == "stats":
        N = len(data)
        Hrms = np.sqrt((1./N)*np.sum(data*data))
    elif method == "spectral":
        f,psd = power_spectrum_density(data,fs)
        hm0 = Hm0(f,psd)
        Hrms =  0.5*np.sqrt(2)*hm0
    elif method == "komar":
         Hrms = np.sqrt(8)*data.std()

    return Hrms

def significant_wave_height(heights,method="stats"):
    """ Compute the Significant Wave Height (Hs or Hsig).

    ----------
    Args:
        heights (Mandatory [np.ndarray]): Array of wave heights [m].

        method (Optional [str]): Method to use. Available methos are:

                "stats":  Calculate Hs as the averaged value of
                          the highest one-third of recorded waves.
                          This is the default method

                "spectral":
    ----------
    Return:
        hs ([float]) Significant wave height [m]
    """

    if method =='stats':
        # Number of waves
        nwaves = len(heights)
        # Descending sorted wave heights
        sheights = np.sort(np.array(heights))[::-1]
        #Significant wave height
        Hs = ((1.)/(nwaves/3.))*(np.sum(sheights[0:int(len(sheights)/3)]))
    #TODO: Implement spectral version
    else:
        raise NotImplementedError("Method {} is not implemented yet".format(method))

    return Hs

def Tm01(f,psd,cut=False,tmin=1,tmax=20):
#FIXME: Fix Docstrings
    """ Extract the wave peak period.

    ----------
    Args:
        f (Mandatory [np.nparray]): Frequencies [Hz]

        psd (Mandatory [np.ndarray]): Power Spectrum Density (PSD) [m**2Hz]

        method (Optional [str]): Method to be used. Available methos are:

                "spectral":  Extract the peak period as the inverse of
                             the most energetic frequency. This is the default method

                "wavelet": Use the wavelet transform to get a better representation
                           in the time domain. Default is to use the Morlet wavelet
    ----------
    Return:
        tp (Mandatory [float]) peak period [s]
    """
    # T0M1=sum(df.*Efb(14:120)./freq(14:120))/sum(df.*Efb(14:120))
    # cut to sea-swell frequencies
    if cut:
        i2 = np.abs((1/f)-tmin).argmin()
        i1 = np.abs((1/f)-tmax).argmin()
        f = f[i1:i2]
        psd = psd[i1:i2]

    # print (len(f),len(psd))

    m0 = np.trapz(np.abs(psd),f)
    m1 = np.trapz(psd*f,f)

    Tm = m0/m1

    return Tm


def Tm02(f,psd,cut=False,tmin=1,tmax=20):
    #FIXME: Fix Docstrings
    """ Extract the wave peak period.

    ----------
    Args:
        f (Mandatory [np.nparray]): Frequencies [Hz]

        psd (Mandatory [np.ndarray]): Power Spectrum Density (PSD) [m**2Hz]

        method (Optional [str]): Method to be used. Available methos are:

                "spectral":  Extract the peak period as the inverse of
                             the most energetic frequency. This is the default method

                "wavelet": Use the wavelet transform to get a better representation
                           in the time domain. Default is to use the Morlet wavelet
    ----------
    Return:
        tp (Mandatory [float]) peak period [s]
    """
    # T0M1=sum(df.*Efb(14:120)./freq(14:120))/sum(df.*Efb(14:120))
    # cut to sea-swell frequencies
    if cut:
        i2 = np.abs((1/f)-tmin).argmin()
        i1 = np.abs((1/f)-tmax).argmin()
        f = f[i1:i2]
        psd = psd[i1:i2]

    # print (len(f),len(psd))

    m0 = np.trapz(np.abs(psd),f)
    m2 = np.trapz(psd*(f*f),f)

    Tm = np.sqrt(m0/m2)

    return Tm


def TM01(data,fs,cut=False,tmin=1,tmax=20):
    """    Same as Tm01() but with data parameter """
    #FIXME: Docstrings
    # power spectrum density
    f, psd = power_spectrum_density(data,fs)

    # print (f)

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
    """    Same as Tm02() but with data parameter """
    #FIXME: Docstrings
    # power spectrum density
    f, psd = power_spectrum_density(data,fs)

    # print (f)

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
    """ Calculate the Significant Wave Period (Ts or Tsig) as the averaged value of
    the highest one-third of recorded wave periods. See Holthijsen for details.

    ----------
    Args:
        periods (Mandatory [np.ndarray]): Array of wave periods [s]

    ----------
    Return:
        Ts ([float]) Significant wave period [s]
    """
    # Number of waves
    nwaves = len(periods)
    # Descending sorted wave periods
    speriods = np.sort(np.array(periods))[::-1]
    # Significant wave period
    Ts = ((1.)/(nwaves/3.))*(np.sum(speriods[0:int(len(speriods)/3)]))

    return Ts


def get_waveheihts(data,fs):
    """ Calculate all wave heights implemented under "wavetools" """

    f,psd = power_spectrum_density(data,fs) # PSD

    hm0 = Hm0(f,psd)

    hbar1 = Hbar(data,fs,method="spectral")
    hbar2 = Hbar(data,method="komar")

    hrms1 = Hrms(data,fs,method="spectral")
    hrms2 = Hrms(data,method="komar")

    return hm0,hbar1,hbar2,hrms1,hrms2