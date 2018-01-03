# FIXME: Docstring


def power_spectrum_density(data, fs, window="hann", plot=False):
    """ Calculate the Power Spectrum Density (PSD)
        using Welch's method.

    ----------
    Args:
        data   [Mandatory (np.ndarray)]: Input Data array

        fs [Mandatory (int)]: Sampling frequency in units of Hz.

        window [Optinal (str)]: Window type to be used. Tested options are
                                  "blackman", "hamming", "hann", "bartlett",
                                  "parzen", "bohman", "blackmanharris",
                                  "nuttall", "barthann".

        nperseg [Mandatory (int)]: Length of each segment. Default is 256.

        nfft   [Mandatory (int)]: Length of the FFT used,
                                  if a zero padded FFT is desired.

    ----------
    Returns:
        f [(np.ndarray)]: Array of sample frequencies.

        psd [(np.ndarray)] Power spectral density (PSD)

    """
    from scipy.signal import welch, get_window

    # window
    Nx = len(data)
    w = get_window(window, Nx, fftbins=True)

    # nperseg and nfft parameters
    nfft = nperseg = Nx

    # calculate the PSD using welch method
    f, psd = welch(data, fs, window=w, nfft=nfft, nperseg=nperseg)

    return f, psd

# The following spectral filters were downloaded form stackoverflow
#  and not tested. I am not sure if they work properly.


def butter_bandpass(lowcut, highcut, fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    from scipy.signal import lfilter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    y[0] = data[0]
    return y


def butter_lowpass(cutoff, fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    from scipy.signal import lfilter
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    y[0] = data[0]
    return y


def butter_highpass(cutoff, fs, order=5):
    from scipy.signal import butter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    from scipy.signal import lfilter
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    y[0] = data[0]
    return y
