def pixel_window(a, i, j, iwin=8, jwin=8):
    """
    Get the surrouding pixels from a given pixel center.

    ----------
    Args:
        a (Mandatory [np.array]): input image array. Only usused to figure out shapes.

        i (Mandatory [int]): pixel center in the i-direction.

        j (Mandatory [int]): pixel center in the j-direction.

        iwin (Optional [int]): window size in the i-direction. Default is 8.

        iwin (Optional [int]): window size in the j-direction. Default is 8.

    ----------
    Returns:
         I (Mandatory [np.array]): surrounding pixels in the i-direction.

         J (Mandatory [np.array]): surrounding pixels in the j-direction.
    """
    import numpy as np

    # compute domain
    i = np.arange(i-iwin,i+iwin+1,1)
    j = np.arange(j-jwin,j+jwin+1,1)

    # all pixels inside the domain
    I,J = np.meshgrid(i,j)

    # Remove pixels outside the borders
    
    # i-dimension
    I = I.flatten()
    I[I<0] = -999
    I[I>a.shape[0]] = -999
    idx = np.where(I==-999)
    
    # j-dimension
    J = J.flatten()
    J[J<0] = -999
    J[J>a.shape[1]] = -999
    jdx = np.where(J==-999)

    Ifinal = np.delete(I,np.hstack([idx,jdx]))
    Jfinal = np.delete(J,np.hstack([idx,jdx]))

    return Ifinal,Jfinal

def equalize(I,method="global"):
    """
    Equalize a low-contrast image.
    
    ----------
    Args:
        I [Mandatory (np.ndarray)]: image array. values must range from 0 to 1.
                                    ex: I = skimage.color.rgb2grey(img)

        method [Optional (str)]: Method to be used. Only global equalization is
                                 implemented by now.
    
    ----------
    Returns:
        IE [Mandatory (np.ndarray)]: equalized image.
    """
    from skimage import exposure
    from skimage.util import img_as_ubyte
    from skimage.morphology import disk, rectangle

    # Scale image to 0-255
    Iubyte = img_as_ubyte(I)

    if method == "global":
        IE = exposure.equalize_hist(I)
    else:
        raise NotImplementedError("Sorry, other methods are not implemented yet.")

    return IE