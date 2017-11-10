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
    Return:
         I (Mandatory [np.array]): surrounding pixels in the i-direction.

         J (Mandatory [np.array]): surrounding pixels in the j-direction.
    """

    # compute domain
    i = np.arange(i - iwin, i + iwin + 1, 1)
    j = np.arange(j - jwin, j + jwin + 1, 1)

    # all pixels inside the domain
    I, J = np.meshgrid(i, j)

    # i-dimension
    I = I.flatten()

    # j-dimension
    J = J.flatten()

    #FIXME: add boundary conditions to avoid errors.
  
    return I, J