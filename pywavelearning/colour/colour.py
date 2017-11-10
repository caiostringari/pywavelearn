
def get_dominant_colour(df, n_colours=8, force_label_to_white=False):
    """
    Get the dominant colour using K-means clustering.

    ----------
    Args:
        df (Mandatory [pd.DataFrame]): Dataframe with training colour
        information.

        n_colours (Optional [int]): Number of colour clusters. Default is 8

        force_foam_to_white (Optional [bool]). Force foam to be pure white.
        Default is true.

    ----------
    Return:
         Labels (Mandatory [list]): List of label identifiers

         Regions (Mandatory [list]): Labels corresponding regions

         DominantColours (Mandatory [list of lists]): RGB values for each dominant
         colour.
    """
    Dom_colours = []
    Labels = []
    Regions = []
    for label, group in df.groupby("label"):
        # get RGB values
        rgb = group[["r", "g", "b"]].values / 255
        X = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb))
        # build the model
        kmeans = KMeans(n_clusters=n_colours, random_state=0).fit(X)
        # predict labels
        labels = kmeans.predict(X)
        # get dominant label
        ulabels, ucounts = np.unique(labels, return_counts=True)
        # dominant label
        dom_label = ulabels[np.argmax(ucounts)]
        # get dominant colour
        dom_colour = colour.XYZ_to_sRGB(colour.Lab_to_XYZ(kmeans.cluster_centers_[dom_label, :]))
        # set foam to white
        if force_label_to_white:
            if group["region"].values[0] == force_label_to_white:
                dom_colour = [1, 1, 1]
        # append to output
        Dom_colours.append(dom_colour)
        Labels.append(group["label"].values[0])
        Regions.append(group["region"].values[0])

    return Labels, Regions, np.array(Dom_colours) * 255


def classify_colour(colour, colour_targets, target_labels):
    """
    Classify a given colour to one of the colour targets using colour similarity
    metrics in the CIECAM02 colour space.

    ----------
    Args:
        colour (Mandatory [list]): List with input RGB values

        colour_targets (Mandatory [list]): List with trained RGB values

        target_labels  (Mandatory [list]): List with  RGB labels

    ----------
    Return:
         Labels (Mandatory [list]): List of classified labels.
    """
    dists = deltaE(colour, colour_targets)

    idx = np.argsort(dists)

    sorted_labels = np.array(target_labels)[idx]

    # probs = (100 - dists[idx]) / 100

    # classified_label = target_labels[np.argmin(np.abs(dists))]

    return sorted_labels