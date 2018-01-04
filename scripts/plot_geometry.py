# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : plot_geometry.py
# POURPOSE : Plot a geometry netCDF4 file created using build_geometry.py
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 04/01/2018 [Caio Stringari]
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
from __future__ import print_function, division
# Arguments
import argparse

# Numpy
import numpy as np

# Data
import xarray as xr

import seaborn as sns
import matplotlib.pyplot as plt
from pywavelearn.image import construct_rgba_vector


def main():
    ds = xr.open_dataset(args.input[0])

    bbox = [float(args.lims[0]), float(args.lims[1]),
            float(args.lims[2]), float(args.lims[3])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # plot in pixel coords
    ax1.imshow(ds["rgb"].values)
    # plot in metric coords
    im = ax2.pcolormesh(ds["X"].values,
                        ds["Y"].values,
                        ds["rgb"].values.mean(axis=2))
    # set to true colour
    rgba = construct_rgba_vector(ds["rgb"].values, n_alpha=0)
    rgba[rgba > 1] = 1
    im.set_array(None)
    im.set_edgecolor('none')
    im.set_facecolor(rgba)
    ax2.set_aspect("equal")
    # set axis
    ax2.set_xlim(bbox[0], bbox[1])
    ax2.set_ylim(bbox[2], bbox[3])
    # show
    plt.show()


if __name__ == '__main__':

    print("\nPlotting geometries, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input data
    parser.add_argument('--input', '-i',
                        nargs=1,
                        action='store',
                        dest='input',
                        help="Input netCDF genereated by build_geometry.py.",
                        required=True)
    # input data
    parser.add_argument('--axis_limits', '-xylim',
                        nargs=4,
                        action='store',
                        dest='lims',
                        help="Metric axes limits. Xmin Xmax Ymin Ymax.",
                        required=True)
    # parser
    args = parser.parse_args()

    main()

    print("\nMy work is done!\n")
