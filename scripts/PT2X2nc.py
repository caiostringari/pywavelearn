# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : PT2X2nc.py
# POURPOSE : Read PT2X data exported from Aqua4Plues and export a netCDF file
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 29/06/2018 [Caio Stringari]
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

import sys

import argparse

import datetime

import numpy as np

import json

import pandas as pd
import xarray as xr

from pywavelearn.sensors import PT2X_parser


def main():

    print("\nConverting file, please wait...\n")

    # arguments
    inp = args.input[0]
    out = args.output[0]

    # read
    df, met = PT2X_parser(inp)

    # convert
    ds = df.to_xarray()

    # add metadata
    if met:
        # add metadata to netcdf
        ds.attrs = met

    # print results
    print(ds)

    # dump to file
    ds.to_netcdf(out)
    print("\nMy work is done!")


if __name__ == '__main__':

    # argument parser
    parser = argparse.ArgumentParser()

    # input data
    parser.add_argument('--input', '-i',
                        nargs=1,
                        action='store',
                        dest='input',
                        required=True,
                        help="Input text file exported from Ruskin.")

    # output netcdf
    parser.add_argument('--output', '-o',
                        nargs=1,
                        action='store',
                        dest='output',
                        required=True,
                        help="Output netCDF file.")

    args = parser.parse_args()

    # data format

    # main call
    main()
