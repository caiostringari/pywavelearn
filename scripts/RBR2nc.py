# bla# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : read_rbr.py
# POURPOSE : Read RBR PT data exported from ruskin. Only ODV text files for now
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 26/06/2018 [Caio Stringari]
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


def main():

    print("\nConverting file, please wait...\n")

    # arguments
    inp = args.input[0]
    met = args.metadata[0]
    out = args.output[0]

    # read the file into a pandas dataframe
    df = pd.read_csv(inp)

    # fix times
    dates = [datetime.datetime.strptime(x, FMT) for x in df["Time"].values]
    df.index = dates
    df.index.name = "time"

    # fix the dataframe
    df = df.drop("Time", axis=1)

    # fix the columns names
    df.columns = [x.lower() for x in df.columns]

    # to xarray
    ds = df.to_xarray()

    # read metadata
    if met:
        with open(met) as f:
            meta = json.load(f)

        # add metadata to netcdf
        ds.attrs = meta

    print(ds)

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

    # input metadata
    parser.add_argument('--metadata', '-m',
                        nargs=1,
                        action='store',
                        default=[False],
                        dest='metadata',
                        required=False,
                        help="Input metadata file exported from Ruskin.")

    # output netcdf
    parser.add_argument('--output', '-o',
                        nargs=1,
                        action='store',
                        dest='output',
                        required=True,
                        help="Output netCDF file.")

    args = parser.parse_args()

    # data format
    FMT = "%Y-%m-%d %H:%M:%S.%f"

    # main call
    main()
