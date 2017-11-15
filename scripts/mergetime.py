#------------------------------------------------------------------------
#------------------------------------------------------------------------
#
#
# SCRIPT   : mergetime.py
# POURPOSE : merge all netcdfs in a folder
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# V1.0     : 13/09/2016 [Caio Stringari]
#
# OBSERVATIONS:
#
# Usage: python mergetime *.nc merged.nc
#
#
#------------------------------------------------------------------------
#------------------------------------------------------------------------

# Dates
import sys

# Data I/O
import xarray as xr

def main():
    # Argument parser
    args = sys.argv

    # input files
    files = sorted(args[1:-1])

    # output file
    output = args[-1]

    # load data
    ds = xr.open_mfdataset(files,concat_dim="time")

    # write data
    ds.to_netcdf(output)


if __name__ == '__main__':

    main()

   
