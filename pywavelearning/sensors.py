""" This module deals with pressure transducer and ADV data I/O """

import pandas as pd
import xarray as xr

import datetime

def search_metadata(fname,mtstr):
    """
    Search for a string in a file and retrive the associated value.
    It is very important that the format is: "string, value", otherwise this
    function will not work.
    
    ----------
    Args:
        fname [Mandatory (str)]: filename or complete path

        mtstr [Mandatory (str)]"" string to search
    
    ----------
    Returns:
        mtd [Mandatory (str)]: the value found for the string requested
    """
    f = open(fname,'r')
    # Loop over the lines
    for l,line in enumerate(f.readlines()):
        # Search for "mtstr" in the current line
        if mtstr in line:
            # break the loop as soon as the string is found
            mtd = line.split(",")[1].strip('\n').strip()
            return mtd
    f.close()

def PT2X_parser(fname):
    """
    Parse pressure sensor data from PT2X csv files. Use Aqua4Plus to generate
    valid input files.
    
    ----------
    Args:
        fname [Mandatory (str)]:  input file name
    
    ----------
    Returns:
        df [Mandatory (pd.DataFrame)]: dataframe with pressure and temp values

        attrs [Mandatory (dict)]: dictionary with extracted metadata
    """

    # Metadata
    sensor_serial = search_metadata(fname,"SensorSN")
    if sensor_serial == None: sensor_serial="unknown"
    sensor_type = search_metadata(fname,"Sensor Type")
    if sensor_type == None: sensor_serial="unknown"
    sensor_name = search_metadata(fname,"Sensor Name")
    if sensor_name == None: sensor_serial="unknown"
    sensor_records = search_metadata(fname,"# Records")
    if sensor_records == None: sensor_serial="unknown"

    attrs = {"Serial Number":sensor_serial,
             "Sensor Type":sensor_type,
             "Sensor Name":sensor_name,
             "Number of Records":sensor_records}

    # Find the number of lines in the header
    f = open(fname,'r')
    for k,line in enumerate(f.readlines()):
        if "Rec #,Date/Time,Pressure(m H2O),Temperature(degC)" in line:
            header = k
        elif "Rec #,Date/Time,Temperature(degC),Pressure(m H2O)" in line:
            header = k

    # Read the data using pandas
    df = pd.read_csv(fname,sep=',',skiprows=header)

    # Figure out dates
    dtimes = []
    for date in df["Date/Time"]:
        fmt = "%d-%b-%y %H:%M:%S"
        # Find if there is microseconds
        for char in date:
            if char == '.':
                fmt = "%d-%b-%y %H:%M:%S.%f"
                break
        dtimes.append(datetime.datetime.strptime(date,fmt))

    # Update the dataframe index to be a time value
    df.index = dtimes

    # Drop the "Date/Time" Column
    df.drop("Date/Time",axis=1,inplace=True)

    # Remame columns
    df.rename(columns={"Temperature(degC)":"temperature"},inplace=True)
    df.rename(columns={"Pressure(m H2O)":"pressure"},inplace=True)
    df.rename(columns={"Rec #":"record"},inplace=True)

    # Re arange columns
    df = df[['record', 'pressure', 'temperature']]

    return df, attrs

def PT2X_to_netcdf(df,fname,attrs=False,returnds=False):
    """
    Convert a dataframe from  PT2X_parser() to netcdf format.
    CF conventions are followed when possible.

    ----------
    Args:
        df [Mandatory (df.DataFrame)]: input dataframe.

        attrs [Optional (dict)]: metadata dictionary

        returnds [Optional (True or False)]:  If True, returns the dataset

    ----------
    Returns:
        ds [Optional (xr.Dataset] xarray dataset
    """

    # metadata
    if attrs:
        ds = xr.Dataset(attrs=attrs)
    else:
        ds = xr.Dataset()

    # Coordinates
    ds.coords["time"] = (("time"),df.index.values)

    # Temperature
    ds["temperature"] = (("time"),df["temperature"].values+273.15)
    ds["temperature"].attrs["long_name"] = "sea_water_temperature"
    ds["temperature"].attrs["units"] = "K"

    # Pressure
    ds["pressure"] = (("time"),df["pressure"].values)
    ds["pressure"].attrs["long_name"] = "sea_water_pressure"
    ds["pressure"].attrs["units"] = "m H2O"

    # Record Number
    ds["record"] = (("time"),df["record"].values)

    ds.to_netcdf(fname)

    if returnds: return ds