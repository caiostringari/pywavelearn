# PyWaveLearn
Machine learning for wave scientists.

This repository make available some useful tools for coastal scientists to
handle common near-shore data with special focus on data mining,
machine learning, and big data.

For example, hard tasks such as detecting wave breaking becomes as easy as in
[this example](notebooks/learn_wavebreaking.ipynb).

```bash
python learn_wavebreaking.py
```


![breaking](doc/image/predict_wavebreaking.gif)


# Installation:

pywavelearn can only be installed from the source code.

First, make sure you have all the dependencies:

Using conda:
```bash
# create a new environment
conda create --name pwl python=3.6
# activate
source activate pwl
# install the netCDF4 and xarray
conda install netCDF4 xarray
# install seaborn for better plots
conda install seaborn
# geoprocessing
conda install -c ioos geopandas
# colour analysis
pip install colorspacious
conda install -c conda-forge colour-science
# OpenCV
conda install -c conda-forge opencv
# science kits
conda install scikit-image scikit-learn
# heavy machine learning machinery
sudo apt-get install gcc gfortran
conda install cython tensorflow keras
# timeseries learning
pip install tslearn
# a nice progress bar
conda install -c conda-forge tqdm
# peak detection
pip install peakutils
```

You may also want ffmpeg and some codecs to process raw video data.
```bash
sudo apt-get install ffmpeg ubuntu-restricted-extras
```

Now, install pywavelearn:

```bash
git clone https://github.com/caiostringari/pywavelearn.git
cd pywavelearn
sudo python setup.py install
```

# pwl.image
The module **image** was designed to make it easier to rectify ARGUS-like
images using the OpenCV and scikit-image packages. Most of the heavy lifting is
done using [Flamingo](http://flamingo-image.readthedocs.io/). This module has
the companion script [extract_timestack.py](scripts/extract_timestack.py) which
extracts space-time transects (timestacks) from a set of coastal images and also
has the option to store rectified frames in a netCDF4 structure suitable for big
data analysis.

Usage examples are available [here](doc/pwl_image.md).

# pwl.colour
The module **colour** is the basis for most of the machine learn tasks available
in this package. It exploits the fact that unbroken waves, broken waves, and
the shoreline have different colour signatures that can be used to "learn"
information about these features.

The [wave breaking detection](scripts/learn_wavebreaking) script shows the full
potential of the colour module.

Usage examples are available [here](doc/pwl_colour.md).


# pwl.stats, pwl.spectral and pwl.linear

In the modules **stats**, **spectral** and **linear** you will find tools to
deal with the most common wave analysis problems. It provide ways to calculate
wave heights, periods, spectral densities and most of the parameters derived
from the linear wave theory.

Usage examples are available [here](doc/pwl_stats_spectral_and_linear.md).

# scripts

Most of the functions available across the various modules have a Command Line
Interface (CLI) companion. The most important ones are:

1. [calibrate_camera.py](scripts/calibrate_camera.py)
1. [extract_frames.py](scripts/extract_frames.py)
2. [get_gcp_uvcoords.py](scripts/get_gcp_uvcoords.py)
3. [extract_timestack.py](scripts/extract_timestack.py)
4. [learn_wavebreaking.py](scripts/learn_wavebreaking.py)

The full help for these scripts can be seen using
```python script_name.py --help```.

**TODOS:**

1. Fix all PEP8 issues
2. Add docs


<!-- # pwl.sensors
TODO:

1. Add docs
2. Work on RBR PT parser
3. Work on Sontek ADV parser -->
