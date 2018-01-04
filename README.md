# PyWaveLearn
Machine learning for wave scientists.

This repository gathers useful tools for coastal scientists to handle common near-shore data with special focus on data mining,  machine learning, and big data.

For example, hard tasks such as detecting wave breaking becomes as easy as:

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

```python
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

Usage examples are available [here](doc/stats_spectral_and_linear.md).




#
TODO: Add docs

# pwl. sensors
TODO: Add docs

# pwl.utils
The module “utils” gathers commonly used  functions shared among the other modules.

# scripts
TODO: Add docs
