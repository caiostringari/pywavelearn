# PyWaveLearn
Machine learning for wave scientists.

This repository gathers useful tools for coastal scientists to handle common nearshore data with special focus on data mining,  machine learning, and big data.

For example, hard tasks sush as detecting wave breaking becomes as easy as:

```bash
python learn_wavebreaking.py
```

![breaking](doc/image/predict_wavebreaking.gif)


# Installation:

PyWaveLearn can only be installed from the source code.

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
images using the OpenCV and scikit-image packages. Most of the heavy lifiting is
done using [Flamingo](http://flamingo-image.readthedocs.io/). This module has
the companion script [extract_timestack.py](scripts/extract_timestack.py) which
extracts space-time transects (timestacks) from a set of coastal images and also
has the option to store rectied frames in a netCDF4 structure suitable for big
data analysis.

Usage examples are available [here](doc/pwl_image.md)

# pwl.colour
The module "colour" was develop to translate coastal images (snapshots, timex, variance and timestacks) into the CIECAM2 colourspace and perform some machine learning tasks. For example, calculate classify colours in a snapshot based on well defined target colours:

```python
import numpy as np
import skimage.io
import PyWaveLearn.colour as cpwl

# read frame
I = skimage.io.imread("/data/Image/OMB.jpg")

# get color bands
snap_colours = np.vstack([I[:,:,0].flatten(),I[:,:,1].flatten(),I[:,:,2].flatten()]).T
user_colours = np.vstack([0,0,0],[255,255,255]).T
colour_labels = [0,1] # zero is black, one is white

# learning step
labels = []
for rgb in snap colours:
  labels.append(cpwl.classify_colour(rgb, user_colours, colour_labels))

# return to original shape
L = np.array(labels).reshape(I.shape[0],I.shape[1])
```
These functions are the basis for extracting more complicated features from coastal images, such as wave breaking and shoreline evolution.

# pwl.linear
The module “linear” brings implementations of some linear wave theory equations:

```python
from numpy import pi
import PyWaveLearn.linear as lpwl

# define a water depth
h = 2.0
# define a non-dimensional water depth
p = 1.2
# define a wave period
T = 10
# wave an angular frequency
omega = 2*pi/T

# wave number
k = lpwl.wave_number(omega)

# wave angular frequency
omega  = lpwl.frequency(k)

# wave celerity at any depth
c = lpwl.celerity(k,h)

# wave group speed at any depth
cg= lpwl.group_speed(k,h)

# dispersion relation for a non-dimensional water depth
q = lpwl.dispersion(p)
```

# pwl.spectral and pwl.stats
TODO: Add docs

# pwl. sensors
TODO: Add docs

# pwl.utils
The module “utils” gathers commonly used  functions shared among the other modules.

# scripts
TODO: Add docs
