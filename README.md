# PyWaveLearning
Machine Learning in Wave Science

This repository bring useful tools for coastal scientists to handle common nearshore data types with special focus on machine learning. For example, to detect wave breaking:

![breaking](doc/image/predict_wavebreaking.gif)

# pwl.image
The module "image" brings functions to manipulate video data, rectify images based on GCPs, calibrate virtually any video camera and extract timestacks.

```python
# Rectify a single frame

import cv2
import skimage.io
import pandas as pd
import pywavelearning.linear as ipwl

# read the camera intrinsic parameters
K,DC = ipwl.camera_parser("data/Calibration/CameraCalib.txt")

# read frame
I = skimage.io.imread("/data/Image/OMB.jpg")
h,  w = I.shape[:2]

# read GCPs coordinates
XYZ = pd.read_csv("/data/Image/xyz.csv")[["x","y","z"]].values

# read UV coords 
UV = pd.read_csv("/data/Image/uv.csv")[["u","v"]].values

# undistort frame
Kn,roi = cv2.getOptimalNewCameraMatrix(K,DC,(w,h),1,(w,h))
I = cv2.undistort(I, K, DC, None, Kn)

# find homography
H = ipwl.find_homography(UV, XYZ, K, z=0, distortion=0)

# rectify coordinates
X,Y = ipwl.rectify_image(I, H)
```

# pwl.colour
The module "colour" was develop to translate coastal images (and timestacks) to the CIECAM2 colourspace and perform machine learning tasks. 

# pwl.linear
The module “linear” brings implementations of some linear wave theory equations:

```python
from numpy import pi
import pywavelearning.linear as lpwl

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

# pwl.utils
The module “utils” gathers commonly used  functions shared among the other modules.

# pwl.spectral and pwl.stats

# scripts
The scripts/ folder brings some cool scripts to make
