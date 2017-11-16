# PyWaveLearning
Machine Learning in Wave Science

This repository bring useful tools for coastal scientists to handle common nearshore data types with special focus on machine learning.

For example, to detect wave breaking:

[include gif]

# pwl.image
The module "image" brings functions to manipulate video data, rectify images based on GCPs, calibrate virtually any video camera, and extract timestacks.

# pwl.colour
The module "colour" was develop to translate coastal images (and timestacks) to the CIECAM2 colourspace and perform machine learning tasks. 

# pwl.linear
The module “linear” brings implementations of some linear wave theory equations:

```python
from numpy import pi
import pywavelearning.linear as pwll

# define water depth
h = 2.0
# define wave period
T = 10
# wave angular frequency
omega = 2*pi/T

# calculates the wave number
k = pwll.wave_number(omega)

# calculates the wave angular frequency
omega  = pwll.frequency(k)

# calculates the wave celerity
c = pwll.celerity(k,h)

# calculates the group speed
cg= pwll.group_speed(k,h)

# calculates the dispersion equation
q = pwll.dispersion(h)
```


# pwl.utils
The module “utils” gathers commonly used  functions shared among the other modules.

# pwl.spectral and pwl.stats


# scripts
The scripts/ folder brings some cool scripts to make
