
Rectify a single frame:

```python
import cv2
import skimage.io
import pandas as pd
import pywavelearn
from pywavelearn.utils import camera_parser
# import pywavelearning as ipwl

# read the camera intrinsic parameters
K,DC = ipwl.camera_parser("../data/Calibration/CameraCalib.txt")

# read frame
I = skimage.io.imread("../data/Image/OMB.jpg")
h,  w = I.shape[:2]

# read GCPs coordinates
XYZ = pd.read_csv("../data/Image/xyz.csv")[["x","y","z"]].values

# read UV coords
UV = pd.read_csv("../data/Image/uv.csv")[["u","v"]].values

# undistort frame
Kn,roi = cv2.getOptimalNewCameraMatrix(K,DC,(w,h),1,(w,h))
I = cv2.undistort(I, K, DC, None, Kn)

# find homography
H = ipwl.find_homography(UV, XYZ, K, z=0, distortion=0)

# rectify coordinates
X,Y = ipwl.rectify_image(I, H)
```
