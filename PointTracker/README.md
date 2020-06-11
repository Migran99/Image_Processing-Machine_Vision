# This applies the Pyramid Lukas Kanade algorithm to track points in a squence of images (a.k.a Video)

## Sources
The main source of information and on what this code is based can be found here:

[OpenCV Wiki](https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323)

[OpenCV Tutorial](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html)

## Installation
In order to install this software it's recommended to use a virtual enviroment: [Virtualenv](https://virtualenv.pypa.io/en/latest/)

`virtualenv env`

Then activate it:

`In windows: ./env/scripts/activate`

And then you have to install the package and dependencies:

`pip install opencv2`

`pip install numpy`

## Usage
Just run this in the command line:

```
python tracker.py ("name of the video file . extension") ("draw trajectories" = 1 / 0) ("number of points to track" = int)
```