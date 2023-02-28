# Face detection & recognition with OpenCV, Python and deep learning
Related links:
https://towardsdatascience.com/using-linkedin-profile-pictures-for-facial-recognition-8be709e8fac

[https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)

Significance of the facial landmarks is given here:
1) https://pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
2) in accordance with: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

## Setup

- Python 3.6.6+ 

### OpenCV 4.x

Make sure OpenCV4 is installed.

### Libraries

`pip install dlib`

`pip install face_recognition`

`pip install imutils`

## Creating image encodings

You will need to run `encode_faces.py`

This script takes 3 parameters:
```python
ap.add_argument("-d", "--dataset", required=False, default="/Volumes/MacBackup/friends_family", help="path to input dataset directory")
ap.add_argument("-e", "--encodings-file", required=False, default='encodings/friends_family_encodings.pkl', help="path to serialized db of facial encodings")
ap.add_argument("-m", "--detection-method", type=str, default='cnn', help="face detection model to use: either 'hog' or 'cnn' ")

```
* dataset is the path to the root directory where subdirectories exist and contain image of people.

For example, above hte 'friends_family' contains folders like:

bob_builder

john_public

jane_doe

where each of those folders contain images of those people.

* encoding-file is the name of the output file that contains the encodings for all of the people in the dataset directory

* detection-method is the face finding method.  If you are on a laptop or better, use 'cnn'

