import cv2
import dlib
import os, sys
# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load input image
#path="./face_utils/tests/files/"
file="face_utils/tests/files/Obama.jpg"
img = cv2.imread(file)
#with open(os.path.join(path, file), 'rb') as f:
with open( file, 'rb') as f:
    check_chars = f.read()[-2:]
if check_chars != b'\xff\xd9':
    print('Not complete image')
else:
    #imrgb = cv2.imread(os.path.join(path, file), 1)
    imrgb = cv2.imread( file, 1)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Loop over detected faces
for face in faces:
    # Predict facial landmarks
    landmarks = predictor(gray, face)

    # Loop over the landmarks and draw them on the original image
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

# Display the output image
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#This code uses the dlib library for face detection and landmark prediction, and OpenCV for image processing and visualization.
# It first loads a pre-trained facial landmark predictor ("shape_predictor_68_face_landmarks.dat") and an input image, then detects faces in the 
# grayscale version of the image using the frontal face detector.
# For each detected face, it predicts the 68 facial landmarks using the predictor, and draws them on the original image. Finally, it displays the output image.



