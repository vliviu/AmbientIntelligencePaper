import cv2
import face_recognition

# Load the image of the person to recognize
known_image = face_recognition.load_image_file("face_utils/tests/files/Obama.jpg")

# Get the face encoding for the known image
known_face_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

# Load the image we want to check
unknown_image = face_recognition.load_image_file("face_utils/tests/files/liviu.jpg")

# Find all the faces in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Initialize the OpenCV window
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)

# Loop through each face found in the unknown image
for face_encoding, face_location in zip(face_encodings, face_locations):

    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces([known_face_encoding], face_encoding)

    # If a match was found, print the name of the known person
    if True in matches:
        face_names.append("Known Person")
    else:
        face_names.append("Unknown Person")

    # Draw a box around the face
    top, right, bottom, left = face_location
    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Draw a label with the name of the person
    cv2.rectangle(unknown_image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(unknown_image, face_names[-1], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Show the image with the face recognized
    cv2.imshow("Face Recognition", unknown_image)
    cv2.waitKey(0)

# Destroy the OpenCV window
cv2.destroyAllWindows()

# This code loads a known image of a person, gets the face encoding for it,
# and then loads an unknown image to check for matches. It uses the face_recognition library to find 
# all the faces in the unknown image and then compares them to the known face encoding to see if there's a match.
# If there is a match, it draws a box around the face and labels it as the known person.
# If there isn't a match, it labels the face as an unknown person.

