import cv2
import dlib
import imutils
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer

# Initialize the mixer for sound alerts
mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    """
    Calculate the Eye Aspect Ratio (EAR)
    :param eye: coordinates of the eye landmarks
    :return: EAR value
    """
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for EAR to detect blink
thresh = 0.25
frame_check = 20

# Initialize dlib's face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Get the indices for the left and right eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start video capture from the webcam
cap = cv2.VideoCapture(0)
flag = 0

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=450)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)

    for subject in subjects:
        # Get the landmarks/parts for the face in box
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Get the coordinates for the left and right eye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate the EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Visualize the eyes using convex hull
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the EAR is below the threshold
        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT! WAKE UP****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT! WAKE UP****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if the 'q' key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
