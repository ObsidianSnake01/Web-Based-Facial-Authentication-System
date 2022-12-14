import cv2

# Load the facial recognition model
model = cv2.face.LBPHFaceRecognizer_create()
model.read('model.xml')

# Initialize the video capture device
video_capture = cv2.VideoCapture(0)

# Continuously capture video frames
while True:
  # Capture a single frame of video
  ret, frame = video_capture.read()

  # Convert the captured frame to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect faces in the frame
  faces = cv2.face.detectMultiScale(gray, 1.3, 5)

  # Iterate over the detected faces
  for (x, y, w, h) in faces:
    # Predict the user's identity using the facial recognition model
    id, confidence = model.predict(gray[y:y+h, x:x+w])

    # Check if the predicted identity is correct
    if id == 1 and confidence < 100:
      print('Hello, User!')
    else:
      print('Unknown user')

  # Exit the loop if the 'q' key is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the video capture device
video_capture.release()
cv2.destroyAllWindows()
