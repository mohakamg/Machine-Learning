# Import the liabs
import cv2
import numpy as np

# Initialize the Camera to use. 0 defaults the 1st webcam availaible
cap = cv2.VideoCapture(0)

# Get a single Frame
ret, frame = cap.read()
# Get the size of the frame
height , width , layers =  frame.shape

# Output Stream
fourcc = cv2.VideoWriter_fourcc('m','p','4','v') # Codec
out = cv2.VideoWriter('output.m4v', fourcc, 30.0, (width, height))

# Run In an infinite loop to get frames
while True:
    # Get the frame
    ret, frame = cap.read()
    # Convert the frame to GrayScale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Display the frames
    cv2.imshow('BGR Frame', frame)
    cv2.imshow('Gray Frame', grayFrame)

    #print(frame.shape)

    # Write the Frame to the file
    out.write(frame)

    # Exit out of the loop if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and the output stream
cap.release()
out.release()

# Destroy the Windows
cv2.destroyAllWindows()
