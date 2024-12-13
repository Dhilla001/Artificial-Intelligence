# install "torch", "torchvision", "facial-emotion-recognition" library
# change to be made in "C:\Users\DELL\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\serialization.py"
#   in def load() change map_location to 'cpu' when you dont have gpu

from facial_emotion_recognition import EmotionRecognition
import cv2

er = EmotionRecognition(device='cpu')  # Make use of cpu instead of gpu
cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()  # Capture a frame from the video
    if frame is not None:  # Ensure a valid frame was captured
        frame = er.recognise_emotion(frame, return_type='BGR')  # Process the frame for emotion recognition
        cv2.imshow("Emotion Detection", frame)  # Display the processed frame

        if cv2.waitKey(10) & 0xFF == 27:  # Wait for 'Esc' key to exit
            break
    else:
        print("Failed to capture frame")
        break

cam.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all OpenCV windows
