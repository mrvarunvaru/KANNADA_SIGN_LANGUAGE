# Import OpenCV2 for image processing
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
# Start capturing video 
vid_cam = cv2.VideoCapture(0)

# Start looping
while(True):

    # Capture video frame
    _, image_frame = vid_cam.read()

    image_frame = Image.fromarray(image_frame)
    
    # Initialize the drawing context with the image
    draw = ImageDraw.Draw(image_frame)

    # Define the Kannada text to be written on the image
    text = "ಒಂದು"

    # Define the font for the text
    font = ImageFont.truetype("kannada.ttf", size=50)

    # Write the text on the image
    draw.text((100, 100), text, fill=(255, 255, 255), font=font)

    image_frame = np.asarray(image_frame)
    
    cv2.imshow('frame', image_frame)
    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop video
vid_cam.release()
cv2.destroyAllWindows()
