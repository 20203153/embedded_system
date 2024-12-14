import cv2
import numpy as np
import os
import pandas as pd
import time
from picamera2 import Picamera2

# Set output directory
output_dir = './data'
os.makedirs(output_dir, exist_ok=True)

# Initialize Picamera2
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(camera_config)
picam2.start()

# Parameters for circle detection
dp = 1.2
min_dist = 20
param1 = 50  # Higher threshold for Canny edge detector
param2 = 30  # Threshold for circle detection
min_radius = 10
max_radius = 50

frame_count = 0

try:
    while True:
        # Capture image
        img = picam2.capture_array()
        
        # Generate a unique timestamp-based filename
        timestamp = int(time.time() * 1000)  # Unique timestamp in ms
        filename = os.path.join(output_dir, f"frame_{timestamp}.png")
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Display the captured frame
        cv2.imshow('Frame', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Wait for 100ms
        if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to exit
            break
        
        frame_count += 1

except KeyboardInterrupt:
    print("Manual interruption")

finally:
    picam2.stop()
    cv2.destroyAllWindows()