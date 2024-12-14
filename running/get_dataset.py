import cv2
import numpy as np
import os
import pandas as pd
import time
from picamera2 import Picamera2

# Set output directory
output_dir = './data'
os.makedirs(output_dir, exist_ok=True)

# Initialize an empty DataFrame for storing labels
labels_columns = ['filename', 'ball_detected', 'x', 'y']
labels_df = pd.DataFrame(columns=labels_columns)

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
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Use HoughCircles to detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist,
                                   param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
        
        # Initialize variables
        ball_detected = 0
        x, y = None, None  # Default if no circle is found
        
        # Check if any circles were detected
        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))  # Convert to integer
            
            # Assuming one main ball to be detected
            for (x, y, r) in circles:
                if min_radius < r < max_radius:
                    ball_detected = 1
                    break  # Get the first valid ball and exit loop
        
            # Visual debugging (optional)
            if ball_detected:
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        
        # Generate a unique timestamp-based filename
        timestamp = int(time.time() * 1000)  # Unique timestamp in ms
        filename = os.path.join(output_dir, f"frame_{timestamp}.png")
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Create a new DataFrame row
        new_row = pd.DataFrame([[filename, ball_detected, x, y]], columns=labels_columns)
        labels_df = pd.concat([labels_df, new_row], ignore_index=True)
        
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
    # Save labels DataFrame to CSV
    labels_df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)