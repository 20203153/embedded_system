import hailo_platform as hlp
import cv2
import numpy as np
from picamera2 import Picamera2, Preview
from picamera2.devices import Hailo

# Initialize Hailo device and model
INPUT_RES_H = 480
INPUT_RES_W = 640

hef_path = 'model.hef'

def sigmoid(z):
    return 1/(1 + np.exp(-z))

with Hailo(hef_path) as hailo:
    # Picamera2
    picam2 = Picamera2()

    picam2.configure(picam2.create_video_configuration(main={"size": (1280, 720)}, lores={'size': hailo.get_input_shape(), 'format': 'RGB888'}))

    try:
        picam2.start()

        while True:
            img = picam2.capture_array('lores')
            results = sigmoid(np.array(hailo.run(img)))

            print(results.max())

            # Assuming results are in float range [0, 1]
            # Scale to [0, 255] and convert to uint8
            scaled_heatmap = (results * 255).astype('uint8')

            # Apply color map to the scaled heatmap
            heatmap = cv2.applyColorMap(scaled_heatmap, cv2.COLORMAP_JET)

            # Display the camera feed
            cv2.imshow("Camera Feed", img)

            # Display the heatmap
            cv2.imshow("Heatmap", heatmap)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
    except KeyboardInterrupt:
        print("inturrupted.")
    except Exception as ex:
        print(ex)
        pass
    finally:
        picam2.stop()