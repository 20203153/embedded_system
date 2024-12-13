import hailo_platform as hlp
import cv2
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams)
from hailo_platform import FormatType
from picamera2 import Picamera2, Preview

# Initialize Hailo device and model
INPUT_RES_H = 480
INPUT_RES_W = 640

hef_path = 'model.hef'
hef = HEF(hef_path)

devices = Device.scan()
# Define prediction format if needed

with VDevice(device_ids=devices) as target:
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    height, width, channels = hef.get_input_vstream_infos()[0].shape

    # Picamera2 ê°ì²´ ìì±
    picam2 = Picamera2()

    # ì¹´ë©ë¼ í´ìë ì¤ì 
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))

    # ë¯¸ë¦¬ë³´ê¸° ìì
    picam2.start_preview(Preview.QTGL)

    # ë¹ëì¤ ì¤í¸ë¦¬ë° ìì
    picam2.start()

picam2.stop()
cv2.destroyAllWindows()