
from imageai.Detection import VideoObjectDetection
import os
import cv2
import process
from matplotlib import pyplot as plt


execution_path = os.getcwd()

camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()

detector.setModelTypeAsRetinaNet()

detector.setModelPath(os.path.join(execution_path , "Models\\for video\\resnet50_coco_best_v2.0.1.h5"))

detector.loadModel(detection_speed="fastest")

# detector is imageai object for deteting object
# detectObjectsFromVideo is function to detect object in video
detector.detectObjectsFromVideo(
    # input_file_path=os.path.join(execution_path, "data-videos\\traffic-mini.mp4"),
    camera_input = camera,
    save_detected_video=False,
    per_second_function=process.forSeconds,
    per_frame_function=process.forFrame,    # after detecting, detector calls per_frame_function(we set per_frame_function to )
    per_minute_function=process.forMinute,
    minimum_percentage_probability=30,
    return_detected_frame=True,
    frames_per_second=10,
    display_object_name=False
)
process.endProcess()
camera.release() # Error is here
cv2.destroyAllWindows()