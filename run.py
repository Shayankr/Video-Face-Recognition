# from app import create_app

# app = create_app()

# if __name__ == '__main__':
#     app.run(debug=True)

from src.components.face_detection.face_detection import FaceDetection
from config.configuration import DetectionConfig
import os







# Create an instance of the BBox Detection class
face_detection_instance = FaceDetection()

face_detection_instance.process_frames_continuously(DetectionConfig.frame_folder, DetectionConfig.bbox_json_path )
print("BBox Detection Successfully Executed!")

