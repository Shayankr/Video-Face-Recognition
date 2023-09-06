# Importing necessary libraries:
import os
import json
import cv2
from PIL import Image
import numpy as np
from retinaface.pre_trained_models import get_model


from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
import sys


@dataclass
class FaceDetectionConfig:
    model_name = "resnet50_2020-07-20"
    max_size = 2048
    device = "cpu"


class FaceDetection:
    def __init__(self):
        self.model = get_model(
            model_name=FaceDetectionConfig.model_name,
            max_size=FaceDetectionConfig.max_size,
            device=FaceDetectionConfig.device
        )
        self.model.eval()

    def detect_faces_and_save_to_json(self, image_path, save_bbox_path):
        try:
            # logging.info("Detecting Faces and Saving in JSON Format:")
            # Open the JPEG image using Pillow
            image = Image.open(image_path)

            # Convert the Pillow image to a NumPy array
            image = np.array(image)

            # Perform face detection
            annotation = self.model.predict_jsons(image)

            # Extract the file name without extension from the image path
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            # Create a JSON file path for saving the results
            json_file_path = os.path.join(save_bbox_path, "acquisition_1", f"{image_name}_results.json")

            # Save the face detection results to the JSON file
            with open(json_file_path, "w") as json_file:
                json.dump(annotation, json_file)
        
        except Exception as e:
            logging.info(f"Error in Processing Face Detection and Saving in JSON Format Step for: {image_path}!")
            raise CustomException(e, sys)

    def process_images_in_folder(self, image_folder, bbox_path):
        # Iterate through all images in the folder
        for filename in os.listdir(image_folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                image_path = os.path.join(image_folder, filename)
                self.detect_faces_and_save_to_json(image_path, bbox_path)

 


#############################################################################################################################
if __name__ == "__main__":
    # Base directory
    base_dir = "artifacts"
    
    # Directory containing your images
    image_folder = os.path.join(base_dir, "raw_face_data", "acquisition_1")

    # Directory to save Bounding-Box:
    bbox_path = os.path.join(base_dir, "bbox")

    face_detection = FaceDetection()
    face_detection.process_images_in_folder(image_folder, bbox_path)




#############################################################################################################################
