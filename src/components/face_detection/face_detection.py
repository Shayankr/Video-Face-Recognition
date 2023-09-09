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
import datetime
import time

# Import the configuration class from config.py
# from config_ import DetectionConfig, device
from config.configuration import DetectionConfig, device


@dataclass
class FaceDetectionConfig(DetectionConfig):
    pass


class FaceDetection:
    def __init__(self):
        self.model = get_model(
            model_name=FaceDetectionConfig.model_name,
            max_size=FaceDetectionConfig.max_size,
            device=device   # Use the device variable from config.py
        )
        self.model.eval()

    def detect_annotations(self, image_path):
        try:
            logging.info(f"Detecting Faces and Saving in JSON Format for image_path: {image_path}")
            # Open the JPEG image using Pillow
            image = Image.open(image_path)

            # Convert the Pillow image to a NumPy array
            image = np.array(image)

            # Perform face detection
            annotation = self.model.predict_jsons(image)
            logging.info(f"annottaion detected for image_path: {image_path}")

            return annotation
        
        except Exception as e:
            logging.info(f"Error in Processing Face Detection and Saving in JSON Format Step for: {image_path}!")
            raise CustomException(e, sys)
        

    def save_annotation_to_final_json(self, final_json_path, annotation, frame_number):
        try:
            # Create or append to the final JSON file

            # Create a dictionary with frame number, timestamp, and annotation           
            frame_info = {
                "time": str(datetime.datetime.now()),
                "annotation": annotation
            }

            # Initialize the json_data variable
            json_data = {}

            # Check if the JSON file exists
            if os.path.isfile(final_json_path):
                # Read the existing JSON data
                with open(final_json_path, "r") as json_file:
                    json_data = json.load(json_file)
            
                # Add the frame_info to the existing JSON data using the frame number as the key
                json_data[f"frame_{frame_number}"] = frame_info

                # Write the updated data back to the JSON file
                with open(final_json_path, "w") as json_file:
                    json.dump(json_data, json_file, indent=4)  # Add indent for readability
                logging.info(f"frame_{frame_number} annoatation is appended in BBox Json file")
            else:
                logging.info(f"JSON File: {final_json_path} does not exist!")
        
        except Exception as e:
            logging.info(f"Error in Saving Annotation to Final JSON: {final_json_path}!")
            raise CustomException(e, sys)
        


    def process_frames_continuously(self, frame_folder, final_json_path):
        try:
            logging.info(f"Continuously processing frames for BBox from folder: {frame_folder}")

            n_frames = 0

            if not os.path.exists(frame_folder):
                logging.info(f"Frame folder not found: {frame_folder}")
                return  # Exit if the folder doesn't exist

            if len(os.listdir(frame_folder))==0:
                logging.info("No frame present inside the frame_folder")

            # Check if the JSON file exists
            if os.path.isfile(final_json_path):
                with open(final_json_path, "w") as json_file:
                    json.dump({}, json_file, indent=4)
            else:
                logging.info(f"JSON File: {final_json_path} does not exist!")

            ## Run this loop until processing all frames inside frame_folder -- since no. of frames in frame_folder is variable.
            while(len(os.listdir(frame_folder)) > n_frames):
                logging.info(f"BBox extraction for frame number: {n_frames}")

                # Get a list of image files in the folder
                frame_files = os.listdir(frame_folder)

                # Sort the list of files based on their names (assuming filenames follow the frame_xxxx.jpg convention)
                frame_files.sort()

                # Iterate through all images in the folder
                framename = frame_files[n_frames]
                n_frames += 1

                if framename.endswith(".jpg") or framename.endswith(".jpeg") or framename.endswith(".png"):
                    frame_path = os.path.join(frame_folder, framename)
                    annot = self.detect_annotations(frame_path)
                    # Here, I want to save each annot with frame number as another key value pair like "frame_no":n_frames-1 and "time":datatime.dateime.now()

                    # Save annotation with frame number and timestamp to the final JSON file
                    self.save_annotation_to_final_json(final_json_path, annot, n_frames)

        except Exception as e:
            logging.info(f"Error in Processing Image folder for finding BBox Step for: {frame_folder}!")
            raise CustomException(e, sys)
        

 


#############################################################################################################################
# if __name__ == "__main__":

#     # Final JSON file to store annotations with frame numbers and timestamps
#     # final_json_path = FaceDetectionConfig.bbox_json_path

#     face_detection = FaceDetection()
#     face_detection.process_frames_continuously(FaceDetectionConfig.frame_folder, FaceDetectionConfig.bbox_json_path )



#############################################################################################################################
