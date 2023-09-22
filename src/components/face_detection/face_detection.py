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
from config.configuration import DetectionConfig, device, TrackingConfig


@dataclass
class FaceDetectionConfig(DetectionConfig, TrackingConfig):
    pass


class FaceDetection:
    def __init__(self):
        self.model = get_model(
            model_name=FaceDetectionConfig.model_name,
            max_size=FaceDetectionConfig.max_size,
            device=device   # Use the device variable from config.py
        )
        self.model.eval()
        self.detection_config = FaceDetectionConfig()

    def detect_annotations(self, image_path):
        try:
            logging.info(f"Detecting Faces and Saving in JSON Format for image_path: {image_path}")
            # Open the JPEG image using Pillow
            image = Image.open(image_path)

            # Convert the Pillow image to a NumPy array
            image = np.array(image)

            # Perform face detection
            annotation = self.model.predict_jsons(image)
            logging.info(f"annotation detected for image_path: {image_path}")

            return image, annotation
        
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

            # Capture the video or webcam for draw the rectangles on each detected face using bbox -- shows face tracking
            # cap = cv2.VideoCapture(self.detection_config.video_data_path)

            # Resolution of frame:
            resolution = self.detection_config.frame_resolution # (width, height)
            fps_rate_save = self.detection_config.fps_output

            # Initialize the VideoWriter
            out = cv2.VideoWriter(
                os.path.join(self.detection_config.video_save_path, "output.avi"),
                cv2.VideoWriter_fourcc(*'DIVX'), fps_rate_save, resolution
            )

            logging.info(f"An object [{out}] is created for saving video with bounding box:")

            # Variable to keep track of whether we have processed all frames
            all_frames_processed = False

            while not all_frames_processed:
                # Check if the frame for the current frame number exists
                framename = f"frame_{n_frames}.jpg"
                frame_path = os.path.join(frame_folder, framename)

                n_frames += 1

                if os.path.exists(frame_path):
                    # Perform face detection
                    logging.info(f"BBox extraction for frame number: {n_frames}")
                    img, annot = self.detect_annotations(frame_path)

                    try:
                        logging.info(f"Tracking face started for frame - {n_frames}")
                        # Draw rectangles on each face based on bbox annot

                        # n_detected_faces = len(annot)
                        # for face_id in range(n_detected_faces):
                        #     left, bottom, right, top = annot[face_id]["bbox"]
                        #     img_with_rectangles = cv2.rectangle(img, (left, bottom), (right, top), self.detection_config.color[face_id % len(self.detection_config.color)], 4)

                        for face_id, bbox in enumerate(annot):
                            if len(bbox.get("bbox", [])) == 4:  # Check if "bbox" has 4 values
                                left, bottom, right, top = bbox["bbox"]
                                img_with_rectangles = cv2.rectangle(
                                    img, (left, bottom), (right, top),
                                    self.detection_config.color[face_id % len(self.detection_config.color)], 3
                                )
                            else:
                                logging.info(f"No valid bounding box for face {face_id}, skipping.")


                        # Update the VideoWriter
                        out.write(img_with_rectangles)
                        logging.info(f"Faces are tracked for frame - {n_frames}")

                        # Here, I want to save each annot with frame number as another key value pair like "frame_no":n_frames-1 and "time":datatime.dateime.now()

                        # Save annotation with frame number and timestamp to the final JSON file
                        self.save_annotation_to_final_json(final_json_path, annot, n_frames)


                    except Exception as e:
                        logging.info(f"Error in face tracking for frame - {n_frames}")
                        raise CustomException(e, sys)

                else:
                    # If the frame for the current frame number doesn't exist, exit the loop
                    all_frames_processed = True

            # Release the VideoWriter object
            out.release()
            logging.info("All frames processed successfully.")


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
