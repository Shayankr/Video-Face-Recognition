import os
import sys
import cv2
import time
import datetime  # Import the datetime module

from src.exception import CustomException
from src.logger import logging


# Initialize the Data Ingestion Configuration:
from dataclasses import dataclass

# Import the configuration class from config.py
# from config_ import FrameExtraction
from config.configuration import FrameExtraction

@dataclass
class DataIngestionConfig(FrameExtraction):
    pass

## Create the data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion method starts from video:')

        try:
            # Create raw_face_data_path folder if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.frame_folder), exist_ok=True)

            # Set the desired FPS (e.g., 30 FPS)
            desired_fps = FrameExtraction.fps

            # Create a 'cap' object: // to select video or webcam:
            if (self.ingestion_config.capture_mode == "video"):
                # Open the video file with the specified FPS
                cap = cv2.VideoCapture(self.ingestion_config.video_data_path)
            elif(self.ingestion_config.capture_mode == "webcam"):
                cap = cv2.VideoCapture(0)

            # Check if camera opened successfully
            if (cap.isOpened()== False): 
                logging.info("Error in opening video stream or file.")

            # see the default fps of video:
            fps = cap.get(cv2.CAP_PROP_FPS)
            logging.info(f'Video is extracting the frames at fps: {fps}')

            # Changing fps rate
            cap.set(cv2.CAP_PROP_FPS, desired_fps)
            logging.info(f' Now, Video is extracting the frames at fps: {cap.get(cv2.CAP_PROP_FPS)}')
                
            # Initialize variables
            frame_count = 0

            while True:
                # Read a frame from the video
                ret, frame = cap.read()

                # Check if we've reached the end of the video
                if not ret:
                    logging.info("Ingestion of Video Data is Ended!-- No More Frames.")
                    break
                
                # Save the frame to the current acquisition folder
                # frame_filename = f"frame_{frame_count:04d}.jpg"
                frame_filename = f"frame_{frame_count}.jpg"
                frame_path = os.path.join(self.ingestion_config.frame_folder, frame_filename)
                cv2.imwrite(frame_path, frame)

                frame_count += 1


            # Release the video capture object
            cap.release()

            logging.info("Ingestion of Video Data is completed")

            return self.ingestion_config.frame_folder

        except Exception as e:
            logging.info('Exception occured at Video Data Ingestion Stage')
            raise CustomException(e, sys)

     
        
#####################################################################################################################################
# if __name__ == "__main___":
#     # Create an instance of the BBox Detection class
#     frame_extraction__instance = DataIngestion()
#     frame_folder_path = frame_extraction__instance.initiate_data_ingestion()

#     print(frame_folder_path)

#####################################################################################################################################