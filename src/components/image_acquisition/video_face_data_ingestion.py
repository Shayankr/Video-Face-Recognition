import os
import sys
import cv2

from src.logger import logging
from src.exception import CustomException

# Initialize the Data Ingestion Configuration:
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_face_data_path:str = os.path.join('artifacts', 'raw_face_data')
    video_data_path:str = os.path.join('artifacts', 'video','Deewangi Deewangi 4k Video Song Om Shanti Om Shahrukh Khan, Deepika Padukone Classic Super HCSN.mp4')

## Create the data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion method starts from video:')

        try:
            # Create raw_face_data_path folder if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_face_data_path), exist_ok=True)

            # Open the video file
            cap = cv2.VideoCapture(self.ingestion_config.video_data_path)

            # Initialize variables
            frame_count = 0
            acquisition_number = 1
            frames_per_acquisition = 30 * 5  # 30 seconds at 5 FPS

            # Create acquisition folders
            acquisition_folder = os.path.join(self.ingestion_config.raw_face_data_path, f"acquisition_{acquisition_number}")  
            os.makedirs(os.path.dirname(acquisition_folder), exist_ok=True)

            while True:
                # Read a frame from the video
                ret, frame = cap.read()

                # Check if we've reached the end of the video
                if not ret:
                    break

                # Save the frame to the current acquisition folder
                frame_filename = f"frame_{frame_count:04d}.jpg"
                frame_path = os.path.join(acquisition_folder, frame_filename)
                cv2.imwrite(frame_path, frame)

                frame_count += 1

                # Check if it's time to create a new acquisition folder
                if frame_count >= frames_per_acquisition:
                    logging.info(f"Ingestion of Video Data to acuisition_no {acquisition_number} is completed")
                    frame_count = 0
                    acquisition_number += 1
                    acquisition_folder = os.path.join(self.ingestion_config.raw_face_data_path, f"acquisition_{acquisition_number}")
                    os.makedirs(os.path.dirname(acquisition_folder), exist_ok=True)

            # Release the video capture object
            cap.release()

            logging.info("Ingestion of Video Data is completed")

            return self.ingestion_config.raw_face_data_path

        except Exception as e:
            logging.info('!!Exception occured at Video Data Ingestion Stage!!')
            raise CustomException(e, sys)

      
        
#####################################################################################################################################