import os
import sys
import cv2
import sqlite3  # Import the sqlite3 module

from src.logger import logging
from src.exception import CustomException

# Initialize the Data Ingestion Configuration:
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_face_data_path:str = os.path.join('artifacts', 'raw_face_data')
    video_data_path:str = os.path.join('artifacts', 'video','Deewangi Deewangi 4k Video Song Om Shanti Om Shahrukh Khan, Deepika Padukone Classic Super HCSN.mp4')
    database_path: str = os.path.join('artifacts', 'subsystem_info', 'frame_acquisition_status.db')  # Specify your database file path
    # frames_per_acquisition = 30 * 5  # 30 seconds at 5 FPS

## Create the data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.ingestion_config.frames_per_acquisition = 30 * 5  # 30 seconds at 5 FPS

    def create_acquisition_table(self, cursor):
        # Create a table to store acquisition data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS acquisition_data (
                subfolder_name TEXT PRIMARY KEY,
                start_time DATETIME,
                first_frame_filename TEXT,
                end_time DATETIME,
                last_frame_filename TEXT,
                flag_status BOOLEAN NOT NULL,
                next_status BOOLEAN
            )
        ''')

    def insert_acquisition_data(self, cursor, data):
        # Insert acquisition data into the table
        cursor.execute('''
            INSERT INTO acquisition_data (subfolder_name, start_time, first_frame_filename, end_time, last_frame_filename, flag_status, next_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)

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
            

            # Create acquisition folders
            acquisition_folder = os.path.join(self.ingestion_config.raw_face_data_path, f"acquisition_{acquisition_number}")  
            os.makedirs(os.path.dirname(acquisition_folder), exist_ok=True)

            # Create or connect to the SQLite database
            conn = sqlite3.connect(self.ingestion_config.database_path)
            cursor = conn.cursor()

            # Create the acquisition_data table if it doesn't exist
            self.create_acquisition_table(cursor)

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
                if frame_count >= self.ingestion_config.frames_per_acquisition:
                    logging.info(f"Ingestion of Video Data to acquisition_no {acquisition_number} is completed")
                    frame_count = 0
                    acquisition_number += 1
                    acquisition_folder = os.path.join(self.ingestion_config.raw_face_data_path, f"acquisition_{acquisition_number}")
                    os.makedirs(os.path.dirname(acquisition_folder), exist_ok=True)

                    # Insert acquisition data into the database
                    acquisition_info = (
                        f"acquisition_{acquisition_number}",
                        "",  # You can add the start time here
                        f"frame_{(acquisition_number - 1) * self.ingestion_config.frames_per_acquisition:04d}.jpg",
                        "",  # You can add the end time here
                        f"frame_{(acquisition_number - 1) * self.ingestion_config.frames_per_acquisition + self.ingestion_config.frames_per_acquisition - 1:04d}.jpg",
                        False,  # Initialize as False
                        True  # You can set this based on your logic
                    )
                    self.insert_acquisition_data(cursor, acquisition_info)
                    conn.commit()

            # Commit the changes and close the database connection
            conn.commit()
            conn.close()

            # Release the video capture object
            cap.release()

            logging.info("Ingestion of Video Data is completed")

            return self.ingestion_config.raw_face_data_path

        except Exception as e:
            logging.info('!!Exception occured at Video Data Ingestion Stage!!')
            raise CustomException(e, sys)

     
        
#####################################################################################################################################