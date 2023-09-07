import os
import sys
import cv2
import sqlite3  # Import the sqlite3 module
import time
import datetime  # Import the datetime module

from src.logger import logging
from src.exception import CustomException

# Initialize the Data Ingestion Configuration:
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_face_data_path:str = os.path.join('artifacts', 'video_raw_face_data')
    video_data_path:str = os.path.join('artifacts', 'video','Billi_Billi.mp4')
    database_path: str = os.path.join('artifacts', 'subsystem_info', 'frame_acquisition_status.db')  # Specify your database file path
    # frames_per_acquisition = 30 * 5  # 30 seconds at 5 FPS


## Create the data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.ingestion_config.frames_per_acquisition = 30 * 10  # 30 seconds at 10 FPS


    def create_acquisition_table(self, cursor):

        try:
            # Generate a unique table name using a timestamp
            table_name = f"acquisition_data_{int(time.time())}"
            logging.info(f'Creating table first time:{table_name}')

            # Create a table with the unique name
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    subfolder_name TEXT PRIMARY KEY,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    first_frame_filename TEXT,
                    end_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_frame_filename TEXT,
                    flag_status BOOLEAN NOT NULL DEFAULT FALSE,
                    next_status BOOLEAN NOT NULL DEFAULT FALSE
                )
            ''')
            return table_name

        except Exception as e:
            logging.info('!!Exception Occured at SQL Table Creation Stage!!')
            raise CustomException(e, sys)
        


    def insert_acquisition_data(self, cursor, table_name, data):
        try:
            # Insert acquisition data into the table
            cursor.execute(f'''
                INSERT INTO {table_name} (subfolder_name, start_time, first_frame_filename, end_time, last_frame_filename, flag_status, next_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', data)
            logging.info(f'Data inserted successfully in sql - data: {data}')

        except Exception as e:
            logging.info('!!Exception occured at Data Insertion in SQL Table Stage!!')
            raise CustomException(e, sys)
        


    def initiate_data_ingestion(self):
        logging.info('Data ingestion method starts from video:')

        try:
            # Create raw_face_data_path folder if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_face_data_path), exist_ok=True)

            # Set the desired FPS (e.g., 30 FPS)
            desired_fps = 10

            # Open the video file with the specified FPS
            cap = cv2.VideoCapture(self.ingestion_config.video_data_path)

            # see the default fps of video:
            fps = cap.get(cv2.CAP_PROP_FPS)
            logging.info(f'Video is extracting the frames at fps: {fps}')

            # Changing fps rate
            cap.set(cv2.CAP_PROP_FPS, desired_fps)
            logging.info(f' Now, Video is extracting the frames at fps: {cap.get(cv2.CAP_PROP_FPS)}')

            # Check if camera opened successfully
            if (cap.isOpened()== False): 
                logging.info("Error in opening video stream or file.")
                

            # Initialize variables
            frame_count = 0
            acquisition_number = 1
            

            # Create acquisition folders
            acquisition_folder = os.path.join(self.ingestion_config.raw_face_data_path, f"acquisition_{acquisition_number}")  
            os.makedirs(acquisition_folder, exist_ok=True) # Create the initial acquisition folder

            # Create or connect to the SQLite database
            conn = sqlite3.connect(self.ingestion_config.database_path)
            cursor = conn.cursor()

            # Create the acquisition_data table if it doesn't exist
            table_name = self.create_acquisition_table(cursor)

            # Get the start time (current time)
            start_time = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

            start_frame_filename = None  # Initialize before the while loop
            frame_filename = None  # Initialize before the while loop

            while True:
                # Read a frame from the video
                ret, frame = cap.read()

                # Check if we've reached the end of the video
                if not ret:
                    logging.info("Ingestion of Video Data is Ended!-- No More Frames.")

                    # if frame_count == 0:
                    #     break

                    # # Insert acquisition data into the database
                    # acquisition_info = (
                    #     f"acquisition_{acquisition_number}",
                    #     None,  # You can add the start time here
                    #     start_frame_filename, # it will assigned if atleast one frame is extracted o.w. throw error and goes to exception handling block.
                    #     datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S'),
                    #     frame_filename, # This will give lastly extracted frame name or throw error --> Exception handling block
                    #     True,  # Initialize as False
                    #     False  # You can set this based on your logic -- next_status willbe False nere.
                    # )
                    # self.insert_acquisition_data(cursor, table_name, acquisition_info)

                    break

                # Save the frame to the current acquisition folder
                frame_filename = f"frame_{frame_count:04d}.jpg"
                if (frame_count) == 0:
                    start_frame_filename = frame_filename
                frame_path = os.path.join(acquisition_folder, frame_filename)
                cv2.imwrite(frame_path, frame)

                frame_count += 1

                # Check if it's time to create a new acquisition folder
                if frame_count >= self.ingestion_config.frames_per_acquisition:
                    logging.info(f"Ingestion of Video Data to acquisition_no {acquisition_number} is completed")

                    # Insert acquisition data into the database
                    acquisition_info = (
                        f"acquisition_{acquisition_number}",
                        start_time,  # You can add the start time here
                        start_frame_filename, # this is assigned in above code based on frame_count == 0
                        datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S'),  # Since you want to use the default value for end_time, you can set it to None
                        frame_filename, # this will provide last executed frame filename automatically
                        True,  # Initialize as False -- current flag_status
                        True  # You can set this based on your logic -- next_flag_status
                    )
                    self.insert_acquisition_data(cursor, table_name=table_name, data=acquisition_info)
                    conn.commit()
                    
                    frame_count = 0
                    acquisition_number += 1
                    acquisition_folder = os.path.join(self.ingestion_config.raw_face_data_path, f"acquisition_{acquisition_number}")
                    os.makedirs(acquisition_folder, exist_ok=True)


                    # Get the start time (current time)
                    if (acquisition_number>1):
                        start_time = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')


            # Commit the changes and close the database connection
            conn.commit()
            conn.close()

            # Release the video capture object
            cap.release()

            logging.info("Ingestion of Video Data is completed")

            return (self.ingestion_config.raw_face_data_path, self.ingestion_config.database_path, table_name)

        except Exception as e:
            logging.info('Exception occured at Video Data Ingestion Stage')
            raise CustomException(e, sys)

     
        
#####################################################################################################################################