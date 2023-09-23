from src.components.image_acquisition.frame_extraction import DataIngestion
from src.components.face_detection.face_detection import FaceDetection
from src.logger import logging
import os

import multiprocessing
from multiprocessing import Pool
import time

# Import the configuration class from config.py
from config.configuration import DetectionConfig #, device, FrameExtraction



# Function for data ingestion
def data_ingestion_worker(data_ingestion_instance):
    raw_face_data_path= data_ingestion_instance.initiate_data_ingestion()
    # output_queue.put(raw_face_data_path)


# Function for bounding box detection and face tracking
def bbox_detection_worker(face_detection_instance, frame_folder, bbox_json_path):
    try:
        face_detection_instance.process_frames_continuously(frame_folder, bbox_json_path)
    except Exception as e:
        logging.exception(f"Error in bbox_detection_worker: {str(e)}")




def main():

    # ---------------------------------------Create Instance--------------------------------------
    # Create an instance of the DataIngestion class
    data_ingestion_instance = DataIngestion()

    # Create an instance of the BBox Detection class
    face_detection_instance = FaceDetection()

    # ---------------------------------------- START PROCESS ---------------------------------------
    # Start data ingestion process
    video_process = multiprocessing.Process(target=data_ingestion_worker, args=(data_ingestion_instance,))
    video_process.start()
    print("Data Ingestion process starts................")
    # print(data_queue.get())

    # Start bbox detection process
    bbox_process = multiprocessing.Process(target=bbox_detection_worker, args=(face_detection_instance, DetectionConfig.frame_folder, DetectionConfig.bbox_json_path))
    bbox_process.start()
    print("BBox Detection and Face Tracking process starts..................")

    # -----------------------------------------JOIN PROCESS -----------------------------------------
    # Wait for both processes to finish
    video_process.join()
    bbox_process.join()

    print("Both Processes Successfully Executed in Parallel!")

    

if __name__ == "__main__":

    main()


# with Pool() as pool:
    #     pool.apply(func=bbox_detection_worker,args=(face_detection_instance, DetectionConfig.frame_folder, DetectionConfig.bbox_json_path))

##################################################################################################
