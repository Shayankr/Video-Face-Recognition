# from src.components.image_acquisition.frame_extraction import DataIngestion as Video_Data_Ingestion
from src.components.face_detection.face_detection import FaceDetection
from src.logger import logging
# import os

import multiprocessing
from multiprocessing import Pool
# import time

# Import the configuration class from config.py
from config.configuration import DetectionConfig #, device, FrameExtraction



# # Function for data ingestion
# def data_ingestion_worker(data_ingestion_instance, output_queue):
#     raw_face_data_path= data_ingestion_instance.initiate_data_ingestion()
#     output_queue.put(raw_face_data_path)


# Function for bounding box detection
def bbox_detection_worker(face_detection_instance, frame_folder, bbox_json_path):
    try:
        face_detection_instance.process_frames_continuously(frame_folder, bbox_json_path)
    except Exception as e:
        logging.exception(f"Error in bbox_detection_worker: {str(e)}")




def main():

    # Create an instance of the BBox Detection class
    face_detection_instance = FaceDetection()

    # Start bbox detection process
    bbox_process = multiprocessing.Process(target=bbox_detection_worker, args=(face_detection_instance, DetectionConfig.frame_folder, DetectionConfig.bbox_json_path))
    bbox_process.start()

    # Wait for the data_acuisition and bbox detection process to finish
    bbox_process.join()
    # with Pool() as pool:
    #     pool.apply(func=bbox_detection_worker,args=(face_detection_instance, DetectionConfig.frame_folder, DetectionConfig.bbox_json_path))
    print("BBox Detection Successfully Executed!")


if __name__ == "__main__":

    main()

    # # Create an instance of the DataIngestion class
    # video_data_ingestion = Video_Data_Ingestion()
   
    # # Create a queue to pass data between processes
    # data_queue = multiprocessing.Queue()

    # # Start data ingestion processes
    # video_process = multiprocessing.Process(target=data_ingestion_worker, args=(video_data_ingestion, data_queue))
    
    # video_process.start()
    # print(data_queue.get())


    # # Wait for 2 second to start bbox detection
    # # time.sleep(2)

    # # Retrieve the data from the queue
    # frame_folder_path = data_queue.get()

    # video_process.join()
    
    
    # print("Both Processes Successfully Executed in Parallel!")

##################################################################################################
