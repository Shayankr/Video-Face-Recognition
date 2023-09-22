from src.components.image_acquisition.frame_extraction import DataIngestion
from src.components.face_detection.face_detection import FaceDetection
from config.configuration import DetectionConfig
import os



def main():

    # Create an instance of the Data Ingestion class
    data_ingestion_instance = DataIngestion()
    frame_folder_path = data_ingestion_instance.initiate_data_ingestion()



    # Create an instance of the BBox Detection class
    face_detection_instance = FaceDetection()
    face_detection_instance.process_frames_continuously(DetectionConfig.frame_folder, DetectionConfig.bbox_json_path )
    
    
    print("Both Processes Successfully Executed !")


if __name__ == "__main__":
    main()


