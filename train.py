from src.components.image_acquisition.video_face_data_ingestion_sqlite import DataIngestion as Video_Data_Ingestion
from src.components.image_acquisition.webcam_face_data_ingestion_sqlite import DataIngestion as Webcam_Data_Ingestion




if __name__ == "__main__":
    # Create an instance of the DataIngestion class
    video_data_ingestion = Video_Data_Ingestion()
    webcam_data_ingestion = Webcam_Data_Ingestion()

    # Call the initiate_data_ingestion method for video
    raw_face_data_path, database_path, table_name = video_data_ingestion.initiate_data_ingestion()

    # Call the initiate_data_ingestion method for Webcam
    # raw_face_data_path, database_path, table_name = webcam_data_ingestion.initiate_data_ingestion()

    #

#############