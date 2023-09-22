import os
import torch


# Global device variable
#device = "cpu"  # Change this to "gpu" when needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DetectionConfig:
    # Model configuration
    model_name = "resnet50_2020-07-20"
    max_size = 2048

    # Directories
    base_dir = "artifacts"
    frame_folder = os.path.join("artifacts", "frames")
    bbox_path = os.path.join("artifacts", "bbox")
    bbox_json_path = os.path.join("artifacts", "bbox", "BBox_JSON.json")


class FrameExtraction:
    # frame rate
    fps = 30
    # Directories
    base_dir = "artifacts"
    frame_folder = os.path.join("artifacts", "frames")
    capture_mode = "video"  # or "webcam"
    video_data_path = os.path.join("artifacts", "video","Deewangi_Deewangi.mp4")
#