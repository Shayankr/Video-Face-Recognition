# Importing necessary libraries:
import cv2
from PIL import Image
import numpy as np
import retinaface
from retinaface.pre_trained_models import get_model
import os
import json


# Initialize the face detection model:
model = get_model("resnet50_2020-07-20", max_size=2048)
model.eval()



# Define a function to perform face detection and save results to a JSON file
def detect_faces_and_save_to_json(image_path, save_bbox_path):
    try:
        # Open the JPEG image using Pillow
        image = Image.open(image_path)

        # Convert the Pillow image to a NumPy array
        image = np.array(image)

        # Perform face detection
        annotation = model.predict_jsons(image)

        # Extract the file name without extension from the image path
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Create a JSON file path for saving the results
        json_file_path = os.path.join(save_bbox_path, "acquisition_1", f"{image_name}_results.json")

        # Save the face detection results to the JSON file
        with open(json_file_path, "w") as json_file:
            json.dump(annotation, json_file)
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")




# Directory containing your images
image_folder = r"C:\Users\EEE\Desktop\Video-Face-Recognition\artifacts\raw_data\acquisition_1"

# Directory to save Bounding-Box:
bbox_path = r"C:\Users\EEE\Desktop\Video-Face-Recognition\artifacts\bbox"

# Iterate through all images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        detect_faces_and_save_to_json(image_path, bbox_path)



################################################################################################################