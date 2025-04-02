from flask import Flask, jsonify, request, send_from_directory
from pdf2image import convert_from_path
import fitz
from PIL import Image
from pathlib import Path
import torch
from flask_cors import CORS
import re
import os
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import easyocr
from datetime import datetime
from difflib import SequenceMatcher
import warnings

# Import YOLOv8
# from yolov8.models.experimental import attempt_load
# from yolov8.utils.general import non_max_suppression, scale_coords
from ultralytics import YOLO

# Use attempt_load from YOLOv8 to load the model
model=YOLO('C:/Users/Sanjay Yadav/Downloads/Yolov8.pt');
# model = attempt_load('C:/Users/Sanjay Yadav/Downloads/Yolov8.pt"', map_location=torch.device('cuda'))
print("YOLOv8 model loaded successfully!")
# model.model.eval()
# Print information about the model
model.info(verbose=True)
# model.summary()


results = model.predict(r"C:\Users\Sanjay Yadav\Downloads\27.jpg")
result = results[0]
print(len(result.boxes))
# box = result.boxes[0]print("Object type:", box.cls)
# print("Coordinates:", box.xyxy)
# print("Probability:", box.conf)
for box in result.boxes:
    label=result.names[box.cls[0].item()]
    cords =[round(x) for x in box.xyxy[0].tolist()]
    prob = box.conf[0].item()
    print("object type:", label)
    print("coordinates",cords)
    print("probability", prob)

# Display the image with bounding boxes
image_with_boxes = Image.fromarray(result.plot()[:, :, ::-1])
image_with_boxes.show()