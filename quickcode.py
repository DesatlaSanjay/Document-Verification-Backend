import cv2
# from pyzbar.pyzbar import decode
from ultralytics import YOLO
import os
from pdf2image import convert_from_path
# Load YOLOv8 model
model = YOLO('C:/Users/Sanjay Yadav/Downloads/Yolov8.pt')
print("YOLOv8 model loaded successfully!")

import cv2
from PIL import Image

# def extract_and_decode_qr_code(cropped_qr_code):
#     # Convert the region to grayscale
#     gray_qr_code = cv2.cvtColor(cropped_qr_code, cv2.COLOR_BGR2GRAY)

#     # Use OpenCV to decode the QR code
#     decoded_objects = cv2.QRCodeDetector().detectAndDecode(gray_qr_code)

#     # Process each decoded QR code
#     for data in decoded_objects[0]:
#         print(f"QR Code Data: {data}")

#     # Display the QR code region
#     cv2.imshow('QR Code Region', cropped_qr_code)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# def read_data():
   
#     image = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_qr_code_qr code.jpg')
#     detector = cv2.QRCodeDetector()
#     data, points, qr_code = detector.detectAndDecode(image)
#     return data
# # Function to perform YOLOv8 predictions
def predict_with_yolov8(image_path):
    # Run YOLOv8 prediction
    results = model.predict(image_path)
    result = results[0]

    # Process YOLOv8 results
    for box in result.boxes:
        label = result.names[box.cls[0].item()]
        cords = [round(x) for x in box.xyxy[0].tolist()]
        prob = box.conf[0].item()
        print("Object type:", label)
        print("Coordinates:", cords)
        print("Probability:", prob)

        # If the object type is 'qr', extract and decode the QR code
        if label == 'QR Code':
            # Extract the QR code region based on coordinates
            x, y, w, h = cords
            cropped_qr_code = cv2.imread(image_path)[y:y+h, x:x+w]

            # Save the cropped QR code image in the current directory
            save_path = f'C:\\Users\\Sanjay Yadav\\OneDrive\\Desktop\\BackendModels\\images\\cropped\\cropped_qr_code_{label.lower()}.jpg'
            cv2.imwrite(save_path, cropped_qr_code)

#             # Call the function to extract and decode QR code
#             read_data()

# # Provide the path to your image
# import cv2

# def read_data(image_path):
#     # Load the image
#     image = cv2.imread(image_path)

#     if image is None:
#         print(f"Error: Unable to read the image at {image_path}")
#         return None

#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Use OpenCV to detect and decode the QR code
#     detector = cv2.QRCodeDetector()
#     data, points, qr_code = detector.detectAndDecode(gray_image)

#     if data:
#         print(f"Decoded data from QR code: {data}")
#     else:
#         print("No QR code detected or unable to decode.")

#     return data

# # Example usage
# cropped_qr_code_path = r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_qr_code_qr code.jpg'
# decoded_data = read_data(cropped_qr_code_path)
def qr_code():
    # Read the image

    # Extract region based on bounding box
    # x, y, w, h = bounding_box
    # qr_code_region = image[y:y+h, x:x+w]
    qr_code_region=r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_qr_code_qr code.jpg'
    image = cv2.imread(qr_code_region)
    

    # Convert the region to grayscale
    gray_qr_code = cv2.cvtColor(qr_code_region, cv2.COLOR_BGR2GRAY)

    # Use PyZbar to decode the QR code
    decoded_objects = decode(gray_qr_code)

    # Process each decoded QR code
    for obj in decoded_objects:
        data = obj.data.decode('utf-8')
        print(f"QR Code Data: {data}")

    # Display the QR code region
    # cv2_imshow(qr_code_region)
qr_code()