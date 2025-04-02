import cv2
# from pyzbar.pyzbar import decode
from ultralytics import YOLO

# Load YOLOv8 model
from PIL import Image
model = YOLO('C:/Users/Sanjay Yadav/Downloads/Yolov8.pt')
print("YOLOv8 model loaded successfully!")

# Function to extract and decode QR code


# Function to perform YOLOv8 predictions
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
        if label.lower() == 'qr':
            # Extract the QR code region based on coordinates
            x, y, w, h = cords
            cropped_qr_code = cv2.imread(image_path)[y:y+h, x:x+w]

            # Save the cropped QR code image in the current directory
            save_path = f'cropped_qr_code_{label.lower()}.png'
            # extract_and_decode_qr_code(cropped_qr_code, save_path)

            # Display the image with bounding boxes
            image_with_boxes = Image.fromarray(cropped_qr_code[:, :, ::-1])
            image_with_boxes.show()

# Provide the path to your image
image_path = r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\uploaded.jpg'

# Call the function to perform YOLOv8 predictions and extract QR codes
predict_with_yolov8(image_path)
