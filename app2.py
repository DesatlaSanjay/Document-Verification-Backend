from flask import Flask, jsonify, request
from pdf2image import convert_from_path
import fitz
from PIL import Image
from pathlib import Path
import torch
from flask_cors import CORS
import pytesseract
import re
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=r"C:\Users\Sanjay Yadav\Downloads\best (1).pt")
print("YOLOv5 model loaded successfully!")
# Set the model to inference mode
model.eval()


# Function to validate Aadhaar number.
def is_valid_aadhaar_number(text):
    # Regex to check valid Aadhaar number.
    regex = ("^[2-9]{1}[0-9]{3}\\" +
             "s[0-9]{4}\\s[0-9]{4}$")

    # Compile the ReGex
    p = re.compile(regex)

    # If the string is empty return false
    if text is None:
        return False
    print('text is ',text)
    # Return if the string matched the ReGex
    return bool(re.search(p, text))


@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        pdf_file = request.files['pdf']

        if pdf_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Convert PDF to image using PyMuPDF (fitz)
        print('PDF received by backend')
        # Save the PDF content to a temporary file
        temp_pdf_path = 'temp.pdf'
        pdf_file.save(temp_pdf_path)



        
        pdf_document = fitz.open(temp_pdf_path)
        first_page = pdf_document[0]
        pix = first_page.get_pixmap()

        # Save the image as temp.jpg
        temp_image_path = 'temp.jpg'
        with open(temp_image_path, 'wb') as img_file:
            img_file.write(pix.tobytes())

        # Open the image using PIL
        pdf_image = Image.open(temp_image_path)
        # Perform inference on the image
        results = model(temp_image_path)
        
        # Extract bounding box coordinates (xmin, ymin, xmax, ymax) from results.xyxy tensor
        boxes = results.xyxy[0].cpu().numpy()

        # # Save the original image
        # pdf_image.save(Path('C:/Users/Sanjay Yadav/OneDrive/Desktop/BackendModels/images/temp.jpg'))

        # Iterate over each bounding box and save the cropped region
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box[:4]
            cropped_image = pdf_image.crop((xmin, ymin, xmax, ymax))
            cropped_image.save(Path(f'C:/Users/Sanjay Yadav/OneDrive/Desktop/BackendModels/images/cropped/cropped_{i}.jpg'))

        # Display the results
        results.show()
        
        # Extract text from the cropped image
        number_path=r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_2.jpg'
        with open(number_path,'rb') as number_file:
            content=number_file.read()
        number=Image.open(number_path)
        text = pytesseract.image_to_string(number)

        # Validate Aadhaar number
        is_valid_aadhaar = is_valid_aadhaar_number(text)


         # Clean up temporary files
        # os.remove(temp_pdf_path)
        os.remove(temp_image_path)

        return jsonify({'message': 'PDF received and converted successfully','is_valid_aadhaar': is_valid_aadhaar}), 200

    except Exception as e:
        # Print or log the exception details
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
