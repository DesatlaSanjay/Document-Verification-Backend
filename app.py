from flask import Flask, request, jsonify
from PIL import Image
from pathlib import Path
import torch
import fitz  # PyMuPDF
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# app = Flask(__name__)

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=r"C:\Users\Sanjay Yadav\Downloads\best (1).pt")
print("YOLOv5 model loaded successfully!")
# Set the model to inference mode
model.eval()

def pdf_to_image(pdf_path):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document[0]  # Assuming only one page in the PDF
    image = page.get_pixmap()
    img = Image.frombytes("RGB", [image.width, image.height], image.samples)
    return img

@app.route('/upload', methods=['POST'])
def upload():
    # Handle PDF upload
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF provided'}), 400

    pdf_file = request.files['pdf']

    # Print a message indicating that the PDF is received
    print("PDF received by the backend!")

    # Save the PDF to a temporary file
    pdf_path = 'temp.pdf'
    pdf_image = pdf_to_image(pdf_path)
    pdf_file.save(pdf_image)

    # Convert PDF to an image

    # Perform inference on the image
    results = model(pdf_image)

    # Extract bounding box coordinates (xmin, ymin, xmax, ymax) from results.xyxy tensor
    boxes = results.xyxy[0].cpu().numpy()

    # Save the original image
    pdf_image.save(Path('C:/Users/Sanjay Yadav/OneDrive/Desktop/BackendModels/images'))

    # Iterate over each bounding box and save the cropped region
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box[:4]
        cropped_image = pdf_image.crop((xmin, ymin, xmax, ymax))
        cropped_image.save(Path(f'C:/Users/Sanjay Yadav/OneDrive/Desktop/BackendModels/images/cropped/cropped_{i}.jpg'))

    # Display the results
    results.show()

    # # Save the results to a folder
    # results.save(Path(f'C:/Users/moses/OneDrive/Desktop/DocumentVerification/Results'))

    return jsonify({'message': 'PDF uploaded and processed successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)
