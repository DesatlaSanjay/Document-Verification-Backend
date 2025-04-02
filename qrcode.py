# import qrcode
# import cv2

# # Function to generate a QR code
# def generate_qr_code(data, filename):
#     qr = qrcode.QRCode(
#         version=1,
#         error_correction=qrcode.constants.ERROR_CORRECT_L,
#         box_size=10,
#         border=4,
#     )
#     qr.add_data(data)
#     qr.make(fit=True)

#     img = qr.make_image(fill_color="black", back_color="white")
#     img.save(filename)

# # Function to read data from a QR code image
# def read_qr_code(filename):
#     image = cv2.imread(filename)
#     detector = cv2.QRCodeDetector()
#     data, points, qr_code = detector.detectAndDecode(image)
#     return data

# # # Example usage
# # data_to_encode = "Hello, this is a QR code generated using Python!"
# qr_code_filename = r"C:\Users\Sanjay Yadav\Downloads\ec731729-1a76-4015-ad8a-5a7930391be61.jpg"

# # # Generate QR code
# # generate_qr_code(data_to_encode, qr_code_filename)
# # print(f"QR code generated and saved as {qr_code_filename}")

# # Read data from QR code
# decoded_data = read_qr_code(qr_code_filename)
# print(f"Decoded data from QR code: {decoded_data}")

import cv2
from pyzbar.pyzbar import decode
from flask import Flask
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
def extract_qr_code_data(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use PyZbar to decode the QR code
    decoded_objects = decode(gray_image)

    # List to store QR code data
    qr_code_data_list = []

    # Process each decoded QR code
    for decoded_object in decoded_objects:
        data = decoded_object.data.decode('utf-8')
        qr_code_data_list.append(data)

    return qr_code_data_list

if __name__ == "__main__":
    # Specify the path to the QR image
    qr_image_path = r"C:\Users\Sanjay Yadav\Downloads\ec731729-1a76-4015-ad8a-5a7930391be61.jpg"

    # Call the function to extract QR code data
    extracted_data = extract_qr_code_data(qr_image_path)

    # Print the extracted QR code data
    print("Extracted QR Code Data:")
    for data in extracted_data:
        print(data)