import cv2
import pytesseract
from matplotlib import pyplot as plt

# Path to Tesseract executable (modify this based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read the original image
image_path = r'C:\Users\Sanjay Yadav\Downloads\sujith.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Default Binary Thresholding
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the original and preprocessed images using matplotlib
plt.subplot(121), plt.imshow(image, 'gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(binary_image, 'gray'), plt.title('Default Binary Thresholding')
plt.show()

# Extract text using Tesseract OCR on the preprocessed image
text = pytesseract.image_to_string(binary_image)

# Print the extracted text
print("Extracted Text:")
print(text)