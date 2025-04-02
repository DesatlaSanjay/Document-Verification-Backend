import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=r"C:\Users\Sanjay Yadav\Downloads\best (1).pt")

# Set the model to inference mode
model.eval()

# Specify the path to the image you want to perform inference on
image_path = r"C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\temp.jpg"
 
# Perform inference on the image
results = model(image_path)

# Extract bounding box coordinates (xmin, ymin, xmax, ymax) from results.xyxy tensor
boxes = results.xyxy[0].cpu().numpy()

# Open the original image
image = Image.open(image_path)

# Iterate over each bounding box and save the cropped region
for i, box in enumerate(boxes):
    xmin, ymin, xmax, ymax = box[:4]
    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    cropped_image.save(Path(f'C:/Users/Sanjay Yadav/OneDrive/Desktop/BackendModels/images/cropped/cropped_{i}.jpg'))

# Display the results
results.show()

# # Save the results to a folder
# results.save(Path(f'C:/Users/moses/OneDrive/Desktop/DocumentVerification/Results'))





from PIL import Image
import pytesseract
image_path = r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_2.jpg'
with open(image_path, 'rb') as image_file:
    content = image_file.read()
image = Image.open(image_path)
text = pytesseract.image_to_string(image)
print(text)


# Aadhaar number using regex. 
import re

# Function to validate Aadhaar number. 
def isValidAadhaarNumber(str):

	# Regex to check valid 
	# Aadhaar number. 
	regex = ("^[2-9]{1}[0-9]{3}\\" +
			"s[0-9]{4}\\s[0-9]{4}$")
	
	# Compile the ReGex
	p = re.compile(regex)

	# If the string is empty 
	# return false
	if (str == None):
		return False

	# Return if the string 
	# matched the ReGex
	if(re.search(p, str)):
		return True
	else:
		return False

print(isValidAadhaarNumber(text))



