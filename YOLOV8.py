from ultralytics import YOLO
from PIL import Image
# Use attempt_load from YOLOv8 to load the model
model = YOLO('C:/Users/Sanjay Yadav/Downloads/Yolov8.pt')

# Set the model to inference mode
model.model.eval()

print("YOLOv8 model loaded successfully!")

# Use the predict method to perform inference
results = model.predict(source=r"C:\Users\Sanjay Yadav\Downloads\WhatsApp Image 2023-12-20 at 12.49.36 PM.jpeg")
result = results[0]  # Assuming you have a single result in the list

# Process and display the results
print(len(result.boxes))

for box in result.boxes:
    label = result.names[box.cls[0].item()]
    cords = [round(x) for x in box.xyxy[0].tolist()]
    prob = box.conf[0].item()
    print("object type:", label)
    print("coordinates", cords)
    print("probability", prob)

# Display the image with bounding boxes
image_with_boxes = Image.fromarray(result.plot()[:, :, ::-1])
image_with_boxes.show()
