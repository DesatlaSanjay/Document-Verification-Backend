# # # # import cv2
# # # # import numpy as np
# # # # img=cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')
# # # # img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
# # # # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # # # kernel = np.ones((1, 1), np.uint8)
# # # # img = cv2.dilate(img, kernel, iterations=1)
# # # # img = cv2.erode(img, kernel, iterations=1)
# # # # cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# # # # cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# # # # cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# # # # cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

# # # # cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

# # # # cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

# # # # cv2.imwrite('filter2.jpg',img)


# # # import cv2
# # # import numpy as np

# # # # Read the image
# # # img = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')

# # # # Resize the image
# # # img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

# # # # Convert to grayscale
# # # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # # Apply dilation and erosion
# # # kernel = np.ones((1, 1), np.uint8)
# # # dilated_img = cv2.dilate(gray_img, kernel, iterations=1)
# # # eroded_img = cv2.erode(dilated_img, kernel, iterations=1)

# # # # Thresholding with different methods
# # # binary_thresh = cv2.threshold(cv2.GaussianBlur(eroded_img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# # # adaptive_thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(eroded_img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

# # # # Save the results
# # # cv2.imwrite('binary_threshold.jpg', binary_thresh)
# # # cv2.imwrite('adaptive_threshold.jpg', adaptive_thresh)


# # import cv2
# # import numpy as np

# # # Read the image
# # img = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')

# # # Resize the image
# # img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

# # # Convert to grayscale
# # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # Apply bilateral filtering
# # bilateral_filtered = cv2.bilateralFilter(gray_img, 9, 75, 75)

# # # Apply morphological operations (dilation and erosion)
# # kernel = np.ones((5, 5), np.uint8)
# # morphology_result = cv2.morphologyEx(bilateral_filtered, cv2.MORPH_CLOSE, kernel)

# # # Apply adaptive thresholding
# # adaptive_thresh = cv2.adaptiveThreshold(morphology_result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

# # # # Display the result
# # # cv2.imshow('Enhanced Image', adaptive_thresh)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()

# # # Save the result
# # cv2.imwrite('enhanced_image.jpg', adaptive_thresh)
# # print("Enhanced image saved as 'enhanced_image.jpg'")

# import cv2
# img=cv2.imread(r"C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg")
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img=cv2.resize(img,(560,900))

# _, result=cv2.threshold(img,20,255,cv2.THRESH_BINARY)

# adaptive_result=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,5)

# cv2.imwrite('ADAPT.jpg',adaptive_result)

import cv2
import numpy as np

# Read the original image
img = cv2.imread(r"C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg")
# img = cv2.resize(img, (560, 900))

# # Display the original image
# cv2.imshow('Original Image', img)
# cv2.waitKey(0)

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# Apply adaptive thresholding
adaptive_result = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 5)


# Save the result
cv2.imwrite('ADAPT.jpg', adaptive_result)

#
