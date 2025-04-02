# import cv2

# # Read the image
# img = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')

# # Apply unsharp masking
# usm = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (0, 0), 10.0), -0.5, 0)

# # Save the enhanced image
# cv2.imwrite('enhanced_image.jpg', usm)

# # You can optionally display the saved image using an external viewer
# # (e.g., a web browser or an image viewer of your choice)
# print("Enhanced image saved as 'enhanced_image.jpg'")

# import cv2

# # Read the image
# img = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')

# # Apply unsharp masking
# usm = cv2.addWeighted(img, 1.5, cv2.GaussianBlur(img, (0, 0), 10.0), -0.5, 0)

# # Save the result to a file
# cv2.imwrite('unsharp_masking_result.jpg', usm)

# import cv2
# import numpy as np

# # Read the image
# img = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')

# # Increase contrast
# alpha = 1.5  # Contrast control (1.0 means no change)
# beta = 0    # Brightness control (0 means no change)
# enhanced_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# # # Display the result
# # cv2.imshow('Contrast Enhanced', enhanced_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()  # Add this line to close the window and release resources

# # Save the enhanced image
# save_path = r'enhanced_image.jpg'
# cv2.imwrite(save_path, enhanced_img)

# print(f"Enhanced image saved at: {save_path}")

# import cv2
# import numpy as np

# # Read the image
# img = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')

# # Apply a small Gaussian blur
# blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# # Convert the image to grayscale
# gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

# # Apply thresholding
# _, threshold_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

# # # Display the result
# # cv2.imshow('Stamp Filter', threshold_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # Save the result
# save_path = 'stamp_filtered_image.jpg'
# cv2.imwrite(save_path, threshold_img)
# print(f"Stamp-filtered image saved at: {save_path}")


# import cv2

# # Read the image
# img = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')

# # Convert the image to grayscale
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply histogram equalization
# equalized_img = cv2.equalizeHist(gray_img)

# # # Display the original and enhanced images side by side for comparison
# # cv2.imshow('Original Image', gray_img)
# # cv2.imshow('Enhanced Image', equalized_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # Save the enhanced image
# save_path = 'enhanced_image.jpg'
# cv2.imwrite(save_path, equalized_img)
# print(f"Enhanced image saved at: {save_path}")


# import cv2
# import numpy as np

# # Read the image
# img = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')

# # Apply a Gaussian blur
# blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# # Convert the image to grayscale
# gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

# # Apply adaptive thresholding
# threshold_img = cv2.adaptiveThreshold(
#     gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
# )

# # Apply morphological operations (dilation) to enhance characters
# kernel = np.ones((3, 3), np.uint8)
# dilated_img = cv2.dilate(threshold_img, kernel, iterations=1)

# # Save the thresholded image
# save_path = 'optimized_for_ocr.jpg'
# cv2.imwrite(save_path, threshold_img)
# print(f"Optimized image for OCR saved at: {save_path}")

# import cv2
# import numpy as np

# # Read the image
# img = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')

# # Apply Gaussian blur
# blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# # Convert to grayscale
# gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

# # Apply adaptive thresholding
# _, threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Apply morphological operations (closing) to enhance characters
# kernel = np.ones((5, 5), np.uint8)
# closed_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel)

# # # Display the result
# # cv2.imshow('Enhanced Image', closed_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # Save the result
# save_path = 'enhanced_characters.jpg'
# cv2.imwrite(save_path, closed_img)
# print(f"Enhanced image saved at: {save_path}")

# import cv2

# # Read the image
# img = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')

# # Convert to grayscale
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply adaptive histogram equalization
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# enhanced_img = clahe.apply(gray_img)

# # # Display the result
# # cv2.imshow('Enhanced Image', enhanced_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # Save the result
# save_path = 'enhanced_characters.jpg'
# cv2.imwrite(save_path, enhanced_img)
# print(f"Enhanced image saved at: {save_path}")


# import cv2
# import numpy as np

# # Read the image
# img = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')

# # Convert to grayscale
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply adaptive thresholding
# adaptive_threshold_img = cv2.adaptiveThreshold(
#     gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
# )

# # Apply contrast stretching
# min_val, max_val, _, _ = cv2.minMaxLoc(gray_img)
# contrast_stretched_img = np.uint8(
#     255 * (gray_img - min_val) / (max_val - min_val)
# )

# # # Display the results
# # cv2.imshow('Adaptive Thresholding', adaptive_threshold_img)
# # cv2.imshow('Contrast Stretched', contrast_stretched_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # Save the results
# cv2.imwrite('adaptive_threshold.jpg', adaptive_threshold_img)
# cv2.imwrite('contrast_stretched.jpg', contrast_stretched_img)

import cv2

# Read the image
img = cv2.imread(r'C:\Users\Sanjay Yadav\OneDrive\Desktop\BackendModels\images\cropped\cropped_1.0.jpg')

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray_img)

# # Display the results
# cv2.imshow('Original Image', gray_img)
# cv2.imshow('CLAHE', clahe_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save the results
cv2.imwrite('original_image.jpg', gray_img)
cv2.imwrite('clahe_result.jpg', clahe_img)
