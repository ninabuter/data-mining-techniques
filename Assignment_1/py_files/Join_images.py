import cv2
import numpy as np

# Read the two images
image1 = cv2.imread('Distribution_activity.png')
image2 = cv2.imread('Distribution_appcatfinance.png')

# Resize images to have the same width (optional, adjust as needed)
width = min(image1.shape[1], image2.shape[1])
height1 = int(image1.shape[0] * width / image1.shape[1])
height2 = int(image2.shape[0] * width / image2.shape[1])
image1 = cv2.resize(image1, (width, height1))
image2 = cv2.resize(image2, (width, height2))

# Vertically concatenate the images
joined_image = np.concatenate((image1, image2), axis=0)

# Display or save the concatenated image
# cv2.imshow('Joined Image', joined_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# To save the joined image:
cv2.imwrite('joined_image.png', joined_image)
