# 1217070078, Rossy Musdawiyah Anisa

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load images
img = cv2.imread('solvay.jpg', 0)
template = cv2.imread('einstein.jpg', 0)

# Check if images are loaded successfully
if img is None:
    print("Error: Could not read image 'solvay.jpg'")
    exit()

if template is None:
    print("Error: Could not read image 'einstein.jpg'")
    exit()

# Ensure template size is not larger than input image size
img_h, img_w = img.shape[:2]
template_h, template_w = template.shape[:2]

if template_h > img_h or template_w > img_w:
    print("Error: Template size is larger than input image size. Please resize the template.")
    exit()

# Make a copy of the grayscale image
img2 = img.copy()

# Resize template if necessary (optional step)
template = cv2.resize(template, (img_w // 2, img_h // 2))

# Get template dimensions
w, h = template.shape[::-1]

# List of all template matching methods
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

# Set figure size for plotting
plt.rcParams["figure.figsize"] = (15, 10)

# Iterate over all template matching methods
for idx, method in enumerate(methods):
    img = img2.copy()

    try:
        # Apply template matching
        res = cv2.matchTemplate(img, template, method)

        # Find minimum and maximum values in the result
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Determine top-left corner of the matched area
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        # Calculate bottom-right corner based on template size
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw rectangle around the detected area
        cv2.rectangle(img, top_left, bottom_right, 255, 2)  # 2 is line thickness

        # Display the template matching result and the detected location
        plt.subplot(2, 3, idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title('Method: {}'.format(method))
        plt.xticks([])
        plt.yticks([])

    except cv2.error as e:
        print(f"OpenCV error with method {method}: {e}")

    except Exception as e:
        print(f"Error with method {method}: {e}")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()