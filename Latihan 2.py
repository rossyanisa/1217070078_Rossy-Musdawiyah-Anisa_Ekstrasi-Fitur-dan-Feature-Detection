# 1217070078, Rossy Musdawiyah Anisa

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img_bgr = cv2.imread("haezy.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Calculate image dimensions
height, width, channels = img_rgb.shape

# Initialize arrays for histogram calculation
hgr = np.zeros((256))
hgg = np.zeros((256))
hgb = np.zeros((256))
hgrgb = np.zeros((768), dtype=np.int32)

# Function to reset histogram arrays
def makeItZero():
    for x in range(0, 256):
        hgr[x] = 0
        hgg[x] = 0
        hgb[x] = 0
    for x in range(0, 768):
        hgrgb[x] = 0

# Reset histogram arrays
makeItZero()

# Calculate histograms for each channel
for y in range(0, height):
    for x in range(0, width):
        red = int(img_rgb[y][x][0])
        green = int(img_rgb[y][x][1])
        blue = int(img_rgb[y][x][2])

        hgr[red] += 1
        hgg[green] += 1
        hgb[blue] += 1

        red_bin = red
        green_bin = green + 256
        blue_bin = blue + 512

        hgrgb[red_bin] += 1
        hgrgb[green_bin] += 1
        hgrgb[blue_bin] += 1

# Plot histograms for each channel
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(20, 7))
ax1.grid(color='b', linestyle='--', linewidth=0.5, alpha=0.3)
ax2.grid(color='b', linestyle='--', linewidth=0.5, alpha=0.3)
ax3.grid(color='b', linestyle='--', linewidth=0.5, alpha=0.3)

ax1.set_title('Red')
ax1.hist(np.arange(256), bins=256, weights=hgr, color="red", alpha=0.7)
ax1.set_xlim([0, 255])

ax2.set_title('Green')
ax2.hist(np.arange(256), bins=256, weights=hgg, color="green", alpha=0.7)
ax2.set_xlim([0, 255])

ax3.set_title('Blue')
ax3.hist(np.arange(256), bins=256, weights=hgb, color="blue", alpha=0.7)
ax3.set_xlim([0, 255])

plt.show()

# Plot combined RGB histogram
plt.figure(figsize=(20, 7))
plt.plot(hgrgb, color='black')
plt.title("Histogram Red Green Blue")
plt.xlabel("Pixel Value (0-767)")
plt.ylabel("Number of Pixels")
plt.show()