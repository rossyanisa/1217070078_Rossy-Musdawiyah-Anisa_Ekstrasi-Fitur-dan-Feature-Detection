# 1217070078, Rossy Musdawiyah Anisa

import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage import data

PATCH_SIZE = 21

# load image
image = data.camera()
grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
grass_patches = [image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE] for loc in grass_locations]

sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
sky_patches = [image[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE] for loc in sky_locations]

# menghitung GLCM
xs_grass = []
ys_grass = []
for patch in grass_patches:
    glcm = graycomatrix(patch, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    xs_grass.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    ys_grass.append(graycoprops(glcm, 'correlation')[0, 0])

xs_sky = []
ys_sky = []
for patch in sky_patches:
    glcm = graycomatrix(patch, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    xs_sky.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    ys_sky.append(graycoprops(glcm, 'correlation')[0, 0])

# Plotting
fig = plt.figure(figsize=(12, 8))

# Tampilkan gambar asli dengan lokasi patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# Plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs_grass, ys_grass, 'go', label='Grass')
ax.plot(xs_sky, ys_sky, 'bo', label='Sky')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# Tampilkan patch grass
for i, patch in enumerate(grass_patches):
    ax = fig.add_subplot(3, len(grass_patches), len(grass_patches) + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax.set_xlabel('Grass %d' % (i + 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

# Tampilkan patch sky
for i, patch in enumerate(sky_patches):
    ax = fig.add_subplot(3, len(sky_patches), 2 * len(sky_patches) + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax.set_xlabel('Sky %d' % (i + 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

# Atur tata letak keseluruhan dan judul
fig.suptitle('Grey Level Co-occurrence Matrix Features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()
