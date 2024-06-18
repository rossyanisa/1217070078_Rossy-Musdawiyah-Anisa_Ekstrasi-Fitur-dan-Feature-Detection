# 1217070078, Rossy Musdawiyah Anisa

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Gunakan gambar
img = cv2.imread('gedung.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deteksi pojok dengan GFTT
corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 10)
corners = np.int0(corners)

# Menampilkan jumlah titik terdeteksi
print("Jumlah titik terdeteksi = ", corners.shape[0])

# Ubah gambar agar sesuai dengan urutan warna RGB untuk matplotlib
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perbesar ukuran hasil plotting
plt.rcParams["figure.figsize"] = (20, 20)

# Untuk tiap pojok yang terdeteksi, gambar lingkaran di atas gambar
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(rgb, (x, y), 3, (255, 0, 0), -1)  # Lingkaran biru dengan radius 3

# Tampilkan gambar dengan pojok terdeteksi
plt.imshow(rgb)
plt.axis('off')  # Matikan penampilan sumbu
plt.show()