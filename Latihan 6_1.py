# 1217070078, Rossy Musdawiyah Anisa

import cv2
from matplotlib import pyplot as plt

# Panggil dan konversi warna agar sesuai dengan Matplotlib
einstein = cv2.imread('einstein.jpg')
einstein = cv2.cvtColor(einstein, cv2.COLOR_BGR2RGB)

# Panggil dan konversi warna agar sesuai dengan Matplotlib
solvay = cv2.imread('solvay.jpg')
solvay = cv2.cvtColor(solvay, cv2.COLOR_BGR2RGB)

# Tampilkan kedua gambar menggunakan subplot
plt.subplot(121)
plt.imshow(einstein)
plt.title('Einstein')
plt.axis('off')  # Tidak menampilkan sumbu
plt.subplot(122)
plt.imshow(solvay)
plt.title('Solvay Conference 1927')
plt.axis('off')  # Tidak menampilkan sumbu
plt.show()
