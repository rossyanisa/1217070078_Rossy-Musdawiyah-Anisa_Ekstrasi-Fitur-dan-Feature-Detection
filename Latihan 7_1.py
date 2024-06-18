# 1217070078, Rossy Musdawiyah Anisa

import cv2
from matplotlib import pyplot as plt

# Panggil dan konversi warna agar sesuai dengan Matplotlib
sawit = cv2.imread('sawitt.jpg')
sawit = cv2.cvtColor(sawit, cv2.COLOR_BGR2RGB)

# Panggil dan konversi warna agar sesuai dengan Matplotlib
kebun_sawit = cv2.imread('kebunsawit.jpg')
kebun_sawit = cv2.cvtColor(kebun_sawit, cv2.COLOR_BGR2RGB)

# Atur ukuran gambar keseluruhan
plt.figure(figsize=(12, 6))

# Tampilkan kedua gambar dalam subplot yang benar
plt.subplot(121)
plt.imshow(sawit)
plt.title('Pohon Sawit')
plt.xticks([]), plt.yticks([])  # Hilangkan label sumbu

plt.subplot(122)
plt.imshow(kebun_sawit)
plt.title('Kebun Sawit')
plt.xticks([]), plt.yticks([])  # Hilangkan label sumbu

plt.tight_layout()  # Atur layout subplot agar rapi
plt.show()