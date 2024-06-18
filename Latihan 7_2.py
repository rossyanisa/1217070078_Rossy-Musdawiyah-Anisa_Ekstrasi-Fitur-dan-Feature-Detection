# 1217070078, Rossy Musdawiyah Anisa

import cv2
import numpy as np

# Membaca gambar utuh untuk dicari
img_rgb = cv2.imread('kebunsawit.jpg')
if img_rgb is None:
    raise FileNotFoundError("Gambar 'kebunsawit.jpg' tidak ditemukan.")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Membaca template sebagai grayscale
template = cv2.imread('sawitt.jpg', 0)
if template is None:
    raise FileNotFoundError("Gambar 'sawitt.jpg' tidak ditemukan.")

# Ukuran template, ukuran ini akan digunakan untuk menggambar kotak
w, h = template.shape[::-1]

# Menggunakan metode COEFF-NORMALIZED
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

# Nilai threshold atau ambang batas deteksi kemiripan titik.
# Lakukan eksperimen dengan merubah nilai ini
threshold = 0.15
loc = np.where(res >= threshold)

# Membuat array kosong untuk menyimpan lokasi-lokasi dari hasil deteksi
lspoint = []
lspoint2 = []
count = 0  # untuk menyimpan jumlah matching yang ditemukan

for pt in zip(*loc[::-1]):
    # Jika sudah ada, skip lokasi tersebut
    if pt[0] not in lspoint and pt[1] not in lspoint2:
        # Gambar persegi warna kuning dengan ketebalan dua poin
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
        for i in range(pt[0] - 9, pt[0] + 9):
            # Tambahkan koordinat x ke list
            lspoint.append(i)
        for k in range(pt[1] - 9, pt[1] + 9):
            # Tambahkan koordinat y ke list
            lspoint2.append(k)
        count += 1
    else:
        continue

print("Jumlah objek ditemukan:", count)

# Simpan gambar hasil deteksi
cv2.imwrite('hasil_deteksi.jpg', img_rgb)

# Tampilkan dengan imshow (untuk lingkungan desktop)
cv2.imshow('Deteksi Objek', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
