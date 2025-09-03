
import PIL
from PIL import Image
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('santali.png', cv.IMREAD_GRAYSCALE)
img = img1
img2=cv.imread('u.png', cv.IMREAD_GRAYSCALE)
img3=img2

img4=cv.imread('b.png', cv.IMREAD_GRAYSCALE)
img5=img4.T
print(len(img))
print(len(img3))
print(len(img5))
print(img.shape[1])
print(img3.shape[1])
print(img5.shape[1])


#assert img is not None, "file could not be read, check with os.path.exists()"

#color = ('b','g','r')
#for i,col in enumerate(color):
   # histr = cv.calcHist(img, [0], None, [256], [0, 256])

    # Plot the histogram
    
hist = []
for row  in img:
    count = 0;
    for cell in row [0:260]:
        if cell <178 :
            count = count + 1;
    hist.append(count);
print(hist)
# histrogram for img 2
hist1 = []
for row in img3:
    count = 0;
    for cell in row [0:260]:
        if cell <120 :
            count = count + 1;
    hist1.append(count);
   
# histrogram for img 4
hist2 = []
for row in img5:
    count = 0;
    for cell in row [0:260]:
        if cell <120 :
            count = count + 1;
    hist2.append(count);

    
plt.figure(figsize=(9, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Grayscale Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')


plt.plot(hist, color='BLUE')
plt.show()

#show img2
plt.figure(figsize=(9, 5))
plt.subplot(1, 2, 1)
plt.imshow(img3,cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Grayscale Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.plot(hist1, color='BLUE')
plt.show()
'''
#show img4
plt.figure(figsize=(9, 5))
plt.subplot(1, 2, 1)
plt.imshow(img5,cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Grayscale Image')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.plot(hist2, color='BLUE')
plt.show()
'''


'''
#plt.subplot(1, 2, 1)
#plt.imshow(img, cmap='gray')
#plt.title('Original Image')
#plt.subplot(1, 2, 2)
#plt.xlim([0, 700])
#plt.plot(histr, color='black')
#plt.xlabel('Intensity')
#plt.ylabel('Frequency')
#plt.title('Image Histogram')
#plt.tight_layout()
plt.show() 


# Flatten the image array
pixels = img .flatten()

# Create histogram
plt.hist(pixels, bins=256, range=(0,256))
plt.figure(figsize=(9, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Grayscale Image')
#plt.title('Grayscale Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()
plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()
'''
