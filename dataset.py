import cv2
import os
import glob
import sys
import numpy as np

path = "/path/where/saving/the/new/images" 
files = glob.glob("/path/where/the/masks/are/*.png")
files.sort()
#print(files)

masks = []
images = []

for f in files:
    if "mask" in f:
        masks.append(f)
    else:
        images.append(f)

for image, mask in zip(images, masks):
    img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    new_mask_1 = np.where((img > 0) & (img <= 128), 1, 0)
    new_mask_2 = np.where(img == 255, 2, 0)
    new_mask = new_mask_1 + new_mask_2
    cv2.imwrite(mask, new_mask)
    print(image+" "+mask)
