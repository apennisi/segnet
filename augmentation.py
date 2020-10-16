import cv2
import os
import glob
import sys
import numpy as np

files = glob.glob("/path/where/the/images/are/*.png")
files.sort()

masks = []
images = []

for f in files:
    if "mask" in f:
        masks.append(f)
    else:
        images.append(f)

counter_rgb = 0
counter_mask = 0

folder = "/path/where/saving/the/images/"

for image, mask in zip(images, masks):
     img = cv2.imread(image)
     flip_vertical = cv2.flip(img, 0)
     flip_horizontal = cv2.flip(img, 1)
     flip_both = cv2.flip(img, -1)

     cv2.imwrite(folder+"rgb_"+str(counter_rgb)+".png", img)
     counter_rgb += 1
     cv2.imwrite(folder+"rgb_"+str(counter_rgb)+".png", flip_vertical)
     counter_rgb += 1
     cv2.imwrite(folder+"rgb_"+str(counter_rgb)+".png", flip_horizontal)
     counter_rgb += 1
     cv2.imwrite(folder+"rgb_"+str(counter_rgb)+".png", flip_both)
     counter_rgb += 1

     img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
     flip_vertical = cv2.flip(img, 0)
     flip_horizontal = cv2.flip(img, 1)
     flip_both = cv2.flip(img, -1)

     cv2.imwrite(folder+"mask_"+str(counter_mask)+".png", img)
     counter_mask += 1
     cv2.imwrite(folder+"mask_"+str(counter_mask)+".png", flip_vertical)
     counter_mask += 1
     cv2.imwrite(folder+"mask_"+str(counter_mask)+".png", flip_horizontal)
     counter_mask += 1
     cv2.imwrite(folder+"mask_"+str(counter_mask)+".png", flip_both)
     counter_mask += 1
