

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import imutils

img = cv2.imread('1.jpg')
img = cv2.resize(img,None,fx=0.5, fy=0.5,
                        interpolation = cv2.INTER_LINEAR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
filtro = cv2.pyrMeanShiftFiltering(img, 20, 40)
gray = cv2.cvtColor(filtro, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

contornos, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print(len(contornos))
buracos = []
for con in contornos:
  area = cv2.contourArea(con)
  if area < 1000:
    buracos.append(con)
cv2.drawContours(thresh, buracos, -1, 255, -1)

dist = ndi.distance_transform_edt(thresh)
dist_visual = dist.copy()
dist_visual = cv2.normalize(src=dist_visual, dst=None, alpha=0, beta=255,
                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

local_max = peak_local_max(dist, indices=False, min_distance=20, labels=thresh)

markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]

labels = watershed(-dist, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

from skimage import exposure
img_watershed  = exposure.rescale_intensity(labels, out_range=(0, 255))
img_watershed  = np.uint8(img_watershed )
img_watershed  = cv2.applyColorMap(img_watershed , cv2.COLORMAP_JET)

# loop over the unique labels returned by the Watershed
# algorithm
for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue

	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255

	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)

	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 2)
	cv2.putText(img, "#{}".format(label), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


cv2.imwrite('images/output.png',img)
cv2.imwrite('images/thresh.png',thresh)
cv2.imwrite('images/dist.png',dist_visual)
cv2.imwrite('images/watershed.png',img_watershed)

cv2.imshow('Salida', img)
cv2.imshow('Umbral', thresh)
cv2.imshow('Distancia', dist_visual)
cv2.imshow('watershed',  img_watershed )
cv2.waitKey(0)
cv2.destroyAllWindows()

