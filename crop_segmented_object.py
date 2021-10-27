import os

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import expand_dims
from skimage import morphology, io, color, exposure, img_as_float, transform


data_path  = 'D:/Dataset/NIH/images/'
def load_image(filename, size=(512,512)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size, grayscale = True)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

def preprocess_for_segmentation(img,im_shape):
    img = transform.resize(img, im_shape)
    img = exposure.equalize_hist(img)
    img = np.expand_dims(img, -1)
    X = np.array(img)
    y = np.array(img)
    X -= X.mean()
    X /= X.std()
    return X

def removal_of_white_text(img):
    ret,image=cv2.threshold(img,250,255,cv2.THRESH_TOZERO_INV)
    return image


def find_cropping_area(xray_image, seg_model):
	# inputs: @xray_image preprocessed dicom image @seg_model The segmentation model
	# returns:@pr prediction of segmentation model
	# @(x1,y1) left top point of the crop area
	# @(x2,y2) bottom right point of the crop area

	im_shape = (512, 512)  # of the segmentation model
	width_scale = 2  # width-scaling factor for resize
	height_scale = 2  # height-scaling factor for resize
	# The scaling factors are later used to find the cropping region for original image size

	img = preprocess_for_segmentation(xray_image, im_shape)  # preprocessing the Xray image for segmentation
	inp_shape = img.shape
	X = np.expand_dims(img, axis=0)
	pred = seg_model.predict(X)[..., 0].reshape(inp_shape[:2])  # predicted segmentation

	# find bounding box around the two lungs
	ret, pr = cv2.threshold(pred, 0.95, 1, cv2.THRESH_BINARY)
	kernel = np.ones((3, 3), np.uint8)
	pr = np.array(pr * 255, dtype=np.uint8)
	pr = cv2.morphologyEx(pr, cv2.MORPH_OPEN, kernel, iterations=3)
	pr_canny = cv2.Canny(pr, 170, 255)
	cnts = cv2.findContours(pr_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cntsSorted = sorted(cnts[0], key=lambda x: cv2.contourArea(x), reverse=True)
	x_c = []
	y_c = []
	b = 0
	for i in range(len(cntsSorted)):
		x, y, w, h = cv2.boundingRect(cntsSorted[i])
		x_c.append(x)
		x_c.append(x + w)
		y_c.append(y)
		y_c.append(y + h)
	w = max(x_c) - min(x_c)
	crp_p1 = (max([min(x_c) - w // 6, 0]), max([min(y_c) - w // 10, 0]))
	crp_p2 = (min([max(x_c) + w // 6, pr.shape[0]]), min([max(y_c) + w // 5, pr.shape[1]]))
	# the crop area is scaled to the original image size
	x1 = int(crp_p1[0] * width_scale)
	y1 = int(crp_p1[1] * height_scale)
	x2 = int(crp_p2[0] * width_scale)
	y2 = int(crp_p2[1] * height_scale)
	pr = cv2.cvtColor(pr, cv2.COLOR_GRAY2RGB)

	# cv2.rectangle(pr, (min(x_c), min(y_c)), (max(x_c), max(y_c)), (255, 0, 0), 1)
	# cv2.rectangle(pr, crp_p1, crp_p2, (0, 255, 0), 2)
	return pr, (x1, y1), (x2, y2)

model = load_model('model_056600.h5')

img_list = os.listdir(data_path)

src_image = load_image(data_path + '00007889_000.png')
img = src_image[0]
print(img.shape)

pr, (x1, y1), (x2, y2) = find_cropping_area(img, model)
src_image = load_image(data_path + '00007889_000.png', size= (1024, 1024))
img = src_image[0]
crop = img[y1:y2, x1:x2]
plt.imshow(crop)
plt.show()


