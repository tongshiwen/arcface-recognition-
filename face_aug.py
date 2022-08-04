#coding= UTF-8
import os
import math
from PIL import Image
import cv2 as cv
from skimage import io
import random
import skimage
import numpy as np
from PIL import Image, ImageDraw
#noise
def addGaussNoise(imgpath, sigma):
    img = cv.imread(imgpath)
    img_height = img.shape[0]
    img_width = img.shape[1]
    img_channels = img.shape[2]
    mean = 0
    gauss = np.random.normal(mean,sigma,(img_height,img_width,img_channels))
    #给图片添加高斯噪声
    noise_img = img + gauss
    #设置图片添加高斯噪声之后的像素值的范围
    noise_img = np.clip(noise_img,a_min=0,a_max=255)
    return noise_img
#blur
def image_blur(image_path, m):
    """
    图像卷积操作：设置卷积核大小，步距
    :param image_path:
    :return:
    """
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    #cv.imshow('input', img)
    # 模糊操作（类似卷积），第二个参数ksize是设置模糊内核大小
    result = cv.blur(img, (m, m))
    return result
#mirror
def mirror(imgpath):
    img = cv.imread(imgpath)
    #tmppath = dir_save + img_name
    mirror_img = cv.flip(img, 1)
    return mirror_img
#mask
def mask(imgpath):
    #img1 = Image.open(impath)
    #img1 = np.asarray(img1)/255.00
    img2 = Image.open(imgpath).convert("RGB") 
    draw = ImageDraw.Draw(img2)
    draw.rectangle((20,20,50,110), fill = (0,0,0))
    #img2 = np.asarray(img2)/255.00
    return img2
#bright
def bright(imgpath,a, b):
    img=cv2.imread(imgpath)
    cv2.imshow('original_img',img)
    rows,cols,channels=img.shape
    dst=img.copy()
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color=img[i,j][c]*a+b
                if color>255:           # 防止像素值越界（0~255）
                    dst[i,j][c]=255
                elif color<0:           # 防止像素值越界（0~255）
                    dst[i,j][c]=0
    return dst
#gamma transfer/gamma_trans(img, 0.5)
def gamma_trans(img,gamma):
	#具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
	gamma_table = [np.power(x/255.0,gamma)*255.0 for x in range(256)]
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
	#实现映射用的是Opencv的查表函数
	return cv2.LUT(img,gamma_table)
g = os.walk("")
#index = 0
for path,d,filelist in g:
    for filename in filelist:
        if filename.endswith('jpg'):
           img_path = os.path.join(path,filename)	
           img_path1 = img_path.split('jpg')[0]
           G1_img = addGaussNoise(img_path, 10)
           G1_img_path = img_path1+ '_G1' + '.jpg'
           cv.imwrite(G1_img_path, G1_img)	
           G2_img = addGaussNoise(img_path, 15)
           G2_img_path = img_path1+'_G2'+'.jpg'
           cv.imwrite(G2_img_path, G2_img)
           B1_img = image_blur(img_path, 3)
           B1_img_path = img_path1+'_B1'+'.jpg'
           cv.imwrite(B1_img_path, B1_img)
           B2_img = image_blur(img_path, 4)
           B2_img_path = img_path1+'_B2'+'.jpg'
           cv.imwrite(B2_img_path, B2_img)
           mir_img = mirror(img_path)
           mir_img_path = img_path1+'_mir'+'.jpg'
           cv.imwrite(mir_img_path, mir_img)
           mask_img = mask(img_path)
           mask_img_path = img_path1+'mask'+'.jpg'
           mask_img.save(mask_img_path)
           #index = index+1
           #print (index)
           #if index == 2:
           #   break;
         


