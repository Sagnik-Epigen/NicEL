import streamlit as st
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import skimage
from skimage import measure, color, io
import scipy
import pandas as pd
from pandas import ExcelWriter
from sklearn.cluster import KMeans
from streamlit_yellowbrick import st_yellowbrick
import yaml
import json

#Helper Functions
def bgr2rgb(img):
    new_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    return new_img

def bgr2hsv(img):
    new_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    return new_img

def hsv2bgr(img):
    new_img = cv.cvtColor(img,cv.COLOR_HSV2BGR)
    return new_img

def hsvMask(img,low_set,high_set):
    mask = cv.inRange(img,low_set,high_set)
    return mask

def rgb2bgr(img):
    new_img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
    return new_img
    
def holeClose(img,kernel_mat,iter_num):
    new_image = cv.morphologyEx(img,cv.MORPH_OPEN,kernel_mat,iterations=iter_num)
    return new_image
 
def dilate(img,kernel_mat,iter_num):
    new_img = cv.dilate(img,kernel=kernel_mat,iterations=iter_num)
    return new_img

def distTrans(img,mask_size):
    dist = cv.distanceTransform(img,cv.DIST_L2,mask_size)
    return dist

def noiseCleanup(img,b_noise,g_noise,r_noise):
    b,g,r = cv.split(img)
    _,b_mask = cv.threshold(b,b_noise,255,cv.THRESH_TOZERO)
    _,g_mask = cv.threshold(g,g_noise,255,cv.THRESH_TOZERO)
    _,r_mask = cv.threshold(r,r_noise,255,cv.THRESH_TOZERO)
    b_new = cv.bitwise_and(b,b,mask=b_mask)
    g_new = cv.bitwise_and(g,g,mask=g_mask)
    r_new = cv.bitwise_and(r,r,mask=r_mask)
    new_img = cv.merge([b_new,g_new,r_new])
    return new_img

def IoU(img_1,img_2):
    intersection = cv.bitwise_and(img_1,img_2)
    union = cv.bitwise_or(img_1,img_2)
    inter_area = np.sum(intersection==255)
    union_area = np.sum(union==255)
    return inter_area/union_area

st.title("Test Image")

with open('parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)
    
img_path = "/home/devesh/cell-segmentation/images/15.tif"
img = cv.imread(img_path)
st.text("Raw Image:")
sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
img_size.image(img,channels="BGR")

b_thres,g_thres,r_thres = parameters["Noise BGR Thresh"]

clean_img = noiseCleanup(img,b_thres,g_thres,r_thres)
st.text("De Noised Image:")
sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
img_size.image(clean_img,channels="BGR")

low_green = np.array(parameters["Low Green HSV"])
high_green = np.array(parameters["High Green HSV"])

low_blue = np.array(parameters["Low Blue HSV"])
high_blue = np.array(parameters["High Blue HSV"])

low_red = np.array(parameters["Low Red HSV"])
high_red = np.array(parameters["High Red HSV"])

img_hsv = bgr2hsv(clean_img)

tab1,tab2,tab3 = st.tabs(["Blue","Green","Red"])
with tab1:
    blue_mask = hsvMask(img_hsv,low_blue,high_blue)
    st.text("Blue Mask:")
    sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
    img_size.image(blue_mask,use_column_width=True)

with tab2:
    green_mask = hsvMask(img_hsv,low_green,high_green)
    st.text("Green Mask:")
    sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
    img_size.image(green_mask,use_column_width=True)

with tab3:    
    red_mask = hsvMask(img_hsv,low_red,high_red)
    st.text("Red Mask:")
    sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
    img_size.image(red_mask,use_column_width=True)

full_mask = cv.bitwise_or(green_mask,blue_mask,red_mask)
#st.image(red_mask)
st.text("Full Mask:")
st.image(full_mask)

kernel_matrix = np.ones((5,5),np.uint8)
d_iterations = parameters["Image Dilation Iter"]
dilated = dilate(full_mask,kernel_matrix,d_iterations)
st.text("Dilated Image:")
sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
img_size.image(dilated)
dilated_b = dilate(blue_mask,kernel_matrix,d_iterations)
dilated_g = dilate(green_mask,kernel_matrix,d_iterations)
dilated_r = dilate(red_mask,kernel_matrix,d_iterations)

kernel_matrix = np.ones((7,7),np.uint8)
h_iterations =  parameters["Hole Closing Iter"]
holeClosed = holeClose(dilated,kernel_matrix,h_iterations)
st.text("Hole Closed Image:")
sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
img_size.image(holeClosed)
holeClosed_b = holeClose(dilated_b,kernel_matrix,h_iterations)
holeClosed_g = holeClose(dilated_g,kernel_matrix,h_iterations)
holeClosed_r = holeClose(dilated_r,kernel_matrix,h_iterations)

st.text("Blue and Green IoU:")
st.write(IoU(holeClosed_b,holeClosed_g))
st.text("Blue and Red IoU:")
st.write(IoU(holeClosed_b,holeClosed_r))
st.text("Red and Green IoU:")
st.write(IoU(holeClosed_r,holeClosed_g))
IoU_bg = IoU(holeClosed_b,holeClosed_g)
IoU_br = IoU(holeClosed_b,holeClosed_r)
IoU_rg = IoU(holeClosed_r,holeClosed_g)
IoUs = pd.DataFrame({'IoU_bg':[IoU_bg],'IoU_br':[IoU_br],'IoU_rg':[IoU_rg]})

mask_size = parameters["Distance Transform Mask Size"]
dist = distTrans(holeClosed,mask_size)
#cv.imwrite("/home/devesh/cell-segmentation/images/dist.jpg",dist)
st.text("Distance Transformed Applied:")
sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
fig = plt.figure()
plt.imshow(dist,cmap="gray")
plt.axis("off")
img_size.pyplot(fig)#("/home/devesh/cell-segmentation/images/dist.jpg")
plt.close(fig)
#st.image(cv.convertScaleAbs(dist))

max_size = parameters["Distance Thresholding Max Size"]
_,sure_fg = cv.threshold(dist,max_size*dist.max(),255,0)
st.text("Sure Foreground:")
sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
img_size.image(cv.convertScaleAbs(sure_fg))

propList = ['Area',
            'intensity_mean',
            'intensity_max',
            'intensity_min',
            'centroid_local']

sure_fg = cv.convertScaleAbs(sure_fg)
etc,markers = cv.connectedComponents(sure_fg)
markers = markers+10
watershed = cv.watershed(img,markers)
img[markers == -1] = [0,255,255]
img_res = color.label2rgb(markers, bg_label=0)
sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
img_size.image(img_res)
regions = measure.regionprops_table(markers, intensity_image=img,properties=propList)
regions = pd.DataFrame(regions)
#regions = regions.drop(columns=["intensity_mean-2","intensity_max-2","intensity_min-2"])
st.text("Cell Data:")
st.write(regions)

watershed_set = set(watershed.flatten())
watershed_set.remove(-1)
individual_masks = []
corr_bg = []
for i in watershed_set:
    bool_mask = (watershed == i)
    int_mask = bool_mask.astype(np.uint8)
    int_mask = 255*int_mask
    individual_masks.append(int_mask)
    onlyCell = cv.bitwise_and(clean_img,clean_img,mask=int_mask)
    onlyCellb,onlyCellg,onlyCellr = cv.split(onlyCell) 
    b = onlyCellb.flatten()
    g = onlyCellg.flatten()
    r = onlyCellr.flatten()
    onlyCell = pd.DataFrame({'blue':b,'green':g,'red':r})
    corr = onlyCell.corr(method='pearson')
    corr_bg.append(corr['blue']['green'])
    
overlap_coeff = pd.DataFrame({'cell_bg':corr_bg})
st.write(overlap_coeff)

intensities = regions.iloc[:,1:4]
st.text("Correlation between intensities found in Cell Regions:")
fig = plt.figure()
sns.heatmap(data=intensities.corr(method="pearson"),annot=True)
sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
img_size.pyplot(fig)
plt.close(fig)
pc_intensities = pd.DataFrame(intensities.corr(method="pearson"))

b,g,r = cv.split(clean_img)
b = b.flatten()
g = g.flatten()
r = r.flatten()
tot_img = pd.DataFrame({'blue':b,'green':g,'red':r})
st.text("Correlation between color channels of the original image:")
fig = plt.figure()
sns.heatmap(data=tot_img.corr(method="pearson"),annot=True)
sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
img_size.pyplot(fig)
plt.close(fig)
pc_raw = pd.DataFrame(tot_img.corr(method="pearson"))

optimal_k_val = parameters["Optimal K"]
kmeans = KMeans(n_clusters=optimal_k_val).fit(intensities)
centroids = kmeans.cluster_centers_
st.text("The Centroids:")
st.write(centroids)
centroids = pd.DataFrame(centroids)

fig,ax = plt.subplots()
sns.kdeplot(data=tot_img["blue"],color="blue",fill=True)
sns.kdeplot(data=tot_img["green"],color="green",fill=True)
sns.kdeplot(data=tot_img["red"],color="red",fill=True)
st.text("Histogram Analysis")
ax.set_xlim(-10,30)
sideL,img_size,sideR = st.columns([0.2, 3, 0.2])
img_size.pyplot(fig)
plt.close(fig)

path = "/home/devesh/cell-segmentation/test_image_data.xlsx"
with ExcelWriter(path) as writer:
    regions.to_excel(writer, sheet_name='Cell Data')
    centroids.to_excel(writer, sheet_name='Centroid Data')
    pc_intensities.to_excel(writer,sheet_name='Cell Intensity Correlation') 
    pc_raw.to_excel(writer,sheet_name="Channel Intensity Correlation")
    IoUs.to_excel(writer,sheet_name="IoUs between different Channels")
    overlap_coeff.to_excel(writer,sheet_name="Overlap Coefficient")
#writer.save()
