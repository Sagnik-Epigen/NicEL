import streamlit as st
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import skimage
from skimage import measure, color, io
from sklearn.metrics.cluster import adjusted_rand_score as ari
import scipy
import pandas as pd
from pandas import ExcelWriter
from sklearn.cluster import KMeans
from streamlit_yellowbrick import st_yellowbrick
import yaml
from scipy import stats
import umap

st.set_page_config(layout="wide")
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
    
#def segmentation():

st.title("Control Image Parameter Generator")

img_path = "/home/devesh/cell-segmentation/Control.tif"
img = cv.imread(img_path)
st.text("Raw Image:")
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
img_size.image(img,channels="BGR",use_column_width=True)

b_thres = st.slider('Blue Threshold', min_value=0, max_value=256, value=30)
g_thres = st.slider('Green Threshold', min_value=0, max_value=256, value=30)
r_thres = st.slider('Red Threshold', min_value=0, max_value=256, value=30)

clean_img = noiseCleanup(img,b_thres,g_thres,r_thres)
st.text("De Noised Image:")
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
img_size.image(clean_img,channels="BGR",use_column_width=True)
cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/DeNoised.png",clean_img)

tab1,tab2,tab3 = st.tabs(["Blue","Green","Red"])

img_hsv = bgr2hsv(clean_img)

with tab1:
    blue_low_h = int(st.slider("Blue Low Hue",min_value=0,max_value=360,value=190)/2)
    blue_high_h = int(st.slider("Blue High Hue",min_value=0,max_value=360,value=240)/2)
    blue_low_s = int(st.slider("Blue Low Saturation",min_value=0,max_value=100,value=0)/100*255)
    blue_high_s = int(st.slider("Blue High Saturation",min_value=0,max_value=100,value=100)/100*255)
    blue_low_v = int(st.slider("Blue Low Value",min_value=0,max_value=100,value=0)/100*255)
    blue_high_v = int(st.slider("Blue High Value",min_value=0,max_value=100,value=100)/100*255)
    low_blue = np.array([blue_low_h,blue_low_s,blue_low_v])
    high_blue = np.array([blue_high_h,blue_high_s,blue_high_v])
    blue_mask = hsvMask(img_hsv,low_blue,high_blue)
    st.text("Blue Mask:")
    sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
    img_size.image(blue_mask,use_column_width=True)
    cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/blue-hsv.png",blue_mask)

with tab2:
    green_low_h = int(st.slider("Green Low Hue",min_value=0,max_value=360,value=128)/2)
    green_high_h = int(st.slider("Green High Hue",min_value=0,max_value=360,value=160)/2)
    green_low_s = int(st.slider("Green Low Saturation",min_value=0,max_value=100,value=0)/100*255)
    green_high_s = int(st.slider("Green High Saturation",min_value=0,max_value=100,value=100)/100*255)
    green_low_v = int(st.slider("Green Low Value",min_value=0,max_value=100,value=0)/100*255)
    green_high_v = int(st.slider("Green High Value",min_value=0,max_value=100,value=100)/100*255)
    low_green = np.array([green_low_h,green_low_s,green_low_v])
    high_green = np.array([green_high_h,green_high_s,green_high_v])
    green_mask = hsvMask(img_hsv,low_green,high_green)
    st.text("Green Mask:")
    sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
    img_size.image(green_mask,use_column_width=True)
    cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/green-hsv.png",green_mask)
    
with tab3:
    red_low_h = int(st.slider("Red Low Hue",min_value=0,max_value=360,value=190)/2)
    red_high_h = int(st.slider("Red High Hue",min_value=0,max_value=360,value=240)/2)
    red_low_s = int(st.slider("Red Low Saturation",min_value=0,max_value=100,value=0)/100*255)
    red_high_s = int(st.slider("Red High Saturation",min_value=0,max_value=100,value=100)/100*255)
    red_low_v = int(st.slider("Red Low Value",min_value=0,max_value=100,value=0)/100*255)
    red_high_v = int(st.slider("Red High Value",min_value=0,max_value=100,value=100)/100*255)
    low_red = np.array([red_low_h,red_low_s,red_low_v])
    high_red = np.array([red_high_h,red_high_s,red_high_v])
    red_mask = hsvMask(img_hsv,low_red,high_red)
    st.text("Red Mask:")
    sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
    img_size.image(red_mask,use_column_width=True)
    cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/red-hsv.png",red_mask)

full_mask = cv.bitwise_or(green_mask,blue_mask,red_mask)
#st.image(red_mask)
st.text("Full Mask:")
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
img_size.image(full_mask,use_column_width=True)
cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/full-hsv.png",full_mask)

kernel_matrix = np.ones((5,5),np.uint8)
d_iterations = st.number_input('No. of Iterations in the operations',value=3)
dilated = dilate(full_mask,kernel_matrix,d_iterations)
dilated_b = dilate(blue_mask,kernel_matrix,d_iterations)
dilated_g = dilate(green_mask,kernel_matrix,d_iterations)
dilated_r = dilate(red_mask,kernel_matrix,d_iterations)
st.text("Dilated Image:")
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
img_size.image(dilated,use_column_width=True)
cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/full-dilated.png",dilated)
cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/blue-dilated.png",dilated_b)
cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/green-dilated.png",dilated_g)
cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/red-dilated.png",dilated_r)

kernel_matrix = np.ones((7,7),np.uint8)
h_iterations =  st.number_input('No. of Iterations in the operations',value=7)
holeClosed = holeClose(dilated,kernel_matrix,h_iterations)
holeClosed_b = holeClose(dilated_b,kernel_matrix,h_iterations)
holeClosed_g = holeClose(dilated_g,kernel_matrix,h_iterations)
holeClosed_r = holeClose(dilated_r,kernel_matrix,h_iterations)
st.text("Hole Closed Image:")
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
img_size.image(holeClosed,use_column_width=True)
cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/full-holeClosed.png",holeClosed)
cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/blue-holeClosed.png",holeClosed_b)
cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/green-holeClosed.png",holeClosed_g)
cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/red-holeClosed.png",holeClosed_r)

st.text("Blue and Green IoU:")
st.write(IoU(holeClosed_b,holeClosed_g))
st.text("Blue and Red IoU:")
st.write(IoU(holeClosed_b,holeClosed_r))
st.text("Red and Green IoU:")
st.write(IoU(holeClosed_r,holeClosed_g))
IoU_bg = IoU(holeClosed_b,holeClosed_g)
IoU_br = IoU(holeClosed_b,holeClosed_r)
IoU_rg = IoU(holeClosed_r,holeClosed_g)
IoUs = pd.DataFrame({'IoU_bg':IoU_bg,'IoU_br':IoU_br,'IoU_rg':IoU_rg},index=[0])

mask_size = st.selectbox("Mask Size:",options=[0,3,5],index=0)
dist = distTrans(holeClosed,mask_size)
#cv.imwrite("/home/devesh/cell-segmentation/images/dist.jpg",dist)
st.text("Distance Transformed Applied:")
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
fig = plt.figure()
plt.imshow(dist,cmap="gray")
plt.axis("off")
img_size.pyplot(fig)#("/home/devesh/cell-segmentation/images/dist.jpg")
plt.close(fig)
cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/Distance_Transformed.png",dist)
#st.image(cv.convertScaleAbs(dist))

max_size = st.slider("Max Value:", min_value=0.0, max_value=1.0, value=0.3)
_,sure_fg = cv.threshold(dist,max_size*dist.max(),255,0)
st.text("Sure Foreground:")
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
img_size.image(cv.convertScaleAbs(sure_fg))
cv.imwrite("/home/devesh/cell-segmentation/Control_Image_Data/sure_fg.png",sure_fg)
sure_fg = cv.convertScaleAbs(sure_fg)

etc,markers = cv.connectedComponents(sure_fg)
markers = markers+10
watershed = cv.watershed(img,markers)
img[markers == -1] = [0,255,255]
img_res = color.label2rgb(markers, bg_label=0)
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
img_size.image(img_res)

watershed_set = set(watershed.flatten())
watershed_set.remove(-1)
individual_masks = []
corr_bg = []
min_intensities_b = []
mean_intensities_b = []
max_intensities_b = []
min_intensities_g = []
mean_intensities_g = []
max_intensities_g = []
min_intensities_r = []
mean_intensities_r = []
max_intensities_r = []
for i in watershed_set:
    bool_mask = (watershed == i)
    int_mask = bool_mask.astype(np.uint8)
    int_mask = 255*int_mask
    individual_masks.append(int_mask)
    onlyCell = cv.bitwise_and(clean_img,clean_img,mask=int_mask)
    onlyCellb,onlyCellg,onlyCellr = cv.split(onlyCell) 
    b = onlyCellb.flatten()
    g = onlyCellg.flatten()
    #r = onlyCellr.flatten()
    non_zero_b = b[np.where(b!=0)]
    non_zero_g = g[np.where(g!=0)]
    
    #non_zero_r = r[np.where(r!=0)]
    min_intensities_b.append(np.amin(non_zero_b))
    min_intensities_g.append(np.amin(non_zero_g))
    #min_intensities_r.append(np.amin(non_zero_r))
    
    mean_intensities_b.append(np.mean(non_zero_b))
    mean_intensities_g.append(np.mean(non_zero_g))
    #mean_intensities_r.append(np.mean(non_zero_r))
    
    max_intensities_b.append(max(non_zero_b))
    max_intensities_g.append(max(non_zero_g))
    #max_intensities_r.append(max(non_zero_r))
    
    onlyCell = pd.DataFrame({'blue':b,'green':g})#,'red':r})
    corr = onlyCell.corr(method='pearson')
    corr_bg.append(corr['blue']['green'])

cell_info = pd.DataFrame({'corr_bg':corr_bg,'min_intensity_b':min_intensities_b,'min_intensity_g':min_intensities_g,#'min_intensity_r':min_intensities_r,
                          'mean_intensity_b':mean_intensities_b,'mean_intensity_g':mean_intensities_g,#'mean_intensity_r':mean_intensities_r,
                          'max_intensity_b':np.array(max_intensities_b),'max_intensity_g':np.array(max_intensities_g)})#,'max_intensity_r':max_intensities_r })
st.write(cell_info)

propList = ['Area','centroid_local']
regions = measure.regionprops_table(markers,intensity_image=clean_img,properties=propList)
regions = pd.DataFrame(regions)
#regions = regions.drop(columns=["intensity_mean-2","intensity_max-2","intensity_min-2"])

st.text("Region Data:")
st.write(regions)

intensities = cell_info.iloc[:,3:5]
st.text("Correlation between mean intensities found in Cell Regions:")
fig = plt.figure()
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
sns.heatmap(data=intensities.corr(method="pearson"),annot=True)
img_size.pyplot(fig)
plt.close(fig)
pc_intensities = pd.DataFrame(intensities.corr(method="pearson"))

b,g,r = cv.split(img)
b = b.flatten()
g = g.flatten()
r = r.flatten()
tot_img = pd.DataFrame({'blue':b,'green':g,'red':r})
st.text("Correlation between color channels of the original image:")
fig = plt.figure()
sns.heatmap(data=tot_img.corr(method="pearson"),annot=True)
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
img_size.pyplot(fig)
plt.close(fig)
pc_raw = pd.DataFrame(tot_img.corr(method="pearson"))

model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,15))
visualizer.fit(intensities)
st.text("Optimal K calculations")
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
with img_size:
    st_yellowbrick(visualizer)

optimal_k_val = visualizer.elbow_value_
#optimal_k_val = 5
kmeans = KMeans(n_clusters=optimal_k_val).fit(intensities)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
st.text("The Centroids:")
st.write(centroids)
centroids = pd.DataFrame(centroids)
df = cell_info.copy(deep=True)
df['cluster_labels'] = labels
st.text("Cluster Standard Deviation and Count:")
st.write(df.groupby('cluster_labels').agg({"mean_intensity_b":["std"],"mean_intensity_g":["std"],"cluster_labels":["count"]}))

k_checkers = [x for x in range(2,16)]
if optimal_k_val in k_checkers:
    k_checkers.remove(optimal_k_val)

score_compare = []
for k in k_checkers:
    kmeans = KMeans(n_clusters=k).fit(intensities)
    labels_k = kmeans.labels_
    score_compare.append(ari(labels,labels_k))
    
ari_df = pd.DataFrame({"ARI Score":score_compare},index=k_checkers)
st.write(ari_df)
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
fig = plt.figure()
#sns.lineplot(x=ari_df.index, y="ARI Score", data=ari_df)
plt.plot(ari_df.index,ari_df["ARI Score"],linestyle='-', marker='o')
img_size.pyplot(fig)
plt.close(fig)

#reducer = umap.UMAP()
#embedding = reducer.fit_transform(cell_info.iloc[:,3:5])
#st.write(embedding.shape)
#fig = plt.figure()
#for i, col in enumerate(['b', 'g', 'r']):
#    hist = cv.calcHist([img], [i], None, [27], [0, 256])
#    histoplot = plt.plot(hist, color = col)
#    plt.xlim([0, 15])
#    
#st.write(hist)
#st.text("Histogram Analysis:")
#st.pyplot(fig)

fig,ax = plt.subplots()
sns.kdeplot(data=tot_img["blue"],color="blue")
sns.kdeplot(data=tot_img["green"],color="green")
sns.kdeplot(data=tot_img["red"],color="red")
ax.set_xlim(-10,30)
st.text("Histogram Analysis")
sideL,img_size,sideR = st.columns([0.2, 1, 0.2])
img_size.pyplot(fig)
plt.close(fig)


path = "/home/devesh/cell-segmentation/control_image_data.xlsx"
with ExcelWriter(path) as writer:
    cell_info.to_excel(writer,sheet_name="Cell Information")
    regions.to_excel(writer, sheet_name='Cell Region Data')
    centroids.to_excel(writer, sheet_name='Centroid Data')
    pc_intensities.to_excel(writer,sheet_name='Cell Intensity Correlation') 
    pc_raw.to_excel(writer,sheet_name="Channel Intensity Correlation")
    IoUs.to_excel(writer,sheet_name="IoUs between different Channels")
#writer.save()

parametric_data = {"Optimal K":optimal_k_val,"Distance Thresholding Max Size":max_size,"Distance Transform Mask Size":mask_size,"Hole Closing Iter":h_iterations,"Image Dilation Iter":d_iterations,"Low Green HSV":low_green.tolist(),"High Green HSV":high_green.tolist(),"Low Blue HSV":low_blue.tolist(),"High Blue HSV":high_blue.tolist(),"Low Red HSV":low_red.tolist(),"High Red HSV":high_red.tolist(),"Noise BGR Thresh":[b_thres,g_thres,r_thres]}

with open("/home/devesh/cell-segmentation/parameters.yaml", mode="wb") as file:
    yaml.dump(parametric_data, file,encoding="utf-8")