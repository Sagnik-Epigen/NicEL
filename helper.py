import cv2 as cv
import numpy as np
import skimage
import yaml
import matplotlib.pyplot as plt
from cellpose.io import imread
from cellpose import models, io

def image_disp(img,colormap="gray"):
    plt.imshow(img,cmap=colormap)
    plt.autoscale(False)
    plt.axis("off")
    plt.show()

def bgr2rgb(img):
    new_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    
    return new_img

def bgr2hsv(img):
    new_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    
    return new_img

def rgb2bgr(img):
    new_img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
    
    return new_img

def toGray(img):    
    new_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    return new_img

def removeZeros(l):
    return l[l != 0]

def rgb2hsv(img):   
    new_img = cv.cvtColor(rgb2bgr(img),cv.COLOR_BGR2HSV)
    
    return new_img

def hsv2bgr(img):
    new_img = cv.cvtColor(img,cv.COLOR_HSV2BGR)
    
    return new_img

def hsvMask(img,low_set,high_set):
    mask = cv.inRange(img,np.array(low_set),np.array(high_set))
    
    return mask

def RGBNoiseCleanup(img,b_noise,g_noise,r_noise):
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

def overlap(green,blue):
    green_area = np.sum(green==255)
    blue_area = np.sum(blue==255)
    union = cv.bitwise_or(green,blue)
    union_area = np.sum(union==255)
    
    return green_area/union_area

def maskedOP(img,mask):
    mask = mask.astype(np.uint8)
    new_img = cv.bitwise_and(img,img,mask=mask)
    
    return new_img

def boundaries(mask):
    return skimage.segmentation.find_boundaries(mask,mode="thick")


def colorMask(img,parameters,color):
    lower = "Low "+color+" HSV"
    higher = "High "+color+" HSV"
    mask = hsvMask(img,parameters[lower],parameters[higher])
    
    return mask

def holeClose(img,kernel_mat,iter_num):
    new_image = cv.morphologyEx(img,cv.MORPH_OPEN,kernel_mat,iterations=iter_num)
    
    return new_image
 
def dilate(img,kernel_mat,iter_num):
    new_img = cv.dilate(img,kernel=kernel_mat,iterations=iter_num)
    
    return new_img

def erode(img,kernel_mat,iter_num):
    new_img = cv.erode(img,kernel=kernel_mat,iterations=iter_num)
    return new_img

def distTrans(img,mask_size):
    dist = cv.distanceTransform(img,cv.DIST_L2,mask_size)
    
    return dist

def hues(img,wholeMask,greenMask,blueMask):
    newImg = maskedOP(img,wholeMask)
    onlyBlue = maskedOP(newImg,blueMask)
    onlyGreen = maskedOP(newImg,greenMask)
    greenHues = onlyGreen[:,:,0].flatten()
    greenHues = greenHues[greenHues != 0]
    blueHues = onlyBlue[:,:,0].flatten()
    blueHues = blueHues[blueHues != 0]

    return greenHues,blueHues
    
def separator(cellMask):
    the_masks = set(cellMask.flatten())
    the_masks.remove(0)
    singular_masks = [cellMask == i for i in the_masks]
    
    return singular_masks

def imgResizer(nr,nc):
    def resizer(img):
        return cv.resize(img,(nc,nr))
    return resizer

def grayNoiseCleanup(img,thres,type=cv.THRESH_OTSU): # type=cv.THRESH_TOZERO
    grayImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _,clearer = cv.threshold(grayImg,thres,255,type)
    return clearer

def maskArea(mask):
    return np.sum(mask==255)
    
def xyPlotter(mask):
    xyCollection = []
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if mask[i,j]==255:
                xyCollection.append((i,j))

'''
class individualCell():
    def __init__(self,img,cellMask,parameters):
        self.individualMasks = separator(cellMask)
        self.img = img
        self.newImgs = [maskedOP(self.img,i) for i in self.individualMasks]
        self.blueMasks = [colorMask(i,parameters,"Blue") for i in self.newImgs]
        self.greenMasks = [colorMask(i,parameters,"Green") for i in self.newImgs]
        self.hues()
        self.OCI = [OCI(green,blue) for green,blue in zip(self.greenMasks,self.blueMasks)]
        self.IoU = [IoU(green,blue) for green,blue in zip(self.greenMasks,self.blueMasks)]
        
    def hues(self):
        #bgrIMG = [rgb2bgr(img) for img in self.newImgs]
        #hsvIMG = [bgr2hsv(img) for img in bgrIMG]
        onlyBlues = [maskedOP(img,blueMask) for img,blueMask in zip(self.newImgs,self.blueMasks)]
        onlyGreens = [maskedOP(img,greenMask) for img,greenMask in zip(self.newImgs,self.greenMasks)]
        self.greenHues = [onlyGreen[:,:,0].flatten() for onlyGreen in onlyGreens]
        self.greenHues = [greenHue[greenHue != 0] for greenHue in self.greenHues]
        self.blueHues = [onlyBlue[:,:,0].flatten() for onlyBlue in onlyBlues]
        self.blueHues = [blueHue[blueHue != 0] for blueHue in self.blueHues]
        
class imgAnalysis():
    
    def __init__(self,img_path,parameters_path,model):
        self.rawImg = imread(img_path)
        self.Img = bgr2hsv(rgb2bgr(self.rawImg))
        self.parameters = yamlReader(parameters_path)
        self.cellMask,_,_,_ = model.eval(self.rawImg, diameter=None, channels=[2,3])
        self.wholeMasker()
        self.Img = maskedOP(self.Img,self.wholeMask)
        self.boundaries = boundaries(self.wholeMask)
        self.blueMask = colorMask(self.Img,self.parameters,"Blue")
        self.greenMask = colorMask(self.Img,self.parameters,"Green")
        self.greenHues,self.blueHues = hues(self.Img,self.wholeMask,self.greenMask,self.blueMask)
        self.BGFG()
        self.OCI = OCI(self.greenMask,self.blueMask)
        self.IOU = IoU(self.greenMask,self.blueMask)
        self.individual = individualCell(self.Img,self.cellMask,self.parameters)
        
    def wholeMasker(self):
        self.wholeMask = self.cellMask.copy()
        self.wholeMask[self.wholeMask != 0] = 255
        self.wholeMask = self.wholeMask.astype(np.uint8)
        
    def BGFG(self):
        self.BG = np.sum(self.wholeMask == 0)
        self.FG = np.sum(self.wholeMask == 255)
'''