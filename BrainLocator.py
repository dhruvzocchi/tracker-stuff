import cv2 as cv2
import numpy as np
from scipy import spatial

calibration=False
displayImage=False

minArea=350
maxArea=1200
minAR=0.15
maxAR=0.25
offset=0.35
sz=15
##calibration params
if calibration:
    minArea=50
    maxArea=100000
    minAR=0
    maxAR=1000
    offset=1.0
    sz=25
stride=5
posHist=[[0,0]]*stride


def movingAvg(headCenter):
    posHist.append(headCenter)
    posHist.remove(posHist[0])
    xvals = [pos[0] for pos in posHist]
    yvals = [pos[1] for pos in posHist]
    avgHeadPos= [sum(xvals)/len(xvals),sum(yvals)/len(yvals)]
    return np.array(avgHeadPos)

def displayImg(img,pts,winName,scale):
    image=img.copy()
    for pt in pts:
        cv2.circle(image,(int(pt[0]),int(pt[1])),1,(0,0,0),-1)
    if len(pts)==2:
        cv2.line(image,pts[0],pts[1],(0,0,0),1)
    width=image.shape[1]*scale
    height=image.shape[0]*scale
    image=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)

    cv2.imshow(winName,image)
    cv2.waitKey(1)

def displayParamValues(area, aspectRatio):
    cv2.namedWindow('paramValues',cv2.WINDOW_NORMAL)
    filler=np.zeros((250,250))
    cv2.putText(filler, 'fish area =' + str(area), (10,10), cv2.FONT_HERSHEY_PLAIN,
                1, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(filler, 'aspect ratio =' + str(aspectRatio), (10,50), cv2.FONT_HERSHEY_PLAIN,
                1, (255,255,255), 1, cv2.LINE_AA)
    cv2.imshow('paramValues',filler)
    cv2.waitKey(1)

def findFish(frame):
    fishFound=False
    frame=cv2.GaussianBlur(frame,(7,7),3)
    threshVal,thresholded=cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    centroid=[]
    area=[]
    contour=[]
    aspectRatio=[]
    for contour in contours:
        area = cv2.contourArea(contour)
        rect=cv2.minAreaRect(contour)
        (center,dim,angle)=rect
        if max(dim)==0:
            aspectRatio=0
        else:
            aspectRatio=min(dim)/max(dim)
        if area > minArea and area < maxArea and aspectRatio>minAR and aspectRatio<maxAR:
            mom = cv2.moments(contour)
            centroid=[int(mom["m10"] / mom["m00"]), int(mom["m01"] / mom["m00"])]
            fishFound=True
            return thresholded, centroid, fishFound, area, contour, aspectRatio 
        else:
            centroid=[]
            contour=[]
    return thresholded, centroid, fishFound, area, contour,aspectRatio

def findHead(contour,centroid):
    reshapedContour=np.reshape(contour,(contour.shape[0],2))
    candidates = reshapedContour[spatial.ConvexHull(reshapedContour).vertices]
    dist_mat = spatial.distance_matrix(candidates, candidates)
    i,j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    pt1=candidates[i]
    pt2=candidates[j]
    v1=pt1-centroid
    v2=pt2-centroid
    dist1 = np.linalg.norm(v1)
    dist2 = np.linalg.norm(v2)
    if dist1<dist2:
        top=pt1
        headCenter=pt1-(offset*v1)
    else:
        top=pt2
        headCenter=pt2-(offset*v2)
    return headCenter, top

def refineHeadPos(thresholded,avgHeadPos,sz):
    lower=avgHeadPos-sz
    upper=avgHeadPos+sz
    headImg=thresholded[int(lower[1]):int(upper[1]),int(lower[0]):int(upper[0])]
    headImg_mom=cv2.moments(headImg)
    try:
        headCtrd=[int(headImg_mom["m10"]/headImg_mom["m00"]),int(headImg_mom["m01"]/headImg_mom["m00"])]
    except:
        headCtrd=[0,0]
    return np.array(headCtrd), headImg, lower

def findBrain(frame):
    frame = np.array(frame).astype('uint8')
    thresholded,centroid,fishFound,area,contour,aspectRatio=findFish(frame)
    if fishFound:
        headCenter,top=findHead(contour,centroid)
        avgHeadPos=movingAvg(headCenter)
        if avgHeadPos[0]>sz and avgHeadPos[1]>sz:
            finalHeadPos,headImg,lower=refineHeadPos(thresholded,avgHeadPos,sz)
            if displayImage:
                displayImg(headImg,[finalHeadPos],'cropped head image',5)
        else:
            finalHeadPos=avgHeadPos
            lower=[0,0]
        if displayImage:
            displayImg(thresholded,[top,centroid], 'thresholded image',3)
            displayImg(frame,[avgHeadPos],'original image',3)
    else:
        finalHeadPos=np.array(posHist[-1])
        lower=[0,0]
    return (finalHeadPos+lower).astype('int32')

