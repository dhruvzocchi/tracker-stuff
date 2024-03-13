import cv2 as cv2
import numpy as np
import socket


def determineSubImgs(frame):
    subImgBds=[]
    for i in range (3):
        selFrame=cv2.normalize(frame,None,0,65535,cv2.NORM_MINMAX)
        selFrame=cv2.resize(selFrame,(int(frame.shape[1]*0.5),int(frame.shape[0]*0.5)))
        bds=cv2.selectROI('frame',selFrame)#bds=[x,y,w,h]
        bds=np.array(bds)
        subImgBds.append(bds)
    return subImgBds
def getImgBds(subImgBds):
    # take top left corners of each subimage, but use the width and height of the leftSubImg for all 3 so that the
    # resulting subimages will have the same size for the correlation calculation
    left = np.array(
        [subImgBds[0][1], subImgBds[0][1] + subImgBds[0][3], subImgBds[0][0], subImgBds[0][0] + subImgBds[0][2]])
    right = np.array(
        [subImgBds[1][1], subImgBds[1][1] + subImgBds[0][3], subImgBds[1][0], subImgBds[1][0] + subImgBds[0][2]])
    template = np.array(
        [subImgBds[2][1], subImgBds[2][1] + subImgBds[0][3], subImgBds[2][0], subImgBds[2][0] + subImgBds[0][2]])
    return left,right,template

def setupSocket():
    HOST='localhost'
    PORT=65432
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()
    conn,addr=s.accept()
    return conn,addr,s


### reads current position of the dover stage, takes nReads readings, averages, discards values greater than
###1 std from average, returns the updated average with outliers removed. All this to remove faulty data
##which sometimes comes and causes problems with the stage motion
def filterZList(posList):
    avgPos=np.mean(posList)
    stdPos=np.std(posList)
    for i,val in enumerate(posList):
        if np.abs(avgPos-val)>stdPos:
            del posList[i]
    pos=np.mean(posList)
    return pos

### generates new z position from old position and error in calculated subimg dist vs set point ###
def getZPos(ref,dist,tol,
    currZPos,stepSz):

    err=ref-dist
    if np.abs(err)>tol:
        if err>0:
            newZPos=currZPos+stepSz
        else:
            newZPos=currZPos-stepSz
    else:
        newZPos=currZPos
    return newZPos

### segments fish brain from fluorescent image ###
def findFish(img,minArea,maxArea,kernel,segmentImgBds):
# filter img and threshold using otsu's method, then run morphological openning
    templateImg=cv2.GaussianBlur(img,(3,3),0)
    #templateImg = cv2.GaussianBlur(img.astype('uint8'),(3,3),0)
    _,thresholded=cv2.threshold(templateImg,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded=cv2.morphologyEx(thresholded,cv2.MORPH_OPEN,kernel)
    thresholded=cv2.morphologyEx(thresholded,cv2.MORPH_CLOSE,kernel)
#find centroid of central fish image
    contours,_=cv2.findContours(thresholded,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    centroids=[]
    area=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if area>minArea and area<maxArea:
            mom=cv2.moments(contour)
            cX=int(mom["m10"]/mom["m00"])
            cY=int(mom["m01"]/mom["m00"])
    #centroids is in [row,column] notation- cY=row index, cX=col index
            centroids.append([cY,cX])
    centroids=np.array(centroids)
    if centroids.shape[0]==1:
    ##in this case, centroids will be a (1,2) array within an array, have to extract it, hence centroids[0]
        centroids=centroids[0]#+np.array([segmentImgBds[0],segmentImgBds[2]])
        #print('segmented central fish centroid '+str(centroids))
        fishFound=True
    elif centroids.shape[0]>1:
        fishFound=False
        print('multiple blobs found')
    elif centroids.shape[0]==0:
        fishFound=False
        print('no blobs found')
    return thresholded,centroids,area,fishFound

### runs correlation analysis of template on leftImg, rightImg as defined by the bounds ###
def getDistance(template,leftImg,rightImg,leftSubImgBds,rightSubImgBds):
# do autocorr calc on 3 subregions of full img, find max of each resulting corr matrix, report index in
##coordinates of the original image (full frame from cam)-to do this, have to add onto max index
##the left and top bounds taken for the template matching
    corrMat1=cv2.matchTemplate(leftImg,template,cv2.TM_CCOEFF_NORMED)
    max1=np.argwhere(corrMat1==np.amax(corrMat1))+np.array([leftSubImgBds[0],leftSubImgBds[2]])
    corrMat3=cv2.matchTemplate(rightImg,template,cv2.TM_CCOEFF_NORMED)
    max3=np.argwhere(corrMat3==np.amax(corrMat3))+np.array([rightSubImgBds[0],rightSubImgBds[2]])
# calculate distance between left and right maxima
    if max1.shape==(1,2) and max3.shape==(1,2):
        dist=np.linalg.norm(max3-max1)
    else:
        dist=0
#This is for plotting the max value idxs on the final image. Input imgs (the subimages) and corr mats have different
##sizes (corr mat values only calculated for those pixel positions in subimages where the template completely fits,
##so the idxs calculated above will be offset from their actual locations in the original img (from cam), fix that
##by adding half the size of the template to the calculated idxs
    max1=max1[0]+(0.5*np.array(template.shape))
    max3=max3[0]+(0.5*np.array(template.shape))
    return dist,max1,max3

### main z tracking function ###
def zTracker(frame,
             area,kernelSz,roiSizes,
             segmentImgBds,leftSubImgBds,rightSubImgBds,
             currZPos,stageMoveParams,
             dispImg=False):
#unpack/define some params
    minArea,maxArea=area
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,kernelSz)
    refDist,tol,stepSz=stageMoveParams
# crop the 3 subimages
    centralFish=frame[segmentImgBds[0]:segmentImgBds[1],segmentImgBds[2]:segmentImgBds[3]]
    leftFish=frame[leftSubImgBds[0]:leftSubImgBds[1], leftSubImgBds[2]:leftSubImgBds[3]]
    rightFish=frame[rightSubImgBds[0]:rightSubImgBds[1],rightSubImgBds[2]:rightSubImgBds[3]]
# take central fish, and find its centroid
    thresholded,centroids,_,fishFound=findFish(centralFish,minArea,maxArea,kernel,segmentImgBds)
# once the central fish is found, calculate corr matrix, get distances between left and right fish,
# repeat for however many elements are in roiSizes
    distances=[]
    if fishFound:
    ##template for corr analysis is square roi around the centroid of the central fish
        for roi in roiSizes:
    ##sometimes the centroid is at the very edges of the frame, which will cause the roi calculation to fail due to
    ##being out of bounds, the if statement above handles that situation
            if np.all(centroids-roi>0):
                template=centralFish[centroids[0]-roi:centroids[0]+roi,centroids[1]-roi:centroids[1]+roi]
                dist,max1,max3=getDistance(template,leftFish,rightFish,
                                                     leftSubImgBds,rightSubImgBds)
                distances.append(dist)
            else:
                print('roi out of central subimage bounds')
                max1,max3=[],[]
        avgDist=np.mean(distances)
    ## calculate new z position
        newZPos=getZPos(refDist,avgDist,tol,
                        currZPos,stepSz)

# display an image of the frame, with dots over calculated fish centroids, and calculated distance printed out
        if dispImg:
        ##resize frame for display
            scaleFac=0.9 #0.5
            frame=cv2.resize(frame,(int(frame.shape[1]*scaleFac),int(frame.shape[0]*scaleFac)))
        ##because frame was resized, have to scale max idxs accordingly so they correspond to correct location on displayed img
            cv2.circle(frame,(int(max1[1]*scaleFac),int(max1[0]*scaleFac)),7,(65535,65535,65535),-1)
            cv2.circle(frame,(int(max3[1]*scaleFac),int(max3[0]*scaleFac)),7,(65535,65535,65535),-1)
            cv2.putText(frame,"dist left to right subimages: "+str(avgDist),(50,50),cv2.FONT_HERSHEY_PLAIN,1,
                        (65535,65535,65535),1,cv2.LINE_AA)
            cv2.imshow('autofocus frame',frame)
            cv2.waitKey(1)
    else:
        newZPos,max1,max3,avgDist=[],[],[],[]
    return newZPos, max1, max3, avgDist
