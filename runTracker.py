from time import perf_counter
from zTrackFcts import *
import datetime
from utils import *
from time import sleep

#all times in s
exposure=0.2
extTime=0.005
duration=60
savePath='C:/Users/labadmin/Desktop/221103/fish1/cont_0.2Exp_5mWPower/'
#flags [saveImage,trackZ,timeLoop] -- for saving,engaging z-tracker,
                                        #and timing the acquisition loop, as well as sendData flag for
                                        #whether or not to send zpos updates to labview
fctFlags=[True,False,False]
sendData=False
#set z tracking parameters
area=[10,50000]
kernelSz=(5,5)
roiSizes=[10,12,20]
#[left,right,template] -- get this from 'getZparams.py' by setting 'pckSubRegions' to True
subImgBds=[[348, 390, 109, 113], [460, 393, 112, 115], [342, 504, 112, 113]]
#set z stage movement params -- [refDist(pxs), tol(pxs), stepSz(mm)]
stageMoveParams=[115,1,0.02]
#some setup for z tracking -- setup socket comms for communicating with LabView, define subimage bounds
if fctFlags[1]:
    conn,addr,s=setupSocket()
    leftSubImgBds,rightSubImgBds,segmentImgBds=getImgBds(subImgBds)
#setup daq connections
cam_trigger,cam_arm,_=setup_DAQ()
#setup camera
cam=setup_cam(exposure,-25)
cam.setup_acquisition(mode='sequence')
#initialize frameCounter and list to store frames for saving
frameList,timing,newZPosList,sentData,readData=[],[],[],[],[]
frameCount=0
cam.start_acquisition()
sleep(5)
print('started acquisition')
#save start time to txt file in same path as imaging files
if fctFlags[0]:
    currentTime=str(datetime.datetime.now())
    with open(savePath+'imaging_start_time.txt','w') as f:
        f.write(currentTime)
#start loop
while frameCount<int(duration/(exposure+extTime)):
    frameCount+=1
    read_frame = False
#send camera trigger
    cam_trigger.write(True)
#read the frame (takes ~4ms)
    try:
        cam.wait_for_frame(timeout=0.5)
        frame = cam.read_newest_image()
        read_frame=True
    except:
        frame=[]
        print('missed frame, sucker!')
    if read_frame:
    #save the (full sized) frame, if applicable
        if fctFlags[0]:
            frameList.append(frame)
    #do the z tracking, if applicable
        if fctFlags[1]:
            ##for every frame, read the current stage position, if it is a valid position, run the tracker to get new position
            conn.sendall(b'getZ')
            conn.sendall(b'\r\n')
            currZPos = conn.recv(20)
            currZPos = float(currZPos.decode())
            ##quite often, currZPos is read as some value close to 0, this screws everything up downstream, so I put a threshold
            ##here which will bypass the z tracker completely if read position is unreasonable
            if currZPos < 2.0:
                currZPos = 0
                proc_frame=[]
            if currZPos > 0:
                ##process_frame rescales frame and remaps to 8bit for cv2 functions down the line
                tic = perf_counter()
                procFrame=process_frame(frame)
                ##run z tracker, get newZPos, it will return as either the calculated new position, or [] if the central
                ##subimage fish was not segmented properly
                zCalc, max1, max3, avgDist = zTracker(procFrame,
                                                      area, kernelSz, roiSizes,
                                                      segmentImgBds, leftSubImgBds, rightSubImgBds,
                                                      currZPos, stageMoveParams,
                                                      dispImg=False)
                toc = perf_counter()
                ##store newZ in newZPosList, if it is a number
                if zCalc:
                    newZPosList.append(zCalc)
            else:
                print('invalid data read ' + str(currZPos))
                proc_frame = []
            ##only update the stage position at frameRate/5 Hz, and if newZPosList has at least one element
            if newZPosList: #frameCount % 5 == 0 and newZPosList:
                ##filterZList removes outliers from pos history, then average, and use that as new position to be sent to stage
                ##for some reason have to call filterZList twice, because on first call, returns just the straight
                ##mean of newZPosList every now and then. No clue why
                newZPos = filterZList(newZPosList)
                ##set upper limit to newZPos at stage travel limit
                if newZPos > 3.0:
                    newZPos = 3.0
                    print('reached upper stage limit')
                ##send newZPos to labview
                if not sendData:
                    newZPos = 2.8
                data = str(newZPos)
                conn.sendall(b'setZ')
                conn.sendall(b'\r\n')
                conn.sendall(data.encode())
                conn.sendall(b'\r\n')
                ##clear zPosHist for the next iteration
                newZPosList = []
                ##store sent and read stage data
                sentData.append(newZPos)
                readData.append(currZPos)
            # display image, if zTracking worked, overlay dots over calculated subimg centroids, and calculated distance
            scaleFac = 0.9
            dispFrame = cv2.resize(procFrame, (int(procFrame.shape[1] * scaleFac), int(procFrame.shape[0] * scaleFac)))
            try:
                ##because frame was resized, have to scale max idxs accordingly so they correspond to correct location on displayed img
                cv2.circle(dispFrame, (int(max1[1] * scaleFac), int(max1[0] * scaleFac)), 7, (65535, 65535, 65535), -1)
                cv2.circle(dispFrame, (int(max3[1] * scaleFac), int(max3[0] * scaleFac)), 7, (65535, 65535, 65535), -1)
                cv2.putText(dispFrame, "dist left to right subimages: " + str(avgDist), (50, 50),
                            cv2.FONT_HERSHEY_PLAIN, 1,
                            (65535, 65535, 65535), 1, cv2.LINE_AA)
            except:
                cv2.circle(dispFrame, (50, 50), 17, (65535, 65535, 65535), -1)
            cv2.imshow('autofocus frame', dispFrame)
            cv2.waitKey(1)
        #if not running z tracker, display the resized frame, after rescaling intensity
        if not fctFlags[1]:
            dispFrame = cv2.resize(frame, (750, 750))
            dispFrame = cv2.normalize(dispFrame, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
            cv2.putText(dispFrame, "frame: " + str(frameCount),
                        (50, 50), cv2.FONT_HERSHEY_PLAIN, 1,
                        (60000, 60000, 60000), 1, cv2.LINE_AA)
            cv2.imshow('frame', dispFrame)
            cv2.waitKey(1)
        if fctFlags[2]:
            timing.append(toc - tic)
    cam_trigger.write(False)
cam.stop_acquisition()
cam.close()
cam_trigger.close()
cam_arm.close()
if fctFlags[1]:
    s.close()
if fctFlags[2]:
    print('average loop runtime '+str(np.mean(timing)))
if fctFlags[0]:

    for i in range(3,len(frameList),4):
        print('saving frame ' + str(i))
        save_frame=frameList[i]
        cv2.imwrite(savePath + 'frame' + str(i) + '.TIFF', save_frame)
'''        
    for i, frame in enumerate(frameList):
        print('saving frame ' + str(i))
        cv2.imwrite(savePath+'frame'+str(i)+'.TIFF',frame)
'''