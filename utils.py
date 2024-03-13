from pylablib.devices import Andor
from nidaqmx import Task
import socket
import numpy as np
import cv2 as cv2
from time import sleep

def setup_cam(exposure,temp):
    cam=Andor.AndorSDK3Camera()
    cam.set_attribute_value('Overlap', False)
    print('rolling shutter, overlap mode engaged? ' + str(cam.get_attribute_value('Overlap')))
    cam.set_exposure(exposure)
    cam.set_temperature(temp)
    print('exposure set to- ' + str(cam.get_exposure()))
    print('camera temp- ' + str(cam.get_temperature()))
    cam.set_trigger_mode('ext')
    print('trigger mode set- ' + str(cam.get_trigger_mode()))
    cam.set_attribute_value("AuxiliaryOutSource", "FireAll")
    print('camera aux output mode set- ' + str(cam.get_attribute_value('AuxiliaryOutSource')))
    print('bit depth set ' + str(cam.get_attribute_value('BitDepth')))
    return cam
def setup_DAQ():
    cam_trigger = Task()
    cam_trigger.do_channels.add_do_chan('Dev2/port2/line0')

    cam_arm = Task()
    cam_arm.di_channels.add_di_chan('Dev2/port0/line4')

    laser_trigger = Task()
    laser_trigger.do_channels.add_do_chan('Dev2/port0/line3')
    return cam_trigger, cam_arm, laser_trigger
def process_frame(frame):
    procFrame = frame.copy()
    procFrame = cv2.resize(procFrame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
    procFrame = cv2.normalize(procFrame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return procFrame
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
def collect_avg_img(cam,cam_trigger,laser_trigger,laser_pulse,numImgs):
    frame_hist = []
    cam.start_acquisition()
    for i in range(numImgs):
        sleep(0.1)
        laser_trigger.write(True)
        cam_trigger.write(True)
        sleep(laser_pulse)
        # turn off led, reset cam trigger
        #led_trigger.write(0)
        laser_trigger.write(False)
        cam_trigger.write(False)
        # read the frame (takes ~4ms)
        try:
            cam.wait_for_frame(timeout=1.0)
            frame = cam.read_newest_image()
            frame_hist.append(frame)
            # print(frame.shape)
        except:
            print('frame missed')
    cam.stop_acquisition()
    frame_array = np.array(frame_hist).transpose(1, 2, 0)
    return np.mean(frame_array, axis=2).astype('uint16')