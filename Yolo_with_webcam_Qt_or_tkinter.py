# YOLO object detection using a webcam
# Exact same demo as the read from disk, but instead of disk a webcam is used.
# import the necessary packages
import numpy as np
# import argparse
import imutils
import time
import cv2
import os
import PySimpleGUIQt as sg
from scipy.spatial import distance as dist
import math

i_vid = r'/Users/deepan/Desktop/ui/PySimpleGUI-YOLO-master/dv.mp4'
o_vid = r'/Users/deepan/Desktop/ui/PySimpleGUI-YOLO-master/videos/output.mp4'
y_path = r'yolo-coco'
sg.ChangeLookAndFeel('LightGrey1')

#### COLLISION DETECTION INPUT PARAMETERS #######
cr1x = int(100)
cr1y = int(600)
collisionRef = (cr1x, cr1y)

cr2x = int(530)
cr2y = int(600)
collisionRef2 = (cr2x, cr2x)

cr3x = int(1000)
cr3y = int(600)
collisionRef3 = (cr3x, cr3y)

WarningMessage = 'Collision Detection, Drive Safe'

############################   ICON CONFIGURATION  ####################################################
# General
# boundaryboxthickness
# boundaryboxcolor
# TextColor
# TextDotColor

# car
CarSize = int(3)
cr = int(255)
cg = int(255)
cb = int(255)
CarColor = (cr, cg, cb)
cwor = int(211)
cwog = int(211)
cwob = int(211)
CarWheelOutColor = (cwor, cwog, cwob)
cwir = int(0)
cwig = int(0)
cwib = int(0)
CarWheelInColor = (cwir, cwig, cwib)

# Truck
TruckSize = int(3)
tr = int(255)
tg = int(255)
tb = int(255)
TruckColor = (cr, cg, cb)
twor = int(211)
twog = int(211)
twob = int(211)
TruckWheelOutColor = (cwor, cwog, cwob)
twir = int(0)
twig = int(0)
twib = int(0)
TruckWheelInColor = (cwir, cwig, cwib)

# Bus
BusSize = int(3)
br = int(255)
bg = int(255)
bb = int(255)
BusColor = (cr, cg, cb)
bwor = int(211)
bwog = int(211)
bwob = int(211)
BusWheelOutColor = (cwor, cwog, cwob)
bwir = int(0)
bwig = int(0)
bwib = int(0)
BusWheelInColor = (cwir, cwig, cwib)

# Bike
# BikeSize
# BikeThickness
# BikeLineColor
# BikeWheelOutColor
# BikeWheelInColor

# Animal
animalSize = int(2)
ar = int(255)
ag = int(255)
ab = int(255)
AnimalBodyColor = (ar, ag, ab)
alcr = int(255)
alcg = int(255)
alcb = int(255)
AnimalLineColor = (alcr, alcg, alcb)

# People
personSize = int(3)
PeopleHeadColor = (255, 255, 255)
PeopleBodyColor = (255, 255, 255)

# TrafficLight
# TrafficLightSize

###################################   UI LAYOUT   #############################################
layout = [
    [sg.Text('YOLO VIDEO PROCESSING TOOL', size=(52, 2), font=('Any', 18), text_color='#8B0000', justification='left')],
    [sg.Text('Path to input video'), sg.In(i_vid, size=(40, 1), text_color='#8B0000', key='input'), sg.FileBrowse()],
    [sg.Text('Optional Path to output video'), sg.In(o_vid, size=(40, 1), text_color='#8B0000', key='output'),
     sg.FileSaveAs()],
    [sg.Text('Yolo base path'), sg.In(y_path, size=(10, 1), text_color='#8B0000', key='yolo'), sg.FolderBrowse()],
    [sg.Text('Confidence'),
    sg.Slider(range=(0, 10), orientation='h', resolution=1, default_value=5, size=(15, 15), text_color='#8B0000',
               key='confidence'), sg.T('  ', key='_CONF_OUT_'),
    sg.Text(' ' * 8), sg.Checkbox('Write to disk', text_color='#8B0000', key='_DISK_')],

    [sg.Text('Collision Detection Config', size=(22, 1), font=('Any', 18), text_color='#8B0000', justification='left')],
    [sg.Text(' Check for Collision Detection '), sg.Checkbox('', text_color='#8B0000', key='cd')],
    [sg.Text('Ref point 1'), sg.In(cr1x, size=(5, 1), text_color='#8B0000', key='_cr1x'),
    sg.In(cr1y, size=(5, 1), text_color='#8B0000', key='_cr1y'),
    sg.Text('Ref point 2'), sg.In(cr2x, size=(5, 1), text_color='#8B0000', key='_cr2x'),
    sg.In(cr2y, size=(5, 1), text_color='#8B0000', key='_cr2y'),
    sg.Text('Ref point 3'), sg.In(cr3x, size=(5, 1), text_color='#8B0000', key='_cr3x'),
    sg.In(cr3y, size=(5, 1), text_color='#8B0000', key='_cr3y')
    ],
    [sg.Text('Collision Warning Message'), sg.In(WarningMessage, size=(40, 1), text_color='#8B0000', key='wm')],

    [sg.Text('Icon Configuration Panel', size=(22, 1), font=('Any', 18), text_color='#8B0000', justification='left')],
    ############## Car Config ##############################
    [sg.Text('Car', size=(22, 1), font=('Any', 15), text_color='#8B0000', justification='left')],
    [sg.Text('Car Size'), sg.In(CarSize, size=(5, 1), text_color='#8B0000', key='c1')],
    # [sg.Text('Car Color'), sg.In(CarColor, size=(20, 1), text_color='#8B0000', key='_C2_')],
    [sg.Text('Car Color rgb'), sg.In(cr, size=(5, 1), text_color='#8B0000', key='_cr_'),
    sg.In(cg, size=(5, 1), text_color='#8B0000', key='_cg_'),
    sg.In(cb, size=(5, 1), text_color='#8B0000', key='_cb_'),
    # [sg.Text('Car Wheel Out Color'), sg.In(CarWheelOutColor, size=(20, 1), text_color='#8B0000', key='c3')],
    sg.Text('Car Wheel Out Color rgb'), sg.In(cwor, size=(5, 1), text_color='#8B0000', key='_cwor_'),
    sg.In(cwog, size=(5, 1), text_color='#8B0000', key='_cwog_'),
    sg.In(cwob, size=(5, 1), text_color='#8B0000', key='_cwob_'),
    # [sg.Text('Car Wheel In Color'), sg.In(CarWheelInColor, size=(20, 1), text_color='#8B0000', key='c4')],
    sg.Text('Car Wheel In Color rgb'), sg.In(cwir, size=(5, 1), text_color='#8B0000', key='_cwir_'),
    sg.In(cwig, size=(5, 1), text_color='#8B0000', key='_cwig_'),
    sg.In(cwib, size=(5, 1), text_color='#8B0000', key='_cwib_')],

############## Truck Config ##############################
    [sg.Text('Truck', size=(22, 1), font=('Any', 15), text_color='#8B0000', justification='left')],
    [sg.Text('Truck Size'), sg.In(TruckSize, size=(5, 1), text_color='#8B0000', key='t1')],
    # [sg.Text('Car Color'), sg.In(CarColor, size=(20, 1), text_color='#8B0000', key='_C2_')],
    [sg.Text('Truck Color rgb'), sg.In(tr, size=(5, 1), text_color='#8B0000', key='_tr_'),
    sg.In(tg, size=(5, 1), text_color='#8B0000', key='_tg_'),
    sg.In(tb, size=(5, 1), text_color='#8B0000', key='_tb_'),
    # [sg.Text('Car Wheel Out Color'), sg.In(CarWheelOutColor, size=(20, 1), text_color='#8B0000', key='c3')],
    sg.Text('Truck Wheel Out Color rgb'), sg.In(cwor, size=(5, 1), text_color='#8B0000', key='_twor_'),
    sg.In(twog, size=(5, 1), text_color='#8B0000', key='_twog_'),
    sg.In(twob, size=(5, 1), text_color='#8B0000', key='_twob_'),
    # [sg.Text('Car Wheel In Color'), sg.In(CarWheelInColor, size=(20, 1), text_color='#8B0000', key='c4')],
    sg.Text('Truck Wheel In Color rgb'), sg.In(twir, size=(5, 1), text_color='#8B0000', key='_twir_'),
    sg.In(twig, size=(5, 1), text_color='#8B0000', key='_twig_'),
    sg.In(twib, size=(5, 1), text_color='#8B0000', key='_twib_')],

############## Bus Config ##############################
    [sg.Text('Bus', size=(22, 1), font=('Any', 15), text_color='#8B0000', justification='left')],
    [sg.Text('Bus Size'), sg.In(BusSize, size=(5, 1), text_color='#8B0000', key='b1')],
    # [sg.Text('Car Color'), sg.In(CarColor, size=(20, 1), text_color='#8B0000', key='_C2_')],
    [sg.Text('Bus Color rgb'), sg.In(br, size=(5, 1), text_color='#8B0000', key='_br_'),
    sg.In(bg, size=(5, 1), text_color='#8B0000', key='_bg_'),
    sg.In(bb, size=(5, 1), text_color='#8B0000', key='_bb_'),
    # [sg.Text('Car Wheel Out Color'), sg.In(CarWheelOutColor, size=(20, 1), text_color='#8B0000', key='c3')],
    sg.Text('Bus Wheel Out Color rgb'), sg.In(bwor, size=(5, 1), text_color='#8B0000', key='_bwor_'),
    sg.In(bwog, size=(5, 1), text_color='#8B0000', key='_bwog_'),
    sg.In(bwob, size=(5, 1), text_color='#8B0000', key='_bwob_'),
    # [sg.Text('Car Wheel In Color'), sg.In(CarWheelInColor, size=(20, 1), text_color='#8B0000', key='c4')],
    sg.Text('Bus Wheel In Color rgb'), sg.In(bwir, size=(5, 1), text_color='#8B0000', key='_bwir_'),
    sg.In(bwig, size=(5, 1), text_color='#8B0000', key='_bwig_'),
    sg.In(bwib, size=(5, 1), text_color='#8B0000', key='_bwib_')],

############## Animal Config ##############################
    [sg.Text('Animal', size=(22, 1), font=('Any', 15), text_color='#8B0000', justification='left')],
    [sg.Text('Animal Size'), sg.In(animalSize, size=(5, 1), text_color='#8B0000', key='a1')],
    # [sg.Text('Car Color'), sg.In(CarColor, size=(20, 1), text_color='#8B0000', key='_C2_')],
    [sg.Text('Animal Body Color rgb'), sg.In(ar, size=(5, 1), text_color='#8B0000', key='_ar_'),
    sg.In(ag, size=(5, 1), text_color='#8B0000', key='_ag_'),
    sg.In(ab, size=(5, 1), text_color='#8B0000', key='_ab_'),
    # [sg.Text('Car Wheel In Color'), sg.In(CarWheelInColor, size=(20, 1), text_color='#8B0000', key='c4')],
    sg.Text('Animal Line Color rgb'), sg.In(alcr, size=(5, 1), text_color='#8B0000', key='_alcr_'),
    sg.In(alcg, size=(5, 1), text_color='#8B0000', key='_alcg_'),
    sg.In(alcb, size=(5, 1), text_color='#8B0000', key='_alcb_')],

    [sg.OK(), sg.Cancel(), sg.Stretch()],
]

win = sg.Window('YOLO Video',
                default_element_size=(15, 1),
                text_justification='right',
                auto_size_text=False,
                resizable=True,).Layout(layout)
event, values = win.Read()
########################### UPDATE VALUES OF ICON CONFIG ON OK ACTION ##################################################
if event is None or event == 'OK':
    cr1x = int(values['_cr1x'])
    cr1y = int(values['_cr1y'])
    collisionRef = (cr1x, cr1y)

    cr2x = int(values['_cr2x'])
    cr2y = int(values['_cr2y'])
    collisionRef2 = (cr2x, cr2y)

    cr3x = int(values['_cr3x'])
    cr3y = int(values['_cr3y'])
    collisionRef3 = (cr3x, cr3y)

    WarningMessage = values['wm']

    CarSize = int(values['c1'])
    cr = int(values['_cr_'])
    cg = int(values['_cg_'])
    cb = int(values['_cb_'])
    CarColor = (cr, cg, cb)
    cwor = int(values['_cwor_'])
    cwog = int(values['_cwog_'])
    cwob = int(values['_cwob_'])
    CarWheelOutColor = (cwor, cwog, cwob)
    cwir = int(values['_cwir_'])
    cwig = int(values['_cwir_'])
    cwib = int(values['_cwir_'])
    CarWheelInColor = (cwir, cwig, cwib)

    TruckSize = int(values['t1'])
    tr = int(values['_tr_'])
    tg = int(values['_tg_'])
    tb = int(values['_tb_'])
    TruckColor = (tr, tg, tb)
    twor = int(values['_twor_'])
    twog = int(values['_twog_'])
    twob = int(values['_twob_'])
    TruckWheelOutColor = (twor, twog, twob)
    twir = int(values['_twir_'])
    twig = int(values['_twir_'])
    twib = int(values['_twir_'])
    TruckWheelInColor = (twir, twig, twib)

    BusSize = int(values['b1'])
    br = int(values['_br_'])
    bg = int(values['_bg_'])
    bb = int(values['_bb_'])
    BusColor = (cr, cg, cb)
    bwor = int(values['_bwor_'])
    bwog = int(values['_bwog_'])
    bwob = int(values['_bwob_'])
    BusWheelOutColor = (bwor, bwog, bwob)
    bwir = int(values['_bwir_'])
    bwig = int(values['_bwir_'])
    bwib = int(values['_bwir_'])
    BusWheelInColor = (bwir, bwig, bwib)

    # Animal
    animalSize = int(values['a1'])
    ar = int(values['_ar_'])
    ag = int(values['_ag_'])
    ab = int(values['_ab_'])
    AnimalBodyColor = (ar, ag, ab)
    alcr = int(values['_alcr_'])
    alcg = int(values['_alcg_'])
    alcb = int(values['_alcb_'])
    AnimalLineColor = (alcr, alcg, alcb)

    #Enable and Disable Collison detection Feature
    EnableCollisionDetect = values['cd']


if event is None or event == 'Cancel':
    exit()
write_to_disk = values['_DISK_']
use_webcam = False
print(use_webcam)
args = values

win.Close()

# imgbytes = cv2.imencode('.png', image)[1].tobytes()  # ditto
gui_confidence = args["confidence"] / 10
gui_threshold = 3 / 10
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# loop over frames from the video file stream
win_started = False
if use_webcam:
    cap = cv2.VideoCapture(0)
while True:
    # read the next frame from the file or webcam
    if use_webcam:
        grabbed, frame = cap.read()
    else:
        grabbed, frame = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > gui_confidence:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, gui_confidence, gui_threshold)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():

            if confidences[i] > 0.85:
                CarSize = 3
            elif confidences[i] < 0.85:
                CarSize = 2
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            frame_h, frame_w, _ = frame.shape
            bbox_thick = int(0.6 * (frame_h + frame_w) / 600)


            # coor0 = y
            # coor1 = x
            # coor2 = y + h
            # coor3 = x + w

            ############################   COORDINATES FOR DRAWING  ####################################################
            # coordinates for default baoundary box
            # 1 2
            # 3 4
            c5 = (x, y)  # 1
            c6 = (x + w, y)  # 2
            c7 = (x, y + h)  # 3
            c8 = (x + w, y + h)  # 4

            # mid points for creating half line
            mp1 = (int((x + (x + w)) / 2), int((y + y) / 2))
            mp2 = (int(((x + w) + (x + w)) / 2), int((y + (y + h)) / 2))
            mp3 = (int((x + (x + w)) / 2), int(((y + h) + (y + h)) / 2))
            mp4 = (int((x + x) / 2), int((y + (y + h)) / 2))

            # mid point for creating half of half line
            ump1 = (int((c6[0] + mp1[0]) / 2), int((c6[1] + mp1[1]) / 2))
            ump2 = (int((c6[0] + mp2[0]) / 2), int((c6[1] + mp2[1]) / 2))
            ump3 = (int((c7[0] + mp3[0]) / 2), int((c7[1] + mp3[1]) / 2))
            ump4 = (int((c7[0] + mp4[0]) / 2), int((c7[1] + mp4[1]) / 2))

            # diagonal line point
            c9 = (int((x + w) + 20), int(y - 20))

            # text Position
            c10 = (int((x + w) + 24), int(y - 24))

            ######################################   ICON COORDINATES   ################################################

            # REF POINT FOR ALL ICON DRAWINGS
            nc11x = int((x + w) + 20)
            nc11y = int(y - 50)

            # CAR COOR
            nc11 = (nc11x, nc11y)  # 1
            nc22 = (int(nc11x + (10 * CarSize)), nc11y)  # 2
            nc23 = (int(nc11x + (10 * CarSize)), int(nc11y - (2 * CarSize)))  # 3
            nc24 = (int(nc11x + (6 * CarSize)), int(nc11y - (3 * CarSize)))  # 4
            nc25 = (int(nc11x + (4 * CarSize)), int(nc11y - (4 * CarSize)))  # 5
            nc26 = (int(nc11x + (2 * CarSize)), int(nc11y - (4 * CarSize)))  # 6
            nc27 = (int(nc11x + (1 * CarSize)), int(nc11y - (2 * CarSize)))  # 7
            nc28 = (int(nc11x), int(nc11y - (2 * CarSize)))  # 8

            ncw1 = (nc11x + (3 * CarSize), nc11y)  # wheel 1
            ncw2 = (int(ncw1[0] + (5 * CarSize)), nc11y)  # wheel 2

            # TRUCK COOR
            t11 = (nc11x, nc11y)  # 1 bottom line start
            t22 = (int(nc11x + (10 * TruckSize)), nc11y)  # 2 bottom line end
            t23 = (int(nc11x + (10 * TruckSize)), int(nc11y - (3 * TruckSize)))  # 3 headlight area
            t24 = (int(nc11x + (7 * TruckSize)), int(nc11y - (3 * TruckSize)))  # 4
            t25 = (int(nc11x + (7 * TruckSize)), int(nc11y - (5 * TruckSize)))  # 5
            t26 = (int(nc11x), int(nc11y - (5 * TruckSize)))  # 6

            tw1 = (nc11x + (3 * TruckSize), nc11y)  # wheel 1
            tw2 = (int(ncw1[0] + (5 * TruckSize)), nc11y)  # wheel 2

            # BUS COOR
            b11 = (nc11x, nc11y)  # 1 bottom line start
            b22 = (int(nc11x + (10 * BusSize)), nc11y)  # 2 bottom line end
            b25 = (int(nc11x + (10 * BusSize)), int(nc11y - (5 * BusSize)))  # 5
            b26 = (int(nc11x), int(nc11y - (5 * BusSize)))  # 6

            bw1 = (nc11x + (3 * BusSize), nc11y)  # wheel 1
            bw2 = (int(ncw1[0] + (5 * BusSize)), nc11y)  # wheel 2

            # PERSON COOR
            p1 = (int(((x + w)) + 27), int(y - 50))  # 1
            p2 = (int((x + w) + (43 + personSize)), int(y - 50))  # 2
            p3 = (int((p1[0] + p2[0]) / 2), int(int((p1[1] + p2[1]) / 2) - (15 + personSize)))  # face

            # points to make traffic signal
            ts1 = (int((x + w) + 27), int(y - 50))  # red
            ts2 = (int((x + w) + 43), int(y - 50))  # green
            ts3 = (int((x + w) + 35), int(y - 50))  # yellow

            # points to make bicyle and motorbike

            bikeSize = int(3)

            # bikebodys
            bm1 = (nc11x, nc11y)  # 1 bottom line start
            bm2 = (int(nc11x + (10 * bikeSize)), nc11y)  # 2 bottom line end

            bm3 = (nc11x + (2 * bikeSize), nc11y + (4 * bikeSize))  # wheel 1
            bm4 = (int(nc11x + (8 * bikeSize)), nc11y + (4 * bikeSize))  # wheel 2

            # points for making a dog

            d11 = (nc11x, nc11y)  # 1
            d22 = (int(nc11x + (10 * animalSize)), nc11y)
            d23 = (int(nc11x + (5 * animalSize)), nc11y - (10 * animalSize))
            d24 = (int(nc11x + (12 * animalSize)), nc11y - (2 * animalSize))
            d25 = (int(nc11x + (8 * animalSize)), nc11y - (2 * animalSize))
            d26 = (int(nc11x + (0 * animalSize)), nc11y - (12 * animalSize))
            d27 = (int(nc11x + (3 * animalSize)), nc11y - (7 * animalSize))



            ######################  LABEL ##################################

            # text = "{}: {:.4f}".format(LABELS[classIDs[i]],
            #                            confidences[i])
            # cv2.putText(frame, text, (x, y - 5),
            # 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            fontScale = 0.5
            cv2.putText(frame, LABELS[classIDs[i]], (c10[0], int(c10[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)

            #################################### COllISION DETECTION  ##############################################

            clr = (x, y + h)  # collision reference point for object
            # clr = (100, 1000)  # collision reference point for object
            # print(clr, 'bbb')
            pof = (100, 570)
            # cv2.line(frame, collisionRef, clr, (255, 0, 0), 1)

            if clr[0] >= 100 and clr[0] <= 1000 and EnableCollisionDetect:
                # print('warning')
                checkForCollisionDetect = True
            else:
                checkForCollisionDetect = False

            if checkForCollisionDetect and clr[1] >= 570:
                collisionDetected = True
            else:
                collisionDetected = False

            print(collisionDetected, 'cd')

            if collisionDetected:
                cv2.putText(frame, 'Collision Warning, Drive Safe!', (clr[0] + 4, int(clr[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)

            # dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # dist = math.sqrt((1000 - 500) ** 2 + (500 - 400) ** 2)
            # print(dist, 'distance')

            # cv2.circle(frame, collisionRef, 2, (0, 0, 0), 20)
            # cv2.circle(frame, collisionRef2, 2, (0, 0, 0), 20)
            # cv2.circle(frame, collisionRef3, 2, (0, 0, 0), 20)
            # cv2.circle(frame, clr, 2, (255, 0, 0), 5)
            # cv2.circle(frame, pof, 2, (255, 0, 0), 5)

            #################################### COllISION DETECTION END  ##############################################


            ############# ICON DRAWINGS ###################################
            thickness = 1


            if LABELS[classIDs[i]] == 'Stop sign':
                cv2.line(frame, c6, ump1, (255, 0, 0), 1)
                cv2.line(frame, c6, ump2, (255, 0, 0), 1)
                cv2.line(frame, c7, ump3, (255, 0, 0), 1)
                cv2.line(frame, c7, ump4, (255, 0, 0), 1)
                cv2.line(frame, c6, c9, (255, 0, 0), 1)
                cv2.circle(frame, c9, radius, (255, 0, 0), 1)

            elif LABELS[classIDs[i]] == 'Bus':
                contoursBus = np.array([[b11],
                                        [b22],
                                        [b25],
                                        [b26],
                                        ])
                cv2.circle(frame, bw1, 2 * BusSize, BusWheelOutColor, thickness)  # wheel out 1
                cv2.circle(frame, bw2, 2 * BusSize, BusWheelOutColor, thickness)  # wheel out 2
                cv2.circle(frame, bw1, 1 * BusSize, BusWheelInColor, thickness)  # wheel in 1
                cv2.circle(frame, bw2, 1 * BusSize, BusWheelInColor, thickness)  # wheel in 2

                cv2.fillPoly(frame, pts=[contoursBus], color=BusColor)


            elif LABELS[classIDs[i]] == 'Truck':

                # Truck
                cv2.circle(frame, tw1, 2 * TruckSize, TruckWheelOutColor, thickness)  # wheel out 1
                cv2.circle(frame, tw2, 2 * TruckSize, TruckWheelOutColor, thickness)  # wheel out 2
                cv2.circle(frame, tw1, 1 * TruckSize, TruckWheelInColor, thickness)  # wheel in 1
                cv2.circle(frame, tw2, 1 * TruckSize, TruckWheelInColor, thickness)  # wheel in 2

                contours = np.array([[t11],
                                     [t22],
                                     [t23],
                                     [t24],
                                     [t25],
                                     [t26],
                                     ])

                cv2.fillPoly(frame, pts=[contours], color=TruckColor)

            elif LABELS[classIDs[i]] == 'Car':

                cv2.circle(frame, ncw1, 2 * CarSize, CarWheelOutColor, thickness)  # wheel out 1
                cv2.circle(frame, ncw2, 2 * CarSize, CarWheelOutColor, thickness)  # wheel out 2
                cv2.circle(frame, ncw1, 1 * CarSize, CarWheelInColor, thickness)  # wheel in 1
                cv2.circle(frame, ncw2, 1 * CarSize, CarWheelInColor, thickness)  # wheel in 2

                contours2 = np.array([[nc11],
                                      [nc22],
                                      [nc23],
                                      [nc24],
                                      [nc25],
                                      [nc26],
                                      [nc27],
                                      [nc28]])

                cv2.fillPoly(frame, pts=[contours2], color=CarColor)

            elif LABELS[classIDs[i]] == 'Person':
                cv2.circle(frame, p3, 1 + personSize, PeopleHeadColor, thickness)

                contours3 = np.array([[p1],
                                      [p2],
                                      [p3]])

                cv2.fillPoly(frame, pts=[contours3], color=PeopleBodyColor)

                # yellow boundary box
                cv2.line(frame, c6, ump1, (255, 255, 0), 2)
                cv2.line(frame, c6, ump2, (255, 255, 0), 2)
                cv2.line(frame, c7, ump3, (255, 255, 0), 2)
                cv2.line(frame, c7, ump4, (255, 255, 0), 2)
                cv2.line(frame, c6, c9, (255, 255, 0), 2)
                cv2.circle(frame, c9, radius, (255, 255, 0), 2)

            elif LABELS[classIDs[i]] == 'Traffic light':
                cv2.circle(frame, ts1, 4, (255, 0, 0), thickness)
                cv2.circle(frame, ts3, 4, (255, 255, 0), thickness)
                cv2.circle(frame, ts2, 4, (0, 128, 0), thickness)
                cv2.line(frame, c6, ump1, (255, 0, 0), 1)
                cv2.line(frame, c6, ump2, (255, 0, 0), 1)
                cv2.line(frame, c7, ump3, (255, 0, 0), 1)
                cv2.line(frame, c7, ump4, (255, 0, 0), 1)
                cv2.line(frame, c6, c9, (255, 0, 0), 1)
                cv2.circle(frame, c9, radius, (255, 0, 0), 1)

            elif LABELS[classIDs[i]] == 'Bicycle':
                cv2.circle(frame, bm3, 2 * bikeSize, color, wheelThickness)
                cv2.circle(frame, bm4, 2 * bikeSize, color, wheelThickness)
                cv2.circle(frame, bm3, 1 * bikeSize, bcolor, wheelThickness)  # inside wheel 1
                cv2.circle(frame, bm4, 1 * bikeSize, bcolor, wheelThickness)  # inside wheel 2
                # cv2.circle(frame,bm3,4,color,thickness)
                cv2.line(frame, bm1, bm2, color, 1)


            elif LABELS[classIDs[i]] == 'Motorbike':
                cv2.circle(frame, bm1, 4, color, thickness)
                cv2.circle(frame, bm2, 4, color, thickness)
                cv2.circle(frame, bm1, 3, bcolor, thickness)  # inside wheel 1
                cv2.circle(frame, bm2, 3, bcolor, thickness)  # inside wheel 2
                # cv2.circle(frame,bm3,4,color,thickness)
                cv2.line(frame, bm3, bm4, color, 1)

            elif LABELS[classIDs[i]] == 'Dog':

                # animal body
                contours4 = np.array([[d11],
                                      [d22],
                                      [d23],
                                      ])

                cv2.fillPoly(frame, pts=[contours4], color=AnimalBodyColor)

                # animal tail

                cv2.line(frame, d22, d24, AnimalLineColor, 1)
                cv2.line(frame, d24, d25, AnimalLineColor, 1)

                # contoursdogTail = np.array([[d22],
                #                             [d24],
                #                             [d25],
                #                            ])

                # cv2.fillPoly(frame, pts =[contoursdogTail], color=(255,255,0))

                # dog mouth

                cv2.line(frame, d23, d26, (255, 255, 0), 1)
                cv2.line(frame, d26, d27, (255, 255, 0), 1)

                # contoursdogMouth = np.array([[d23],
                #                              [d26],
                #                              [d27],
                #                            ])

                # cv2.fillPoly(frame, pts =[contoursdogMouth], color=(255,255,0))

                # yellow boundary box
                cv2.line(frame, c6, ump1, (255, 255, 0), 2)
                cv2.line(frame, c6, ump2, (255, 255, 0), 2)
                cv2.line(frame, c7, ump3, (255, 255, 0), 2)
                cv2.line(frame, c7, ump4, (255, 255, 0), 2)
                cv2.line(frame, c6, c9, (255, 255, 0), 2)
                cv2.circle(frame, c9, radius, (255, 255, 0), 2)

            ################################# Drawing starts from here  ################################################

            ###################### NEW BOUNDARY BOX DRAWING #############################
            color = (255, 255, 255)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) ----> default boundary box

            thickness = 2
            wheelThickness = 1

            cv2.line(frame, c6, ump1, color, thickness)
            cv2.line(frame, c6, ump2, color, thickness)
            cv2.line(frame, c7, ump3, color, thickness)
            cv2.line(frame, c7, ump4, color, thickness)


            # diagnonal line pointing label
            cv2.line(frame, c6, c9, color, thickness)

            radius = 4
            color = (255, 255, 255)
            bcolor = (128, 128, 128)
            thickness = -1

            # circle for label
            cv2.circle(frame, c9, radius, color, thickness)


    if write_to_disk:
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total))

        # write the output frame to disk
        writer.write(frame)
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto

    if not win_started:
        win_started = True
        layout = [
            [sg.Text('YOLO', size=(30, 1))],
            [sg.Image(data=imgbytes, key='_IMAGE_')],
            [sg.Text('Confidence'),
             sg.Slider(range=(0, 10), orientation='h', resolution=1, default_value=5, size=(15, 15), key='confidence'),
            ],
            [sg.Exit()]
        ]
        win = sg.Window('YOLO Output',
                        default_element_size=(14, 1),
                        text_justification='right',
                        auto_size_text=False).Layout(layout).Finalize()
        image_elem = win.FindElement('_IMAGE_')
    else:
        image_elem.Update(data=imgbytes)

    event, values = win.Read(timeout=0)
    if event is None or event == 'Exit':
        break
    gui_confidence = values['confidence'] / 10
    gui_threshold = 3/ 10

win.Close()

# release the file pointers
print("[INFO] cleaning up...")
writer.release() if writer is not None else None
vs.release()
