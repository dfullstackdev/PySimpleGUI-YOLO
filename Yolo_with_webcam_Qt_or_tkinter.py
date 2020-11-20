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

i_vid = r'/Users/deepan/Desktop/ui/PySimpleGUI-YOLO-master/np.mp4'
o_vid = r'/Users/deepan/Desktop/detections/new/o.mp4'
y_path = r'yolo-coco'
sg.ChangeLookAndFeel('LightGrey1')


#### COLLISION DETECTION INPUT PARAMETERS #######
cr1x = int(100)
cr1y = int(600)
collisionRef = (cr1x, cr1y)

# cr2x = int(530)
# cr2y = int(600)

# (int((cr1x + cr3x)/2), cdpof)

cr3x = int(1000)
cr3y = int(600)
collisionRef3 = (cr3x, cr3y)

collisionRef2 = (int((cr1x + cr3x)/2), cr1y)

ch = 570
WarningMessage = 'Potential Collision Detection, Drive Safe'
wmSize = float(0.5)

cdpof = 630
accidentMessage = 'Accident Detected'
amSize = float(0.5)

#### LABEL COLOR
lc1 = int(255)
lc2 = int(255)
lc3 = int(255)
labelColor = (lc1, lc2, lc3)

lc1a = int(0)
lc2a = int(0)
lc3a = int(0)
labelColora = (lc1a, lc2a, lc3a)

lc1b = int(0)
lc2b = int(0)
lc3b = int(0)
labelColorb = (lc1b, lc2b, lc3b)

lc1c = int(0)
lc2c = int(0)
lc3c= int(0)
labelColorc = (lc1c, lc2c, lc3c)


########## Boundary Box Color ################
bbb = int(0)
bbg = int(69)
bbr = int(255)
Bbcolor = (bbb, bbg, bbr)
Bbthickness = int(3)
Bbthickness1 = int(3)
Bbthickness2 = int(2)
Bbthickness3 = int(1)

bbb1 = int(0)
bbg1 = int(0)
bbr1 = int(255)
Bbcolor1 = (bbb1, bbg1, bbr1)

bbb2 = int(3)
bbg2 = int(200)
bbr2 = int(255)
Bbcolor2 = (bbb2, bbg2, bbr2)

bbb3 = int(255)
bbg3 = int(255)
bbr3 = int(255)
Bbcolor3 = (bbb3, bbg3, bbr3)

lpb = int(0)
lpg = int(69)
lpr = int(255)
Lpcolor = (lpb, lpg, lpr)

lpb1 = int(0)
lpg1 = int(69)
lpr1 = int(255)
Lpcolor1 = (lpb1, lpg1, lpr1)

lpb2 = int(0)
lpg2 = int(255)
lpr2 = int(255)
Lpcolor2 = (lpb2, lpg2, lpr2)

lpb3 = int(0)
lpg3 = int(255)
lpr3 = int(0)
Lpcolor3 = (lpb3, lpg3, lpr3)

Lpthickness = int(3)
Lpthickness1 = int(3)
Lpthickness2 = int(2)
Lpthickness3 = int(1)

ldpb = int(0)
ldpg = int(69)
ldpr = int(255)
Ldpcolor = (ldpb, ldpg, ldpr)

ldpb1 = int(0)
ldpg1 = int(69)
ldpr1 = int(255)
Ldpcolor1 = (ldpb1, ldpg1, ldpr1)

ldpb2 = int(0)
ldpg2 = int(255)
ldpr2 = int(255)
Ldpcolor2 = (ldpb2, ldpg2, ldpr2)

ldpb3 = int(0)
ldpg3 = int(255)
ldpr3 = int(0)
Ldpcolor3 = (ldpb3, ldpg3, ldpr3)

Ldpradius = int(4)
Ldpradius1 = int(4)
Ldpradius2 = int(3)
Ldpradius3 = int(2)
Ldpthickness = int(-1)

###### Label arrow length ############
c9Postx = int(20)
c9Posty = int(20)

# radius = 4
# color = (255, 255, 255)
# thickness = -1
#
# # circle for label
# cv2.circle(frame, c9, radius, Bbcolor, thickness)

############################   ICON CONFIGURATION  ####################################################
# General
# boundaryboxthickness
# boundaryboxcolor
# TextColor
# TextDotColor
iconPositionx = int(20)
iconPositiony = int(50)

LabelTextPositionx = int(-60)
LabelTextPositiony = int(-20)
LabelSize = float(0.5)

# car
CarSize = int(3)
cr = int(0)
cg = int(0)
cb = int(0)
CarColor = (cr, cg, cb)
cwor = int(0)
cwog = int(0)
cwob = int(0)
CarWheelOutColor = (cwor, cwog, cwob)
cwir = int(255)
cwig = int(255)
cwib = int(255)
CarWheelInColor = (cwir, cwig, cwib)

# Truck
TruckSize = int(3)
tr = int(0)
tg = int(0)
tb = int(0)
TruckColor = (cr, cg, cb)
twor = int(0)
twog = int(0)
twob = int(0)
TruckWheelOutColor = (cwor, cwog, cwob)
twir = int(255)
twig = int(255)
twib = int(255)
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
phb = int(255)
phg = int(255)
phr = int(255)
PeopleHeadColor = (phb, phg, phr)
pfb = int(255)
pfg = int(255)
pfr = int(255)
PeopleBodyColor = (pfb, pfg, pfr)

# TrafficLight
# TrafficLightSize

###################################   UI LAYOUT   #############################################
layout = [
    # [sg.Text('ROADHOW VIDEO PROCESSING TOOL', size=(52, 2), font=('Any', 18), text_color='#8B0000', justification='left')],
    [sg.Text('Path to input video'), sg.In(i_vid, size=(40, 1), text_color='#8B0000', key='input'), sg.FileBrowse(),
    sg.Text('      Optional Path to output video'), sg.In(o_vid, size=(40, 1), text_color='#8B0000', key='output'),
     sg.FileSaveAs()],

    [sg.Text('Training Type'), sg.In(y_path, size=(10, 1), text_color='#8B0000', key='yolo'), sg.FolderBrowse(),
    sg.Text('                        Confidence'),
    sg.Slider(range=(0, 10), orientation='h', resolution=1, default_value=5, size=(15, 15), text_color='#8B0000',
               key='confidence'), sg.T('  ', key='_CONF_OUT_'),
    sg.Text(' '), sg.Checkbox('Write to disk', text_color='#8B0000', key='_DISK_')],

    [sg.Text('Collision Detection Config', size=(40, 1), font=('Any', 18), text_color='#8B0000', justification='left')],
    [sg.Text(' Check for Collision Detection'), sg.Checkbox('           ', text_color='#8B0000', key='cd'),
     sg.Text(' Show Detection Reference'), sg.Checkbox('', text_color='#8B0000', key='srp')],

    [sg.Text('Dashboard Left Coordinate'), sg.In(cr1x, size=(5, 1), text_color='#8B0000', key='_cr1x'),
    sg.In(cr1y, size=(5, 1), text_color='#8B0000', key='_cr1y'),
    # sg.Text('    Dashboard Center Coordinate'), sg.In(cr2x, size=(5, 1), text_color='#8B0000', key='_cr2x'),
    # sg.In(cr2y, size=(5, 1), text_color='#8B0000', key='_cr2y'),
    sg.Text('    Dashboard Right Coordinate'), sg.In(cr3x, size=(5, 1), text_color='#8B0000', key='_cr3x'),
    sg.In(cr3y, size=(5, 1), text_color='#8B0000', key='_cr3y')
    ],

    [sg.Text(' Potential Collision Message'), sg.In(WarningMessage, size=(20, 1), text_color='#8B0000', key='wm'),
     sg.Text('     Text Size'), sg.In(wmSize, size=(5, 1), text_color='#8B0000', key='_wms'),
     sg.Text('     Distance'), sg.In(ch, size=(5, 1), text_color='#8B0000', key='_ch')],

    [sg.Text(' Collision Message'), sg.In(accidentMessage, size=(20, 1), text_color='#8B0000', key='am'),
     sg.Text('     Text Size'), sg.In(amSize, size=(5, 1), text_color='#8B0000', key='_ams'),
     sg.Text('     Distance'), sg.In(cdpof, size=(5, 1), text_color='#8B0000', key='_cdpof')],


    [sg.Text('Layout Configuration Panel', size=(80, 1), font=('Any', 18), text_color='#8B0000', justification='left')],

    [sg.Text(' Layer 1'), sg.Checkbox('Icon', text_color='#8B0000', default=False, key='l1I'),
     sg.Checkbox('Label', text_color='#8B0000', default=True, key='l1L'),
     sg.Checkbox('Arrow', text_color='#8B0000', default=False, key='l1A'),
     sg.Checkbox('BoundaryBox', text_color='#8B0000', default=True, key='l1B'),
     sg.Checkbox('Distance Line', text_color='#8B0000', default=False, key='l1D'),
     sg.Checkbox('c', text_color='#8B0000', default=False, key='l1T'),

     sg.Text('                  Layer 2'), sg.Checkbox('Icon', text_color='#8B0000', default=False, key='l2I'),
     sg.Checkbox('Label', text_color='#8B0000', default=True, key='l2L'),
     sg.Checkbox('Arrow', text_color='#8B0000', default=False, key='l2A'),
     sg.Checkbox('BoundaryBox', text_color='#8B0000', default=True, key='l2B'),
     sg.Checkbox('Distance Line', text_color='#8B0000', default=False, key='l2D'),
     sg.Checkbox('c', text_color='#8B0000', default=False, key='l2T'),


     sg.Text('                   Layer 3'), sg.Checkbox('Icon', text_color='#8B0000', default=False, key='l3I'),
     sg.Checkbox('Label', text_color='#8B0000', default=False, key='l3L'),
     sg.Checkbox('Arrow', text_color='#8B0000', default=False, key='l3A'),
     sg.Checkbox('BoundaryBox', text_color='#8B0000', default=True, key='l3B'),
     sg.Checkbox('Distance Line', text_color='#8B0000', default=False, key='l3D'),
     sg.Checkbox('c', text_color='#8B0000', default=False, key='l3T'),

     sg.Text('                  Directional Visibility:'), sg.Checkbox('Left', text_color='#8B0000', default=True, key='lv'),
     sg.Checkbox('Right', text_color='#8B0000', default=True, key='rv'),

     ],
    #
    # [sg.Text('Boundary Box Color'), sg.In(bbb, size=(5, 1), text_color='#8B0000', key='_bbb_'),
    #  sg.In(bbg, size=(5, 1), text_color='#8B0000', key='_bbg_'),
    #  sg.In(bbr, size=(5, 1), text_color='#8B0000', key='_bbr_'),
    #  ],

    # [sg.Text('Boundary Box Color Gen'), sg.In(bbb, size=(5, 1), text_color='#8B0000', key='_bbb_'),
    #  sg.In(bbg, size=(5, 1), text_color='#8B0000', key='_bbg_'),
    #  sg.In(bbr, size=(5, 1), text_color='#8B0000', key='_bbr_'),

     [sg.Text('Boundary Box Color 1'), sg.In(bbb1, size=(5, 1), text_color='#8B0000', key='_bbb1_'),
     sg.In(bbg1, size=(5, 1), text_color='#8B0000', key='_bbg1_'),
     sg.In(bbr1, size=(5, 1), text_color='#8B0000', key='_bbr1_'), # = Bbcolor1

     sg.Text('      Boundary Box Color 2'), sg.In(bbb2, size=(5, 1), text_color='#8B0000', key='_bbb2_'),
     sg.In(bbg2, size=(5, 1), text_color='#8B0000', key='_bbg2_'),
     sg.In(bbr2, size=(5, 1), text_color='#8B0000', key='_bbr2_'), # = Bbcolor2

     sg.Text('      Boundary Box Color 3'), sg.In(bbb3, size=(5, 1), text_color='#8B0000', key='_bbb3_'),
     sg.In(bbg3, size=(5, 1), text_color='#8B0000', key='_bbg3_'),
     sg.In(bbr3, size=(5, 1), text_color='#8B0000', key='_bbr3_'), # = Bbcolor3

     sg.Text('Boundary Box Thickness 1,2,3'), sg.In(Bbthickness1, size=(5, 1), text_color='#8B0000', key='_bbt1_'),
     sg.In(Bbthickness2, size=(5, 1), text_color='#8B0000', key='_bbt2_'),
     sg.In(Bbthickness3, size=(5, 1), text_color='#8B0000', key='_bbt3_')
     ],

    [sg.Text('Label Arrow Color1 '), sg.In(lpb1, size=(5, 1), text_color='#8B0000', key='_lpb1_'),
     sg.In(lpg1, size=(5, 1), text_color='#8B0000', key='_lpg1_'),
     sg.In(lpr1, size=(5, 1), text_color='#8B0000', key='_lpr1_'),

     sg.Text('Label Arrow Color2 '), sg.In(lpb2, size=(5, 1), text_color='#8B0000', key='_lpb2_'),
     sg.In(lpg2, size=(5, 1), text_color='#8B0000', key='_lpg2_'),
     sg.In(lpr2, size=(5, 1), text_color='#8B0000', key='_lpr2_'),

     sg.Text('Label Arrow Color3 '), sg.In(lpb3, size=(5, 1), text_color='#8B0000', key='_lpb3_'),
     sg.In(lpg3, size=(5, 1), text_color='#8B0000', key='_lpg3_'),
     sg.In(lpr3, size=(5, 1), text_color='#8B0000', key='_lpr3_'),

     sg.Text('Label arrow Thickness 1,2,3'), sg.In(Lpthickness1, size=(5, 1), text_color='#8B0000', key='_lat1_'),
     sg.In(Lpthickness2, size=(5, 1), text_color='#8B0000', key='_lat2_'),
     sg.In(Lpthickness3, size=(5, 1), text_color='#8B0000', key='_lat3_'),

     sg.Text('Label arrow Distance'), sg.In(c9Postx, size=(5, 1), text_color='#8B0000', key='_c9Postx'),
     sg.In(c9Posty, size=(5, 1), text_color='#8B0000', key='_c9Posty'),
     ],

    [sg.Text('Label Point Color 1 '), sg.In(ldpb1, size=(5, 1), text_color='#8B0000', key='_ldpb1_'),
     sg.In(ldpg1, size=(5, 1), text_color='#8B0000', key='_ldpg1_'),
     sg.In(ldpr1, size=(5, 1), text_color='#8B0000', key='_ldpr1_'),

    sg.Text('Label Point Color 2 '), sg.In(ldpb1, size=(5, 1), text_color='#8B0000', key='_ldpb2_'),
     sg.In(ldpg2, size=(5, 1), text_color='#8B0000', key='_ldpg2_'),
     sg.In(ldpr2, size=(5, 1), text_color='#8B0000', key='_ldpr2_'),

    sg.Text('Label Point Color 3 '), sg.In(ldpb1, size=(5, 1), text_color='#8B0000', key='_ldpb3_'),
     sg.In(ldpg3, size=(5, 1), text_color='#8B0000', key='_ldpg3_'),
     sg.In(ldpr3, size=(5, 1), text_color='#8B0000', key='_ldpr3_'),



     sg.Text('Label Point Radius 1,2,3'), sg.In(Ldpradius1, size=(5, 1), text_color='#8B0000', key='_ldps1_'),
     sg.In(Ldpradius2, size=(5, 1), text_color='#8B0000', key='_ldps2_'),
     sg.In(Ldpradius3, size=(5, 1), text_color='#8B0000', key='_ldps3_'),

     sg.Text('Label Point Thickness'), sg.In(Ldpthickness, size=(5, 1), text_color='#8B0000', key='_ldpt_')],

    [sg.Text('Icon Position (x,y) '), sg.In(iconPositionx, size=(5, 1), text_color='#8B0000', key='ipx'),
     sg.In(iconPositiony, size=(5, 1), text_color='#8B0000', key='ipy'),
     sg.Text('       Label Position (x,y) '), sg.In(LabelTextPositionx, size=(5, 1), text_color='#8B0000', key='lpx'),
     sg.In(LabelTextPositiony, size=(5, 1), text_color='#8B0000', key='lpy'),
     sg.Text('Label Color 1 '), sg.In(lc1a, size=(5, 1), text_color='#8B0000', key='_lc1a'),
     sg.In(lc2a, size=(5, 1), text_color='#8B0000', key='_lc2a'),
     sg.In(lc3a, size=(5, 1), text_color='#8B0000', key='_lc3a'),

     sg.Text('Label Color 2 '), sg.In(lc1b, size=(5, 1), text_color='#8B0000', key='_lc1b'),
     sg.In(lc2b, size=(5, 1), text_color='#8B0000', key='_lc2b'),
     sg.In(lc3b, size=(5, 1), text_color='#8B0000', key='_lc3b'),

     sg.Text('Label Color 3 '), sg.In(lc1c, size=(5, 1), text_color='#8B0000', key='_lc1c'),
     sg.In(lc2c, size=(5, 1), text_color='#8B0000', key='_lc2c'),
     sg.In(lc3c, size=(5, 1), text_color='#8B0000', key='_lc3c'),
     sg.Text('Label Size'), sg.In(LabelSize, size=(5, 1), text_color='#8B0000', key='_ls'),],

    ############## Car Config ##############################
    [sg.Text('Car', size=(5, 1), font=('Any', 15), text_color='#8B0000', justification='left'),
     sg.Text(' '), sg.Checkbox('Enable Detection', default=True, text_color='#8B0000', key='cdetec')],
    [sg.Text('Car Size'), sg.In(CarSize, size=(5, 1), text_color='#8B0000', key='c1'),
    # [sg.Text('Car Color'), sg.In(CarColor, size=(20, 1), text_color='#8B0000', key='_C2_')],
    sg.Text('Car Color rgb'), sg.In(cr, size=(5, 1), text_color='#8B0000', key='_cr_'),
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
    [sg.Text('Truck', size=(5, 1), font=('Any', 15), text_color='#8B0000', justification='left'),
     sg.Text(' '), sg.Checkbox('Enable Detection', default=True, text_color='#8B0000', key='tdetec')],
    [sg.Text('Truck Size'), sg.In(TruckSize, size=(5, 1), text_color='#8B0000', key='t1'),
    # [sg.Text('Car Color'), sg.In(CarColor, size=(20, 1), text_color='#8B0000', key='_C2_')],
    sg.Text('Truck Color rgb'), sg.In(tr, size=(5, 1), text_color='#8B0000', key='_tr_'),
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
    [sg.Text('Bus', size=(5, 1), font=('Any', 15), text_color='#8B0000', justification='left'),
     sg.Text(' '), sg.Checkbox('Enable Detection', default=True, text_color='#8B0000', key='bdetec')],
    [sg.Text('Bus Size'), sg.In(BusSize, size=(5, 1), text_color='#8B0000', key='b1'),
    # [sg.Text('Car Color'), sg.In(CarColor, size=(20, 1), text_color='#8B0000', key='_C2_')],
    sg.Text('Bus Color rgb'), sg.In(br, size=(5, 1), text_color='#8B0000', key='_br_'),
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
    [sg.Text('Animal', size=(5, 1), font=('Any', 15), text_color='#8B0000', justification='left'),
     sg.Text(' '), sg.Checkbox('Enable Detection', default=True, text_color='#8B0000', key='adetec')],
    [sg.Text('Animal Size'), sg.In(animalSize, size=(5, 1), text_color='#8B0000', key='a1'),
    # [sg.Text('Car Color'), sg.In(CarColor, size=(20, 1), text_color='#8B0000', key='_C2_')],
    sg.Text('Animal Body Color rgb'), sg.In(ar, size=(5, 1), text_color='#8B0000', key='_ar_'),
    sg.In(ag, size=(5, 1), text_color='#8B0000', key='_ag_'),
    sg.In(ab, size=(5, 1), text_color='#8B0000', key='_ab_'),
    # [sg.Text('Car Wheel In Color'), sg.In(CarWheelInColor, size=(20, 1), text_color='#8B0000', key='c4')],
    sg.Text('Animal Line Color rgb'), sg.In(alcr, size=(5, 1), text_color='#8B0000', key='_alcr_'),
    sg.In(alcg, size=(5, 1), text_color='#8B0000', key='_alcg_'),
    sg.In(alcb, size=(5, 1), text_color='#8B0000', key='_alcb_')],

    ############## People Config ##############################
    [sg.Text('Person', size=(5, 1), font=('Any', 15), text_color='#8B0000', justification='left'),
     sg.Text(' '), sg.Checkbox('Enable Detection', default=True, text_color='#8B0000', key='pdetec')],
    [sg.Text('Person Size'), sg.In(personSize, size=(5, 1), text_color='#8B0000', key='p1'),
     # [sg.Text('Car Color'), sg.In(CarColor, size=(20, 1), text_color='#8B0000', key='_C2_')],
     sg.Text('People Head Color bgr'), sg.In(phb, size=(5, 1), text_color='#8B0000', key='_phb'),
     sg.In(phg, size=(5, 1), text_color='#8B0000', key='_phg'),
     sg.In(phr, size=(5, 1), text_color='#8B0000', key='_phr'),
     # [sg.Text('Car Wheel In Color'), sg.In(CarWheelInColor, size=(20, 1), text_color='#8B0000', key='c4')],
     sg.Text('Person Body Color bgr'), sg.In(pfb, size=(5, 1), text_color='#8B0000', key='_pfb'),
     sg.In(pfg, size=(5, 1), text_color='#8B0000', key='_pfg'),
     sg.In(pfr, size=(5, 1), text_color='#8B0000', key='_pfr')],

    [sg.OK(), sg.Cancel(), sg.Stretch()],
]

win = sg.Window('ROADHOW VIDEO PROCESSING TOOL',
                default_element_size=(15, 1),
                text_justification='right',
                auto_size_text=False,
                ).Layout(layout)

# ([[sg.Column(layout=layout, scrollable=True)]])

event, values = win.Read()
########################### UPDATE VALUES OF ICON CONFIG ON OK ACTION ##################################################
if event is None or event == 'OK':
    EnableCar = int(values['cdetec'])
    EnableCarBackup = int(values['cdetec'])
    EnableTruck = int(values['tdetec'])
    EnableTruckBackup = int(values['tdetec'])
    EnableBus = int(values['bdetec'])
    EnableAnimal = int(values['adetec'])
    EnablePerson = int(values['pdetec'])

    cr1x = int(values['_cr1x'])
    cr1y = int(values['_cr1y'])
    collisionRef = (cr1x, cr1y)

    # cr2x = int(values['_cr2x'])
    # cr2y = int(values['_cr2y'])
    # collisionRef2 = (cr2x, cr2y)

    cr3x = int(values['_cr3x'])
    cr3y = int(values['_cr3y'])
    collisionRef3 = (cr3x, cr3y)

    collisionRef2 = (int((cr1x + cr3x) / 2), cr1y)

    WarningMessage = values['wm']
    ch = int(values['_ch'])
    wmSize = float(values['_wms'])

    cdpof = int(values['_cdpof'])
    accidentMessage = values['am']
    accidentMessageback = values['am']
    amSize = float(values['_ams'])

    # lc1 = int(values['_lc1'])
    # lc2 = int(values['_lc2'])
    # lc3 = int(values['_lc3'])
    # labelColor = (lc1, lc2, lc3)

    lc1a = int(values['_lc1a'])
    lc2a = int(values['_lc2a'])
    lc3a = int(values['_lc3a'])
    labelColora = (lc1a, lc2a, lc3a)

    lc1b = int(values['_lc1b'])
    lc2b = int(values['_lc2b'])
    lc3b = int(values['_lc3b'])
    labelColorb = (lc1b, lc2b, lc3b)

    lc1c = int(values['_lc1c'])
    lc2c = int(values['_lc2c'])
    lc3c = int(values['_lc3c'])
    labelColorc = (lc1c, lc2c, lc3c)

    LabelSize = float(values['_ls'])
    LabelSizeBackup = float(values['_ls'])

    # bbb = int(values['_bbb_'])
    # bbg = int(values['_bbg_'])
    # bbr = int(values['_bbr_'])
    # Bbcolor = (bbb, bbg, bbr)

    bbb1 = int(values['_bbb1_'])
    bbg1 = int(values['_bbg1_'])
    bbr1 = int(values['_bbr1_'])
    Bbcolor1 = (bbb1, bbg1, bbr1)

    bbb2 = int(values['_bbb2_'])
    bbg2 = int(values['_bbg2_'])
    bbr2 = int(values['_bbr2_'])
    Bbcolor2 = (bbb2, bbg2, bbr2)

    bbb3 = int(values['_bbb3_'])
    bbg3 = int(values['_bbg3_'])
    bbr3 = int(values['_bbr3_'])
    Bbcolor3 = (bbb3, bbg3, bbr3)

    Bbthickness1 = int(values['_bbt1_'])
    Bbthickness2 = int(values['_bbt2_'])
    Bbthickness3 = int(values['_bbt3_'])

    EnableArrowVisibility = True
    EnableArrowVisibilityBackup = True
    EnableIconVisibility =  True
    EnableIconVisibilityBackup = True
    EnableLabelVisibility = True
    EnableLabelVisibilityBackup = True
    EnableLabelVisibilityT = True
    EnableLabelVisibilityBackupT = True
    EnableBbVisibility = True
    EnableBbVisibilityBackup = True
    EnableDistanceLine = True
    EnableDistanceLineBackup = True
    iconPositionx = int(values['ipx'])
    iconPositiony = int(values['ipy'])
    LabelTextPositionx = int(values['lpx'])
    LabelTextPositiony = int(values['lpy'])

    Layer1I = values['l1I']
    Layer1L = values['l1L']
    Layer1A = values['l1A']
    Layer1B = values['l1B']
    Layer1D = values['l1D']
    Layer1T = values['l1T']

    Layer2I = values['l2I']
    Layer2L = values['l2L']
    Layer2A = values['l2A']
    Layer2B = values['l2B']
    Layer2D = values['l2D']
    Layer2T = values['l2T']

    Layer3I = values['l3I']
    Layer3L = values['l3L']
    Layer3A = values['l3A']
    Layer3B = values['l3B']
    Layer3D = values['l3D']
    Layer3T = values['l3T']

    LV = values['lv']
    RV = values['rv']

    lpb1 = int(values['_lpb1_'])
    lpg1 = int(values['_lpg1_'])
    lpr1 = int(values['_lpr1_'])
    Lpcolor1 = (lpb1, lpg1, lpr1)

    lpb2 = int(values['_lpb2_'])
    lpg2 = int(values['_lpg2_'])
    lpr2 = int(values['_lpr2_'])
    Lpcolor2 = (lpb2, lpg2, lpr2)

    lpb3 = int(values['_lpb3_'])
    lpg3 = int(values['_lpg3_'])
    lpr3 = int(values['_lpr3_'])
    Lpcolor3 = (lpb2, lpg2, lpr2)

    Lpthickness1 = int(values['_lat1_'])
    Lpthickness2 = int(values['_lat2_'])
    Lpthickness3 = int(values['_lat3_'])

    ldpb1 = int(values['_ldpb1_'])
    ldpg1 = int(values['_ldpg1_'])
    ldpr1 = int(values['_ldpr1_'])
    Ldpcolor1 = (ldpb1, ldpg1, ldpr1)

    ldpb2 = int(values['_ldpb2_'])
    ldpg2 = int(values['_ldpg2_'])
    ldpr2 = int(values['_ldpr2_'])
    Ldpcolor2 = (ldpb2, ldpg2, ldpr2)

    ldpb3 = int(values['_ldpb3_'])
    ldpg3 = int(values['_ldpg3_'])
    ldpr3 = int(values['_ldpr3_'])
    Ldpcolor3 = (ldpb3, ldpg3, ldpr3)

    Ldpradius1 = int(values['_ldps1_'])
    Ldpradius2 = int(values['_ldps2_'])
    Ldpradius3 = int(values['_ldps3_'])
    Ldpthickness = int(values['_ldpt_'])

    c9Postx = int(values['_c9Postx'])
    c9Posty = int(values['_c9Posty'])

    CarSize = int(values['c1'])
    newC = int(values['c1'])

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

    personSize = int(values['p1'])
    phb = int(values['_phb'])
    phg = int(values['_phg'])
    phr = int(values['_phr'])
    PeopleHeadColor = (phb, phg, phr)
    pfb = int(values['_pfb'])
    pfg = int(values['_pfg'])
    pfr = int(values['_pfr'])
    PeopleBodyColor = (pfb, pfg, pfr)

    TruckSize = int(values['t1'])
    newT = int(values['t1'])
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
    ShowCdRp = values['srp']

    ##########Coordinates as percentage############
    var0 = cr1x
    var100 = collisionRef2[0]
    var10 = round((10 * (var100 - var0) / 100) + var0)
    var20 = round((20 * (var100 - var0) / 100) + var0)
    var30 = round((30 * (var100 - var0) / 100) + var0)
    var40 = round((40 * (var100 - var0) / 100) + var0)
    var50 = round((50 * (var100 - var0) / 100) + var0)
    var60 = round((60 * (var100 - var0) / 100) + var0)
    var70 = round((70 * (var100 - var0) / 100) + var0)
    var80 = round((80 * (var100 - var0) / 100) + var0)
    var90 = round((90 * (var100 - var0) / 100) + var0)

    print('Coordinates as percentage', ' o%: ',var0,' 1o%: ', var10, ' 2o%: ',var20, ' 3o%: ',var30, ' 4o%: ',var40, ' 5o%: ',var50,' 6o%: ',var60, ' 7o%: ',var70, ' 8o%: ',var80, ' 9o%: ',var90, ' 1oo%: ',var100)
    # (int(round(CarSize / 2)))

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
            # FCS =(int(round(CarSize / 2)))
            # distLabel1 = int(c7[1] - c6[1])
            # distLabel = str(distLabel1)
            # print(FCS, CarSize, round(CarSize/2))
            # print(round(3.5))
            # if distLabel1 > 32:
            #     CarSize = 3
            # if distLabel1 < 32:
            #     CarSize = FCS
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

            ObjectCenter = (int((c6[0] + c7[0])/2), int((c6[1] + c7[1])/2))

            # Distance based layout logic
            # LABELS[classIDs[i]] == 'Car'
            distCoor = (int((x + (x + w)) / 2), int((y + (y + h)) / 2))
            FCS = int(CarSize - 1)
            FCS3 = int(CarSize - 2)
            LCS =  float(LabelSize - 0.1)
            LCS3 = float(LabelSize - 0.2)
            distLabel1 = int(c7[1] - c6[1])
            distLabel = str(distLabel1)
            if distLabel1 >= 100:
                EnableDistanceLine = Layer1D
                EnableBbVisibility = Layer1B
                EnableIconVisibility = Layer1I
                EnableLabelVisibility = Layer1L
                EnableLabelVisibilityT = Layer1T
                EnableArrowVisibility = Layer1A
                CarSize = FCS
                LabelSize = LCS
                labelColor = labelColora
                Bbthickness = Bbthickness1
                Lpthickness = Lpthickness1
                Ldpradius = Ldpradius1
                Bbcolor = Bbcolor1
                Lpcolor = Lpcolor1
                Ldpcolor = Ldpcolor1
            if distLabel1 < 100:
                # EnableCar = False
                # EnableTruck = False
                # EnableBus = False
                EnableDistanceLine = Layer2D
                EnableBbVisibility = Layer2B
                EnableIconVisibility = Layer2I
                EnableLabelVisibility = Layer2L
                EnableLabelVisibilityT = Layer2T
                EnableArrowVisibility = Layer2A
                CarSize = FCS
                LabelSize = LCS
                labelColor = labelColorb
                Bbthickness = Bbthickness2
                Lpthickness = Lpthickness2
                Ldpradius = Ldpradius2
                Bbcolor = Bbcolor2
                Lpcolor = Lpcolor2
                Ldpcolor = Ldpcolor2

            if distLabel1 < 32:
                # EnableCar = False
                # EnableTruck = False
                EnableDistanceLine = Layer3D
                EnableBbVisibility = Layer3B
                EnableIconVisibility = Layer3I
                EnableLabelVisibility = Layer3L
                EnableLabelVisibilityT = Layer3T
                EnableArrowVisibility = Layer3A
                CarSize = FCS3
                LabelSize = LCS3
                labelColor = labelColorc
                Bbthickness = Bbthickness3
                Lpthickness = Lpthickness3
                Ldpradius = Ldpradius3
                Bbcolor = Bbcolor3
                Lpcolor = Lpcolor3
                Ldpcolor = Ldpcolor3


            # cv2.circle(frame, distCoor, 1, (0, 255, 255), 5)  # MINPOINT of object
            # cv2.putText(frame, distLabel, (distCoor[0], distCoor[1] + 3),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)


            wt1l = (int((x + (x + w)) / 2), int((y + (y + h)) / 2))
            wt1 = (int((x + (x + w)) / 2), int((y + (y + h)) / 2)-30)
            wt2 = (int((x + (x + w)) / 2)-30 , int((y + (y + h)) / 2) + 30)
            wt3 = (int((x + (x + w)) / 2) + 30, int((y + (y + h)) / 2) + 30)

            # wt2 = (x+60, (y + h)+10)

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

            c9 = (int((x + w) + c9Postx), int(y - c9Posty))

            # text Position
            c10 = (int((x + w) + 24), int(y - 24))

            # label bg
            lbg1 = (x + w, y)
            lbg2 = (x + w, y - 19)
            lbg3 = ((x + w) - 38, y - 19)
            lbg4 = ((x + w) - 42, y - 10)
            lbg5 = ((x + w) - 42, y)





            ######################################   ICON COORDINATES   ################################################

            # REF POINT FOR ALL ICON DRAWINGS
            # iconPositionx = int(20)
            # iconPositiony = int(50)

            nc11x = int((x + w) + iconPositionx)
            nc11y = int(y - iconPositiony)


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

            cw24 = (int(nc11x + (6 * CarSize)), int(nc11y - (2 * CarSize)))  # 4
            cw25 = (int(nc11x + (4 * CarSize)), int(nc11y - (3 * CarSize)))  # 5
            cw26 = (int(nc11x + (3 * CarSize)), int(nc11y - (3 * CarSize)))  # 6
            cw27 = (int(nc11x + (2 * CarSize)), int(nc11y - (2 * CarSize)))  # 7


            # TRUCK COOR
            t11 = (nc11x, nc11y)  # 1 bottom line start
            t22 = (int(nc11x + (10 * TruckSize)), nc11y)  # 2 bottom line end
            t23 = (int(nc11x + (10 * TruckSize)), int(nc11y - (3 * TruckSize)))  # 3 headlight area
            t24 = (int(nc11x + (7 * TruckSize)), int(nc11y - (3 * TruckSize)))  # 4
            t25 = (int(nc11x + (7 * TruckSize)), int(nc11y - (5 * TruckSize)))  # 5
            t26 = (int(nc11x), int(nc11y - (5 * TruckSize)))  # 6

            tw1 = (nc11x + (3 * TruckSize), nc11y)  # wheel 1
            tw2 = (int(ncw1[0] + (5 * TruckSize)), nc11y)  # wheel 2

            tw11 = (int(nc11x + (1 * TruckSize)), int(nc11y - (2 * TruckSize)))  # 5
            tw12 = (int(nc11x + (3 * TruckSize)), int(nc11y - (2 * TruckSize)))  # 5
            tw13 = (int(nc11x + (3 * TruckSize)), int(nc11y - (4 * TruckSize)))  # 5
            tw14 = (int(nc11x + (1 * TruckSize)), int(nc11y - (4 * TruckSize)))  # 5

            tw15 = (int(nc11x + (4 * TruckSize)), int(nc11y - (2 * TruckSize)))  # 5
            tw16 = (int(nc11x + (6 * TruckSize)), int(nc11y - (2 * TruckSize)))  # 5
            tw17 = (int(nc11x + (6 * TruckSize)), int(nc11y - (4 * TruckSize)))  # 5
            tw18 = (int(nc11x + (4 * TruckSize)), int(nc11y - (4 * TruckSize)))  # 5

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



            ######################  GENERAL LABEL ##################################

            # text = "{}: {:.4f}".format(LABELS[classIDs[i]],
            #                            confidences[i])
            # cv2.putText(frame, text, (x, y - 5),
            # 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # fontScale = 0.5
            # LabelTextPositionx = int(0)
            # LabelTextPositiony = int(2)

            # cv2.putText(frame, LABELS[classIDs[i]], (c10[0] + LabelTextPositionx, int(c10[1] - LabelTextPositiony)), cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

            #################################### OBJECT SIDE DETECTION W.R.T dashboard #############################################
            ObjectMidPoint = mp3
            DashBoardMidPoint = collisionRef2
            # and ObjectMidPoint[1] >= ch:
            # cv2.circle(frame, collisionRef2, 5, (255, 255, 255), 5)
            if ShowCdRp:
                cv2.circle(frame, ObjectMidPoint, 1, (0, 255, 0), 5) ###### Object middle point to decide its position w.r.t dashboard #################

            ######### Logic to decide object's position w.r.t dashboard ############
            # and ObjectMidPoint[1] >= ch ##### ----> Value to enable the logic to decide object's position w.r.t dashboard only when it's in potential collision range ##########
            if ObjectMidPoint[0] < DashBoardMidPoint[0]:
                ObjectPositionLeft = True
                EnableCar = LV
                EnableTruck = LV
                EnableBus = LV
                # EnableBbVisibility = LV
                # EnableIconVisibility = LV
                # EnableLabelVisibility = LV
                # EnableArrowVisibility = LV
                if ShowCdRp and ObjectPositionLeft:
                    cv2.putText(frame, 'left', (ObjectMidPoint[0], int(ObjectMidPoint[1] - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                wmSize, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)

            elif ObjectMidPoint[0] > DashBoardMidPoint[0]:
                ObjectPositionRight = True
                EnableCar = RV
                EnableTruck = RV
                EnableBus = RV
                # EnableBbVisibility = RV
                # EnableIconVisibility = RV
                # EnableLabelVisibility = RV
                # EnableArrowVisibility = RV
                if ShowCdRp and ObjectPositionRight:
                    cv2.putText(frame, 'right', (ObjectMidPoint[0], int(ObjectMidPoint[1] - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                wmSize, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)

            elif ObjectMidPoint[0] == DashBoardMidPoint[0]:
                ObjectMiddle = True
                EnableCar = RV
                EnableTruck = RV
                EnableBus = RV
                # EnableBbVisibility = RV
                # EnableIconVisibility = RV
                # EnableLabelVisibility = RV
                # EnableArrowVisibility = RV
                if ShowCdRp and ObjectMiddle:
                    cv2.putText(frame, 'middle', (ObjectMidPoint[0], int(ObjectMidPoint[1] - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                wmSize, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)

                ############################### HUD #########################################

            # roadline = (255, 255, 255)
            # roadlinebackup = (255, 255, 255)
            # one = (cr1x + 90, ch + 25)
            # two = (cr3x - 360, ch + 25)
            # mot = (int((one[0] + two[0])/2), (int((one[1] + two[1])/2)) - 50)
            #
            # three = (cr1x + 90, ch + 1)
            # four = (cr3x - 360, ch + 1)
            # mtf = (int((three[0] + four[0]) / 2), (int((three[1] + four[1]) / 2)) - 50)
            #
            # five = (cr1x + 120, ch - 10)
            # six = (cr3x - 390, ch - 10)
            # seven = (cr1x + 120, ch - 30)
            # eight = (cr3x - 390, ch - 30)
            #
            # nine = (cr1x + 150, ch - 40)
            # ten = (cr3x - 420, ch - 40)
            # eleven = (cr1x + 150, ch - 60)
            # twelve = (cr3x - 420, ch - 60)
            #
            # thirteen = (cr1x + 180, ch - 70)
            # fourteen = (cr3x - 450, ch - 70)
            # fifteen = (cr1x + 180, ch - 90)
            # sixteen = (cr3x - 450, ch - 90)
            #
            # hod = False
            # alldraw = False
            # drawzero = False
            # drawone = False
            # drawtwo = False
            # drawthree = False
            #
            # cv2.line(frame, one, mot, (255, 255, 255), 1)
            # cv2.line(frame, mot, two, (255, 255, 255), 1)
            # cv2.line(frame, two, four, (255, 255, 255), 1)
            # cv2.line(frame, four, mtf, (255, 255, 255), 1)
            # cv2.line(frame, mtf, three, (255, 255, 255), 1)
            # cv2.line(frame, three, one, (255, 255, 255), 1)
            #
            # # cv2.line(frame, five, six, (255, 255, 255), 1)
            # # cv2.line(frame, six, eight, (255, 255, 255), 1)
            # # cv2.line(frame, eight, seven, (255, 255, 255), 1)
            # # cv2.line(frame, seven, five, (255, 255, 255), 1)
            # #
            # # cv2.line(frame, nine, ten, (255, 255, 255), 1)
            # # cv2.line(frame, ten, twelve, (255, 255, 255), 1)
            # # cv2.line(frame, twelve, eleven, (255, 255, 255), 1)
            # # cv2.line(frame, eleven, nine, (255, 255, 255), 1)
            # #
            # # cv2.line(frame, thirteen, fourteen, (255, 255, 255), 1)
            # # cv2.line(frame, fourteen, sixteen, (255, 255, 255), 1)
            # # cv2.line(frame, sixteen, fifteen, (255, 255, 255), 1)
            # # cv2.line(frame, fifteen, thirteen, (255, 255, 255), 1)
            #
            # if mp3[0] >= fifteen[0] and mp3[0] <= sixteen[0] and mp3[1] >= fifteen[1]:
            #     hod = True
            # if hod:
            #     print('eligible')
            #     # cv2.putText(frame, 'eligible', mp3,
            #     #             cv2.FONT_HERSHEY_SIMPLEX,
            #     #             0.5, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
            #
            # if distLabel1 >= 100 and hod and mp3[1] >= three[1]:
            #     drawzero = True
            #
            # elif distLabel1 >= 100 and hod and mp3[1] < three[1] and mp3[1] >= seven[1]:
            #    drawone = True
            #    cv2.line(frame, one, two, (0, 0, 255), 1)
            #    cv2.line(frame, two, four, (0, 0, 255), 1)
            #    cv2.line(frame, four, three, (0, 0, 255), 1)
            #    cv2.line(frame, three, one, (0, 0, 255), 1)
            #
            # elif distLabel1 >= 100 and hod and mp3[1] < seven[1] and mp3[1] >= eleven[1]:
            #    drawtwo = True
            #    cv2.line(frame, one, two, (0, 0, 180), 1)
            #    cv2.line(frame, two, four, (0, 0, 180), 1)
            #    cv2.line(frame, four, three, (0, 0, 180), 1)
            #    cv2.line(frame, three, one, (0, 0, 180), 1)
            #
            #    cv2.line(frame, five, six, (0, 0, 180), 1)
            #    cv2.line(frame, six, eight, (0, 0, 180), 1)
            #    cv2.line(frame, eight, seven, (0, 0, 180), 1)
            #    cv2.line(frame, seven, five, (0, 0, 180), 1)
            #
            # elif distLabel1 >= 100 and hod and mp3[1] < eleven[1] and mp3[1] >= fifteen[1]:
            #     drawthree = True
            #     cv2.line(frame, one, two, (3, 200, 255), 1)
            #     cv2.line(frame, two, four, (3, 200, 255), 1)
            #     cv2.line(frame, four, three, (3, 200, 255), 1)
            #     cv2.line(frame, three, one, (3, 200, 255), 1)
            #
            #     cv2.line(frame, five, six, (3, 200, 255), 1)
            #     cv2.line(frame, six, eight,(3, 200, 255), 1)
            #     cv2.line(frame, eight, seven,(3, 200, 255), 1)
            #     cv2.line(frame, seven, five, (3, 200, 255), 1)
            #
            #     cv2.line(frame, nine, ten, (3, 200, 255), 1)
            #     cv2.line(frame, ten, twelve,(3, 200, 255), 1)
            #     cv2.line(frame, twelve, eleven,(3, 200, 255), 1)
            #     cv2.line(frame, eleven, nine, (3, 200, 255), 1)
            #
            # if hod and drawzero:
            #     cv2.putText(frame, '  MAINTAIN SAFE DISTANCE', mp3,
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.5, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
            # # if hod and drawone:
            # #     cv2.putText(frame, 'one', mp3,
            # #                 cv2.FONT_HERSHEY_SIMPLEX,
            # #                 0.5, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
            # # if hod and drawtwo:
            # #     cv2.putText(frame, 'two', mp3,
            # #                 cv2.FONT_HERSHEY_SIMPLEX,
            # #                 0.5, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
            # # if hod and drawthree:
            # #     cv2.putText(frame, 'three', mp3,
            # #                 cv2.FONT_HERSHEY_SIMPLEX,
            # #                 0.5, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)


            #################################### COllISION DETECTION  ##############################################

            # clr = (x, y + h)  # collision reference point for object ----> OBJECT REFERENCE OF DETECTION
            if ObjectPositionLeft :
                clr = (x + w, y + h)

            elif ObjectPositionRight:
                clr = (x + w, y + h)
            clr = (x + w, y + h)


            ##############POTENTIAL COLLISION LOC BLUE LINE COORD ########################
            pof = (cr1x, ch)
            pof2 = (int((cr1x + cr3x)/2), ch)
            pof3 = (cr3x, ch)
            # print('var0 :',var0, 'cr1x: ', cr1x)



            ############## COLLISION LOC RED LINE COORD ########################
            cd1 = (cr1x, cdpof)
            cd2 = (int((cr1x + cr3x)/2), cdpof)
            cd3 = (cr3x, cdpof)


            if clr[0] >= cr1x and clr[0] <= cr3x and EnableCollisionDetect: ###### Checking between var0 and var100 --> between min val to max val
                # print('warning')
                checkForCollisionDetect = True
            else:
                checkForCollisionDetect = False

            if checkForCollisionDetect and clr[1] >= ch:
                collisionDetected = True
            else:
                collisionDetected = False

            if checkForCollisionDetect and clr[1] >= cdpof:
                accidentDetected = True
            else:
                accidentDetected = False

            print(collisionDetected, 'cd')

            if collisionDetected and not accidentDetected:
                contourswt = np.array([[wt1],
                                       [wt2],
                                       [wt3],
                                       ])

                cv2.fillPoly(frame, pts=[contourswt], color=(211, 211, 211))
                ##################### PERCENTAGE OF POTENTIAL COLLISION ##########################  --> add and Object ObjectPositionRight
                if clr[0]>=var0 and clr[0]<=var10:
                    cv2.putText(frame, '10%', (wt1l[0] - 18, wt1l[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                elif clr[0]>var10 and clr[0]<=var20:
                    cv2.putText(frame, '20%', (wt1l[0] - 18, wt1l[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                elif clr[0]>var20 and clr[0]<=var30:
                    cv2.putText(frame, '30%', (wt1l[0] - 18, wt1l[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                elif clr[0]>var30 and clr[0]<=var40:
                    cv2.fillPoly(frame, pts=[contourswt], color=(0, 174, 255))
                    cv2.putText(frame, '40%', (wt1l[0] - 18, wt1l[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)

                elif clr[0]>var40 and clr[0]<=var50:
                    cv2.fillPoly(frame, pts=[contourswt], color=(0, 174, 255))
                    cv2.putText(frame, '50%', (wt1l[0] - 18, wt1l[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                elif clr[0]>var50 and clr[0]<=var60:
                    cv2.fillPoly(frame, pts=[contourswt], color=(0, 174, 255))
                    cv2.putText(frame, '60%', (wt1l[0] - 18, wt1l[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                elif clr[0]>var60 and clr[0]<=var70:
                    cv2.fillPoly(frame, pts=[contourswt], color=(68, 65, 165))
                    cv2.putText(frame, '70%', (wt1l[0] - 18, wt1l[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                elif clr[0]>var70 and clr[0]<=var80:
                    cv2.fillPoly(frame, pts=[contourswt], color=(68, 65, 165))
                    cv2.putText(frame, '80%', (wt1l[0] - 18, wt1l[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                elif clr[0]>var80 and clr[0]<=var90:
                    cv2.fillPoly(frame, pts=[contourswt], color=(68, 65, 165))
                    cv2.putText(frame, '90%', (wt1l[0] - 18, wt1l[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                elif clr[0]>var90 and clr[0]<=collisionRef2[0]:
                    cv2.fillPoly(frame, pts=[contourswt], color=(68, 65, 165))
                    cv2.putText(frame, '99%', (wt1l[0] - 18, wt1l[1] + 18),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                ########## POTENTIAL COLLISION MESSAGE ######################
                # cv2.putText(frame, WarningMessage, (c5[0] + 4, int(c5[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                #             wmSize, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)
                cv2.putText(frame, WarningMessage, (wt1l[0] - 40, wt1l[1] + 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)

                # contourswt = np.array([[wt1],
                #                        [wt2],
                #                        [wt3],
                #                        ])
                #
                # cv2.fillPoly(frame, pts=[contourswt], color=(68, 68, 165))
                # cv2.putText(frame, '!', (wt1l[0]-5, wt1l[1] + 20),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.9, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)

            if accidentDetected:
                cv2.putText(frame, accidentMessage, (c5[0] + 4, int(c5[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            amSize, (0, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)

                contourswt2 = np.array([[wt1],
                                       [wt2],
                                       [wt3],
                                       ])

                cv2.fillPoly(frame, pts=[contourswt2], color=(68, 68, 165))
                cv2.putText(frame, '!', (wt1l[0]-5, wt1l[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), bbox_thick // 2, lineType=cv2.LINE_AA)


            accidentMessageback



            # dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # dist = math.sqrt((1000 - 500) ** 2 + (500 - 400) ** 2)
            # print(dist, 'ncence')
            if ShowCdRp:
                cv2.circle(frame, collisionRef, 1, (0, 255, 255), 5) #MINPOINT 0%
                # cv2.putText(frame, '0%',
                #             (collisionRef[0], int(collisionRef[1] - 1)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

                cv2.circle(frame, (var10, cr1y), 1, (0, 255, 255), 5) # 10%
                # cv2.putText(frame, '10%',
                #             (var10, int(cr1y - 1)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)
                cv2.circle(frame, (var20, cr1y), 1, (0, 255, 255), 5) # 20%
                # cv2.putText(frame, '20%',
                #             (var20, int(cr1y - 1)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

                cv2.circle(frame, (var30, cr1y), 1, (0, 255, 255), 5) # 30%
                # cv2.putText(frame, '30%',
                #             (var30, int(cr1y - 1)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

                cv2.circle(frame, (var40, cr1y), 1, (0, 255, 255), 5) # 40%
                # cv2.putText(frame, '40%',
                #             (var40, int(cr1y - 1)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

                cv2.circle(frame, collisionRef2, 1, (0, 0, 255), 5) # 100%
                cv2.circle(frame, collisionRef2, 1, (0, 255, 255), 4)  # 100%
                # cv2.putText(frame, '100%',
                #             (collisionRef2[0], int(collisionRef2[1] + 2)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

                cv2.circle(frame, (var50, cr1y), 1, (0, 255, 255), 5)  # 40%
                # cv2.putText(frame, '50%',
                #             (var50, int(cr1y - 1)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

                # cv2.circle(frame, collisionRef2, 5, (255, 255, 255), 4)

                cv2.circle(frame, (var60, cr1y), 1, (0, 255, 255), 5) # 60%
                # cv2.putText(frame, '60%',
                #             (var60, int(cr1y - 1)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

                cv2.circle(frame, (var70, cr1y), 1, (0, 255, 255), 5) # 70%
                # cv2.putText(frame, '70%',
                #             (var70, int(cr1y - 1)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

                cv2.circle(frame, (var80, cr1y), 1, (0, 255, 255), 5) # 80%
                # cv2.putText(frame, '80%',
                #             (var80, int(cr1y - 1)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

                cv2.circle(frame, (var90, cr1y), 1, (0, 255, 255), 5) # 90%
                # cv2.putText(frame, '90%',
                #             (var90, int(cr1y - 1)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

                cv2.circle(frame, collisionRef3, 1, (0, 255, 255), 5) #MAXPOINT 100%
                # cv2.putText(frame, '100%',
                #             (collisionRef3[0], int(collisionRef3[1] - 1)),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

###################  POTENTIAL COLLISION LINE OF CONTROL BLUE LINE #########################
                cv2.circle(frame, pof, 2, (255, 0, 0), 5)
                cv2.circle(frame, pof2, 2, (255, 0, 0), 5)
                cv2.circle(frame, pof3, 2, (255, 0, 0), 5)
                cv2.line(frame, pof, pof3, (255, 0, 0), 1)

################### COLLISION LINE OF CONTROL RED LINE #########################
                cv2.circle(frame, cd1, 2, (0, 0, 255), 5)
                cv2.circle(frame, cd2, 2, (0, 0, 255), 5)
                cv2.circle(frame, cd3, 2, (0, 0, 255), 5)
                cv2.line(frame, cd1, cd3, (0, 0, 255), 1)



            ####################################  DETECTION END  ##############################################
            ####################################


            ############# ICON DRAWINGS ###################################
            # img_path = '/Users/deepan/Desktop/ui/PySimpleGUI-YOLO-master/car.png'
            # watermark = cv2.imread(img_path, -1)
            # cv2.imshow('watermark', watermark)

            thickness = -1
            radius = 2
            (int((c6[0] + c7[0]) / 2), (int((c6[1] + c7[1]) / 2)))


            ########################### DISTANCE FROM MY CAR TO OBJECTS ######################
            Distance = int(c7[1] - pof2[1]) * -1
            if Distance < 0:
                DistanceVisibility = False
            elif Distance > 0:
                DistanceVisibility = True
            D2 = str(Distance)


            if LABELS[classIDs[i]] == 'Stop sign':
                cv2.line(frame, c6, ump1, (255, 0, 0), 1)
                cv2.line(frame, c6, ump2, (255, 0, 0), 1)
                cv2.line(frame, c7, ump3, (255, 0, 0), 1)
                cv2.line(frame, c7, ump4, (255, 0, 0), 1)
                cv2.line(frame, c6, c9, (255, 0, 0), 1)
                cv2.circle(frame, c9, radius, (255, 0, 0), 1)

                thickness = 2
                wheelThickness = 1
                # color = (255, 255, 255)

                cv2.line(frame, c6, ump1, Bbcolor, Bbthickness)
                cv2.line(frame, c6, ump2, Bbcolor, Bbthickness)
                cv2.line(frame, c7, ump3, Bbcolor, Bbthickness)
                cv2.line(frame, c7, ump4, Bbcolor, Bbthickness)

                if EnableArrowVisibility:
                    # diagnonal line pointing label
                    cv2.line(frame, c6, c9, Lpcolor, Lpthickness)


                    radius = 4
                    color = (255, 255, 255)
                    bcolor = (128, 128, 128)
                    thickness = -1

                    # circle for label
                    cv2.circle(frame, c9, Ldpradius, Ldpcolor, Ldpthickness)

            elif LABELS[classIDs[i]] == 'Bus' and EnableBus:
                contours = np.array([[t11],
                                     [t22],
                                     [t23],
                                     [t24],
                                     [t25],
                                     [t26],
                                     ])
                contoursTW2 = np.array([[tw11],
                                        [tw12],
                                        [tw13],
                                        [tw14]])

                contoursTW1 = np.array([[tw15],
                                        [tw16],
                                        [tw17],
                                        [tw18]])

                if EnableIconVisibility:
                    cv2.fillPoly(frame, pts=[contours], color=TruckColor)

                    cv2.fillPoly(frame, pts=[contoursTW2], color=(255, 255, 255))
                    cv2.fillPoly(frame, pts=[contoursTW1], color=(255, 255, 255))

                    cv2.circle(frame, tw1, 2 * TruckSize, TruckWheelOutColor, thickness)  # wheel out 1
                    cv2.circle(frame, tw2, 2 * TruckSize, TruckWheelOutColor, thickness)  # wheel out 2
                    cv2.circle(frame, tw1, 1 * TruckSize, TruckWheelInColor, thickness)  # wheel in 1
                    cv2.circle(frame, tw2, 1 * TruckSize, TruckWheelInColor, thickness)  # wheel in 2

                thickness = 2
                wheelThickness = 1
                # color = (255, 255, 255)

                if EnableBbVisibility:
                    cv2.line(frame, c6, mp1, Bbcolor, Bbthickness)
                    cv2.line(frame, c6, mp2, Bbcolor, Bbthickness)
                    cv2.line(frame, c7, mp3, Bbcolor, Bbthickness)
                    cv2.line(frame, c7, mp4, Bbcolor, Bbthickness)

                if EnableArrowVisibility:
                    # diagnonal line pointing label
                    cv2.line(frame, c6, c9, Lpcolor, Lpthickness)

                    radius = 4
                    color = (255, 255, 255)
                    bcolor = (128, 128, 128)
                    thickness = -1

                    # circle for label
                    cv2.circle(frame, c9, Ldpradius, Ldpcolor, Ldpthickness)

                if EnableLabelVisibility:
                    contourslblbg = np.array([[lbg1],
                                              [lbg2],
                                              [lbg3],
                                              [lbg4],
                                              [lbg5]])

                    cv2.fillPoly(frame, pts=[contourslblbg], color=Bbcolor)

                    cv2.putText(frame, 'Truck',
                                (c10[0] + LabelTextPositionx, int(c10[1] - LabelTextPositiony)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                LabelSize, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)
                # contoursBus = np.array([[b11],
                #                         [b22],
                #                         [b25],
                #                         [b26],
                #                         ])
                # cv2.circle(frame, bw1, 2 * BusSize, BusWheelOutColor, thickness)  # wheel out 1
                # cv2.circle(frame, bw2, 2 * BusSize, BusWheelOutColor, thickness)  # wheel out 2
                # cv2.circle(frame, bw1, 1 * BusSize, BusWheelInColor, thickness)  # wheel in 1
                # cv2.circle(frame, bw2, 1 * BusSize, BusWheelInColor, thickness)  # wheel in 2
                #
                # cv2.fillPoly(frame, pts=[contoursBus], color=BusColor)
                #
                # thickness = 2
                # wheelThickness = 1
                # # color = (255, 255, 255)
                #
                # cv2.line(frame, c6, ump1, Bbcolor, Bbthickness)
                # cv2.line(frame, c6, ump2, Bbcolor, Bbthickness)
                # cv2.line(frame, c7, ump3, Bbcolor, Bbthickness)
                # cv2.line(frame, c7, ump4, Bbcolor, Bbthickness)
                #
                # if EnableArrowVisibility:
                #     # diagnonal line pointing label
                #     cv2.line(frame, c6, c9, Lpcolor, Lpthickness)
                #
                #     radius = 4
                #     color = (255, 255, 255)
                #     bcolor = (128, 128, 128)
                #     thickness = -1
                #
                #     # circle for label
                #     cv2.circle(frame, c9, Ldpradius, Ldpcolor, Ldpthickness)
                # if EnableLabelVisibility:
                #     cv2.putText(frame, LABELS[classIDs[i]],
                #                 (c10[0] + LabelTextPositionx, int(c10[1] - LabelTextPositiony)),
                #                 cv2.FONT_HERSHEY_SIMPLEX,
                #                 LabelSize, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)


            elif LABELS[classIDs[i]] == 'Truck' and EnableTruck:

                # Truck
                contours = np.array([[t11],
                                     [t22],
                                     [t23],
                                     [t24],
                                     [t25],
                                     [t26],
                                     ])
                contoursTW2 = np.array([[tw11],
                                        [tw12],
                                        [tw13],
                                        [tw14]])

                contoursTW1 = np.array([[tw15],
                                        [tw16],
                                        [tw17],
                                        [tw18]])

                if EnableIconVisibility:
                    cv2.fillPoly(frame, pts=[contours], color=TruckColor)

                    cv2.fillPoly(frame, pts=[contoursTW2], color=(255, 255, 255))
                    cv2.fillPoly(frame, pts=[contoursTW1], color=(255, 255, 255))

                    cv2.circle(frame, tw1, 2 * TruckSize, TruckWheelOutColor, thickness)  # wheel out 1
                    cv2.circle(frame, tw2, 2 * TruckSize, TruckWheelOutColor, thickness)  # wheel out 2
                    cv2.circle(frame, tw1, 1 * TruckSize, TruckWheelInColor, thickness)  # wheel in 1
                    cv2.circle(frame, tw2, 1 * TruckSize, TruckWheelInColor, thickness)  # wheel in 2

                thickness = 2
                wheelThickness = 1
                # color = (255, 255, 255)

                if EnableBbVisibility:
                    cv2.line(frame, c6, mp1, Bbcolor, Bbthickness)
                    cv2.line(frame, c6, mp2, Bbcolor, Bbthickness)
                    cv2.line(frame, c7, mp3, Bbcolor, Bbthickness)
                    cv2.line(frame, c7, mp4, Bbcolor, Bbthickness)

                if EnableDistanceLine and DistanceVisibility:

                    cv2.line(frame, pof2, c7, Bbcolor, 1)
                    cv2.putText(frame, D2, (mp3[0] + 2, mp3[1]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                wmSize, Bbcolor, bbox_thick // 2, lineType=cv2.LINE_AA)

                    # ObjectCenter = (int((c6[0] + c7[0]) / 2), int((c6[1] + c7[1]) / 2))

                if EnableArrowVisibility:
                    # diagnonal line pointing label
                    cv2.line(frame, c6, c9, Lpcolor, Lpthickness)

                    radius = 4
                    color = (255, 255, 255)
                    bcolor = (128, 128, 128)
                    thickness = -1

                    # circle for label
                    cv2.circle(frame, c9, Ldpradius, Ldpcolor, Ldpthickness)

                if EnableLabelVisibility:
                    # cv2.circle(frame, lbg1, 2, Bbcolor, -1)
                    # cv2.circle(frame, lbg2, 2, Bbcolor, -1)
                    # cv2.circle(frame, lbg3, 2, Bbcolor, -1)
                    # cv2.circle(frame, lbg4, 2, Bbcolor, -1)
                    # cv2.circle(frame, lbg5, 2, Bbcolor, -1)

                    contourslblbg = np.array([[lbg1],
                                              [lbg2],
                                              [lbg3],
                                              [lbg4],
                                              [lbg5]])

                    cv2.fillPoly(frame, pts=[contourslblbg], color=Bbcolor)


                    cv2.putText(frame, 'Truck',
                                (c10[0] + LabelTextPositionx, int(c10[1] - LabelTextPositiony)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                LabelSize, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)
                if EnableLabelVisibilityT:
                    contourslblbg = np.array([[lbg1],
                                              [lbg2],
                                              [lbg3],
                                              [lbg4],
                                              [lbg5]])

                    cv2.fillPoly(frame, pts=[contourslblbg], color=Bbcolor)

                    cv2.putText(frame, 'Car',
                                (c10[0] + LabelTextPositionx, int(c10[1] - LabelTextPositiony)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                LabelSize, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)



            elif LABELS[classIDs[i]] == 'Car' and EnableCar:

                contours2 = np.array([[nc11],
                                      [nc22],
                                      [nc23],
                                      [nc24],
                                      [nc25],
                                      [nc26],
                                      [nc27],
                                      [nc28]])


                contoursWindow = np.array([[cw24],
                                      [cw25],
                                      [cw26],
                                      [cw27]])

                if EnableIconVisibility:
                    cv2.fillPoly(frame, pts=[contours2], color=CarColor)

                    cv2.fillPoly(frame, pts=[contoursWindow], color=(255, 255, 255))

                    cv2.circle(frame, ncw1, 2 * CarSize, CarWheelOutColor, thickness)  # wheel out 1
                    cv2.circle(frame, ncw2, 2 * CarSize, CarWheelOutColor, thickness)  # wheel out 2
                    cv2.circle(frame, ncw1, 1 * CarSize, CarWheelInColor, thickness)  # wheel in 1
                    cv2.circle(frame, ncw2, 1 * CarSize, CarWheelInColor, thickness)  # wheel in 2

                thickness = 2
                wheelThickness = 1
                # color = (255, 255, 255)
                if EnableBbVisibility:
                    cv2.line(frame, c6, mp1, Bbcolor, Bbthickness)
                    cv2.line(frame, c6, mp2, Bbcolor, Bbthickness)
                    cv2.line(frame, c7, mp3, Bbcolor, Bbthickness)
                    cv2.line(frame, c7, mp4, Bbcolor, Bbthickness)

                if EnableDistanceLine and DistanceVisibility:
                    cv2.line(frame, pof2, c7, Bbcolor, 1)
                    cv2.putText(frame, D2, (mp3[0] + 2, mp3[1]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                wmSize, Bbcolor, bbox_thick // 2, lineType=cv2.LINE_AA)

                    # ump3 = (int((c7[0] + mp3[0]) / 2), int((c7[1] + mp3[1]) / 2))


                if EnableArrowVisibility:
                    # diagnonal line pointing label
                    cv2.line(frame, c6, c9, Lpcolor, Lpthickness)

                    radius = 4
                    color = (255, 255, 255)
                    bcolor = (128, 128, 128)
                    thickness = -1

                    # circle for label
                    cv2.circle(frame, c9, Ldpradius, Ldpcolor, Ldpthickness)

                if EnableLabelVisibility:
                    contourslblbg = np.array([[lbg1],
                                              [lbg2],
                                              [lbg3],
                                              [lbg4],
                                              [lbg5]])

                    cv2.fillPoly(frame, pts=[contourslblbg], color=Bbcolor)

                    cv2.putText(frame, LABELS[classIDs[i]],
                                (c10[0] + LabelTextPositionx, int(c10[1] - LabelTextPositiony)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                LabelSize, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)
                if EnableLabelVisibilityT:
                    contourslblbg = np.array([[lbg1],
                                              [lbg2],
                                              [lbg3],
                                              [lbg4],
                                              [lbg5]])

                    cv2.fillPoly(frame, pts=[contourslblbg], color=Bbcolor)

                    cv2.putText(frame, 'Car',
                                (c10[0] + LabelTextPositionx, int(c10[1] - LabelTextPositiony)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                LabelSize, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)



            # elif LABELS[classIDs[i]] == 'Person' and EnablePerson:
            #     cv2.circle(frame, p3, 1 + personSize, PeopleHeadColor, thickness)
            #
            #     contours3 = np.array([[p1],
            #                           [p2],
            #                           [p3]])
            #
            #     if EnableIconVisibility:
            #         cv2.fillPoly(frame, pts=[contours3], color=PeopleBodyColor)
            #         cv2.line(frame, c6, ump1, Bbcolor, Bbthickness)
            #         cv2.line(frame, c6, ump2, Bbcolor, Bbthickness)
            #         cv2.line(frame, c7, ump3, Bbcolor, Bbthickness)
            #         cv2.line(frame, c7, ump4, Bbcolor, Bbthickness)
            #
            #     if EnableArrowVisibility:
            #         # diagnonal line pointing label
            #         cv2.line(frame, c6, c9, Lpcolor, Lpthickness)
            #
            #         radius = 4
            #         color = (255, 255, 255)
            #         bcolor = (128, 128, 128)
            #         thickness = -1
            #
            #         # circle for label
            #         cv2.circle(frame, c9, Ldpradius, Ldpcolor, Ldpthickness)
            #
            #     cv2.putText(frame, LABELS[classIDs[i]], (c10[0] + LabelTextPositionx, int(c10[1] - LabelTextPositiony)),
            #                 cv2.FONT_HERSHEY_SIMPLEX,
            #                 LabelSize, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)



            # elif LABELS[classIDs[i]] == 'Traffic light':
            #     # cv2.circle(frame, ts1, 4, (0, 0, 255), -1)
            #     # cv2.circle(frame, ts3, 4, (0, 255, 255), -1)
            #     # cv2.circle(frame, ts2, 4, (0, 128, 0), -1)
            #     #
            #     # thickness = 2
            #     # wheelThickness = 1
            #     # # color = (255, 255, 255)
            #     #
            #     # cv2.line(frame, c6, ump1, Bbcolor, Bbthickness)
            #     # cv2.line(frame, c6, ump2, Bbcolor, Bbthickness)
            #     # cv2.line(frame, c7, ump3, Bbcolor, Bbthickness)
            #     # cv2.line(frame, c7, ump4, Bbcolor, Bbthickness)
            #     #
            #     # if EnableArrowVisibility:
            #     #     # diagnonal line pointing label
            #     #     cv2.line(frame, c6, c9, Lpcolor, Lpthickness)
            #     #
            #     #     radius = 4
            #     #     color = (255, 255, 255)
            #     #     bcolor = (128, 128, 128)
            #     #     thickness = -1
            #     #
            #     #     # circle for label
            #     #     cv2.circle(frame, c9, Ldpradius, Ldpcolor, Ldpthickness)
            #     #
            #     #     cv2.putText(frame, LABELS[classIDs[i]],
            #     #                 (c10[0] + LabelTextPositionx, int(c10[1] - LabelTextPositiony)),
            #     #                 cv2.FONT_HERSHEY_SIMPLEX,
            #     #                 LabelSize, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

            elif LABELS[classIDs[i]] == 'Bicycle':
                cv2.circle(frame, bm3, 2 * bikeSize, color, wheelThickness)
                cv2.circle(frame, bm4, 2 * bikeSize, color, wheelThickness)
                cv2.circle(frame, bm3, 1 * bikeSize, bcolor, wheelThickness)  # inside wheel 1
                cv2.circle(frame, bm4, 1 * bikeSize, bcolor, wheelThickness)  # inside wheel 2
                # cv2.circle(frame,bm3,4,color,thickness)
                cv2.line(frame, bm1, bm2, color, 1)

                thickness = 2
                wheelThickness = 1
                # color = (255, 255, 255)

                cv2.line(frame, c6, ump1, Bbcolor, Bbthickness)
                cv2.line(frame, c6, ump2, Bbcolor, Bbthickness)
                cv2.line(frame, c7, ump3, Bbcolor, Bbthickness)
                cv2.line(frame, c7, ump4, Bbcolor, Bbthickness)

                if EnableArrowVisibility:
                    # diagnonal line pointing label
                    cv2.line(frame, c6, c9, Lpcolor, Lpthickness)

                    radius = 4
                    color = (255, 255, 255)
                    bcolor = (128, 128, 128)
                    thickness = -1

                    # circle for label
                    cv2.circle(frame, c9, Ldpradius, Ldpcolor, Ldpthickness)

            # elif LABELS[classIDs[i]] == 'Motorbike':
            #     cv2.circle(frame, bm1, 4, color, thickness)
            #     cv2.circle(frame, bm2, 4, color, thickness)
            #     cv2.circle(frame, bm1, 3, bcolor, thickness)  # inside wheel 1
            #     cv2.circle(frame, bm2, 3, bcolor, thickness)  # inside wheel 2
            #     # cv2.circle(frame,bm3,4,color,thickness)
            #     cv2.line(frame, bm3, bm4, color, 1)
            #
            #     thickness = 2
            #     wheelThickness = 1
            #     # color = (255, 255, 255)
            #
            #     cv2.line(frame, c6, ump1, Bbcolor, Bbthickness)
            #     cv2.line(frame, c6, ump2, Bbcolor, Bbthickness)
            #     cv2.line(frame, c7, ump3, Bbcolor, Bbthickness)
            #     cv2.line(frame, c7, ump4, Bbcolor, Bbthickness)
            #
            #     if EnableArrowVisibility:
            #         # diagnonal line pointing label
            #         cv2.line(frame, c6, c9, Lpcolor, Lpthickness)
            #
            #         radius = 4
            #         color = (255, 255, 255)
            #         bcolor = (128, 128, 128)
            #         thickness = -1
            #
            #         # circle for label
            #         cv2.circle(frame, c9, Ldpradius, Ldpcolor, Ldpthickness)

            elif LABELS[classIDs[i]] == 'Dog' and EnableAnimal:

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

                thickness = 2
                wheelThickness = 1
                # color = (255, 255, 255)

                cv2.line(frame, c6, ump1, Bbcolor, Bbthickness)
                cv2.line(frame, c6, ump2, Bbcolor, Bbthickness)
                cv2.line(frame, c7, ump3, Bbcolor, Bbthickness)
                cv2.line(frame, c7, ump4, Bbcolor, Bbthickness)

                if EnableArrowVisibility:
                    # diagnonal line pointing label
                    cv2.line(frame, c6, c9, Lpcolor, Lpthickness)

                    radius = 4
                    color = (255, 255, 255)
                    bcolor = (128, 128, 128)
                    thickness = -1

                    # circle for label
                    cv2.circle(frame, c9, Ldpradius, Ldpcolor, Ldpthickness)

                    cv2.putText(frame, LABELS[classIDs[i]],
                                (c10[0] + LabelTextPositionx, int(c10[1] - LabelTextPositiony)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, labelColor, bbox_thick // 2, lineType=cv2.LINE_AA)

                # contoursdogMouth = np.array([[d23],
                #                              [d26],
                #                              [d27],
                #                            ])

                # cv2.fillPoly(frame, pts =[contoursdogMouth], color=(255,255,0))


            ################################# Drawing starts from here  ################################################

            ###################### NEW BOUNDARY BOX DRAWING #############################
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) ----> default boundary box

            # thickness = 2
            # wheelThickness = 1
            # # color = (255, 255, 255)
            #
            # cv2.line(frame, c6, ump1, Bbcolor, Bbthickness)
            # cv2.line(frame, c6, ump2, Bbcolor, Bbthickness)
            # cv2.line(frame, c7, ump3, Bbcolor, Bbthickness)
            # cv2.line(frame, c7, ump4, Bbcolor, Bbthickness)
            #
            # if EnableArrowVisibility:
            #     # diagnonal line pointing label
            #     cv2.line(frame, c6, c9, Lpcolor, Lpthickness)
            #
            #     radius = 4
            #     color = (255, 255, 255)
            #     bcolor = (128, 128, 128)
            #     thickness = -1
            #
            #     # circle for label
            #     cv2.circle(frame, c9, Ldpradius, Ldpcolor, Ldpthickness)

            CarSize = newC
            LabelSize = LabelSizeBackup
            TruckSize = newT
            EnableIconVisibility = EnableIconVisibilityBackup
            EnableArrowVisibility = EnableArrowVisibilityBackup
            EnableLabelVisibility = EnableLabelVisibilityBackup
            EnableLabelVisibilityT = EnableLabelVisibilityBackup
            EnableBbVisibility = EnableBbVisibilityBackup
            EnableDistanceLine = EnableDistanceLineBackup
            EnableCar = EnableCarBackup
            EnableTruck = EnableTruckBackup
            EnableBus = EnableTruckBackup

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
        EnableArrowVisibility
        layout = [
            # [sg.Image("/Users/deepan/Desktop/ui/PySimpleGUI-YOLO-master/images/logo.png", key="imageContainer")]
            [sg.Text('Roadhow', size=(30, 1))],
            [sg.Image(data=imgbytes, key='_IMAGE_')],
            [sg.Text('Confidence'),
             sg.Slider(range=(0, 10), orientation='h', resolution=1, default_value=5, size=(15, 15), key='confidence'),
             sg.Text('              Icon Position (x,y) '), sg.In(iconPositionx, size=(5, 1), text_color='#8B0000', key='ipxrt'),
             sg.In(iconPositiony, size=(5, 1), text_color='#8B0000', key='ipyrt'),
             sg.Text('       Label Position (x,y) '),
             sg.In(LabelTextPositionx, size=(5, 1), text_color='#8B0000', key='lpxrt'),
             sg.In(LabelTextPositiony, size=(5, 1), text_color='#8B0000', key='lpyrt'),
             sg.Text('Label arrow Distance'), sg.In(c9Postx, size=(5, 1), text_color='#8B0000', key='_c9Postxrt'),
             sg.In(c9Posty, size=(5, 1), text_color='#8B0000', key='_c9Postyrt'),
            ],
            [sg.Text(' Layer 1'), sg.Checkbox('Icon', text_color='#8B0000', default=Layer1I, key='l1Irt'),
             sg.Checkbox('Label', text_color='#8B0000', default=Layer1L, key='l1Lrt'),
             sg.Checkbox('Arrow', text_color='#8B0000', default=Layer1A, key='l1Art'),
             sg.Checkbox('BoundaryBox', text_color='#8B0000', default=Layer1B, key='l1Brt'),
             sg.Checkbox('Distance line', text_color='#8B0000', default=Layer1D, key='l1Drt'),
             sg.Checkbox('c', text_color='#8B0000', default=Layer1T, key='l1Trt'),

             sg.Text('                  Layer 2'), sg.Checkbox('Icon', text_color='#8B0000', default=Layer2I, key='l2Irt'),
             sg.Checkbox('Label', text_color='#8B0000', default=Layer2L, key='l2Lrt'),
             sg.Checkbox('Arrow', text_color='#8B0000', default=Layer2A, key='l2Art'),
             sg.Checkbox('BoundaryBox', text_color='#8B0000', default=Layer2B, key='l2Brt'),
             sg.Checkbox('Distance line', text_color='#8B0000', default=Layer2D, key='l2Drt'),
             sg.Checkbox('c', text_color='#8B0000', default=Layer2T, key='l2Trt'),

             sg.Text('                   Layer 3'), sg.Checkbox('Icon', text_color='#8B0000', default=Layer3I, key='l3Irt'),
             sg.Checkbox('Label', text_color='#8B0000', default=Layer3L, key='l3Lrt'),
             sg.Checkbox('Arrow', text_color='#8B0000', default=Layer3A, key='l3Art'),
             sg.Checkbox('BoundaryBox', text_color='#8B0000', default=Layer3B, key='l3Brt'),
             sg.Checkbox('Distance line', text_color='#8B0000', default=Layer3D, key='l3Drt'),
             sg.Checkbox('c', text_color='#8B0000', default=Layer3T, key='l3Trt'),

             sg.Text('                  Directional Visibility:'),
             sg.Checkbox('Left', text_color='#8B0000', default=LV, key='lvrt'),
             sg.Checkbox('Right', text_color='#8B0000', default=RV, key='rvrt'),

             ],
            [sg.Text('Boundary Box Color 1'), sg.In(bbb1, size=(5, 1), text_color='#8B0000', key='_bbb1rt_'),
             sg.In(bbg1, size=(5, 1), text_color='#8B0000', key='_bbg1rt_'),
             sg.In(bbr1, size=(5, 1), text_color='#8B0000', key='_bbr1rt_'),  # = Bbcolor1

             sg.Text('      Boundary Box Color 2'), sg.In(bbb2, size=(5, 1), text_color='#8B0000', key='_bbb2rt_'),
             sg.In(bbg2, size=(5, 1), text_color='#8B0000', key='_bbg2rt_'),
             sg.In(bbr2, size=(5, 1), text_color='#8B0000', key='_bbr2rt_'),  # = Bbcolor2

             sg.Text('      Boundary Box Color 3'), sg.In(bbb3, size=(5, 1), text_color='#8B0000', key='_bbb3rt_'),
             sg.In(bbg3, size=(5, 1), text_color='#8B0000', key='_bbg3rt_'),
             sg.In(bbr3, size=(5, 1), text_color='#8B0000', key='_bbr3rt_'),  # = Bbcolor3
            ],
            [ sg.Text('Label Color 1 '), sg.In(lc1a, size=(5, 1), text_color='#8B0000', key='_lc1art'),
              sg.In(lc2a, size=(5, 1), text_color='#8B0000', key='_lc2art'),
              sg.In(lc3a, size=(5, 1), text_color='#8B0000', key='_lc3art'),

              sg.Text('Label Color 2 '), sg.In(lc1b, size=(5, 1), text_color='#8B0000', key='_lc1brt'),
              sg.In(lc2b, size=(5, 1), text_color='#8B0000', key='_lc2brt'),
              sg.In(lc3b, size=(5, 1), text_color='#8B0000', key='_lc3brt'),

              sg.Text('Label Color 3 '), sg.In(lc1c, size=(5, 1), text_color='#8B0000', key='_lc1crt'),
              sg.In(lc2c, size=(5, 1), text_color='#8B0000', key='_lc2crt'),
              sg.In(lc3c, size=(5, 1), text_color='#8B0000', key='_lc3crt'),

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
    iconPositionx = int(values['ipxrt'])
    iconPositiony = int(values['ipyrt'])
    LabelTextPositionx = int(values['lpxrt'])
    LabelTextPositiony = int(values['lpyrt'])
    c9Postx = int(values['_c9Postxrt'])
    c9Posty = int(values['_c9Postyrt'])

    bbb1 = int(values['_bbb1rt_'])
    bbg1 = int(values['_bbg1rt_'])
    bbr1 = int(values['_bbr1rt_'])
    Bbcolor1 = (bbb1, bbg1, bbr1)

    bbb2 = int(values['_bbb2rt_'])
    bbg2 = int(values['_bbg2rt_'])
    bbr2 = int(values['_bbr2rt_'])
    Bbcolor2 = (bbb2, bbg2, bbr2)

    lc1a = int(values['_lc1art'])
    lc2a = int(values['_lc2art'])
    lc3a = int(values['_lc3art'])
    labelColora = (lc1a, lc2a, lc3a)

    lc1b = int(values['_lc1brt'])
    lc2b = int(values['_lc2brt'])
    lc3b = int(values['_lc3brt'])
    labelColorb = (lc1b, lc2b, lc3b)

    lc1c = int(values['_lc1crt'])
    lc2c = int(values['_lc2crt'])
    lc3c = int(values['_lc3crt'])
    labelColorc = (lc1c, lc2c, lc3c)

    bbb3 = int(values['_bbb3rt_'])
    bbg3 = int(values['_bbg3rt_'])
    bbr3 = int(values['_bbr3rt_'])
    Bbcolor3 = (bbb3, bbg3, bbr3)

    Layer1I = values['l1Irt']
    Layer1L = values['l1Lrt']
    Layer1A = values['l1Art']
    Layer1B = values['l1Brt']
    Layer1D = values['l1Drt']
    Layer1T = values['l1Trt']


    Layer2I = values['l2Irt']
    Layer2L = values['l2Lrt']
    Layer2A = values['l2Art']
    Layer2B = values['l2Brt']
    Layer2D = values['l2Drt']
    Layer2T = values['l2Trt']

    Layer3I = values['l3Irt']
    Layer3L = values['l3Lrt']
    Layer3A = values['l3Art']
    Layer3B = values['l3Brt']
    Layer3D = values['l3Drt']
    Layer3T = values['l3Trt']

    LV = values['lvrt']
    RV = values['rvrt']
    gui_threshold = 3/ 10

win.Close()

# release the file pointers
print("[INFO] cleaning up...")
writer.release() if writer is not None else None
vs.release()
