#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import time
import cv2
import numpy as np
from os import path
from subprocess import call
import pickle
import sys
import torch
import os

import warnings
warnings.filterwarnings("ignore")

from monitor import monitor
from camera import cam_calibrate
from person_calibration import collect_data, fine_tune
from frame_processor import frame_processer

#################################
# Start camera
#################################


# calibrate camera
cam_calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
if path.exists("calib_cam0.pkl"):
    cam_calib = pickle.load(open("calib_cam0.pkl", "rb"))
else:
    sys.exit('ERR: No calibration file!')

#################################
# Load gaze network
#################################
ted_parameters_path = 'demo_weights/weights_ted.pth.tar'
maml_parameters_path = 'demo_weights/weights_maml'
k = 9

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create network
sys.path.append("../src")
from models import DTED
gaze_network = DTED(
    growth_rate=32,
    z_dim_app=64,
    z_dim_gaze=2,
    z_dim_head=16,
    decoder_input_c=32,
    normalize_3d_codes=True,
    normalize_3d_codes_axis=1,
    backprop_gaze_to_encoder=False,
).to(device)

#################################

# Load T-ED weights if available
assert os.path.isfile(ted_parameters_path)
print('> Loading: %s' % ted_parameters_path)
ted_weights = torch.load(ted_parameters_path)
if torch.cuda.device_count() == 1:
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])

#####################################

# Load MAML MLP weights if available
full_maml_parameters_path = maml_parameters_path +'/%02d.pth.tar' % k
assert os.path.isfile(full_maml_parameters_path)
print('> Loading: %s' % full_maml_parameters_path)
maml_weights = torch.load(full_maml_parameters_path)
ted_weights.update({  # rename to fit
    'gaze1.weight': maml_weights['layer01.weights'],
    'gaze1.bias':   maml_weights['layer01.bias'],
    'gaze2.weight': maml_weights['layer02.weights'],
    'gaze2.bias':   maml_weights['layer02.bias'],
})
gaze_network.load_state_dict(ted_weights)

#################################
# Personalize gaze network
#################################

# Initialize monitor and frame processor
mon = monitor()
frame_processor = frame_processer(cam_calib)

# collect person calibration data and fine-
# tune gaze network
data = collect_data(mon)
# adjust steps and lr for best results
# To debug calibration, set show=True
gaze_network = fine_tune(data, frame_processor, mon, device, gaze_network, k, steps=1000, lr=1e-5, show=False)

#################################
# Run on live webcam feed and
# show point of regard on screen
#################################
# open calibration file
video_path = './calibration/video.webm'

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if cap.isOpened():
    data = frame_processor.process_video(cap, mon, device, gaze_network, show=True)
else:
    print("Can not open video file")
