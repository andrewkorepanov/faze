#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import time
import cv2
import numpy as np
import pandas as pd
from os import path
import pickle
import sys
import torch
import os

import warnings

warnings.filterwarnings("ignore")

from monitor import Monitor
from person_calibration import PersonCalibration
from frame_processor import FrameProcessor

#################################
# Start camera
#################################

# calibrate camera
camera_calibration = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
if path.exists("calib_cam0.pkl"):
    camera_calibration = pickle.load(open("calib_cam0.pkl", "rb"))
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
full_maml_parameters_path = maml_parameters_path + '/%02d.pth.tar' % k
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

CALIBRATION_EVENTS_PATH = './calibration/calibration.csv'
CALIBRATION_VIDEO_PATH = './calibration/calibration.webm'

TRACKING_VIDEO_PATH = './calibration/video.webm'

# Initialize monitor and frame processor
monitor = Monitor()
processor = FrameProcessor(camera_calibration)
calibration = PersonCalibration(monitor, processor)

# collect person calibration data
# calibration events
calibration_events = pd.read_csv(CALIBRATION_EVENTS_PATH, float_precision='round_trip')
# calibration video
cap = cv2.VideoCapture(CALIBRATION_VIDEO_PATH)
if not cap.isOpened():
    print('Can not open the calibration video')
    sys.exit(-1)

calibration_data = calibration.collect_data(cap, calibration_events)

# find the monitor screen-to-camera transform

# fine-tune gaze network
# adjust steps and lr for best results
# To debug calibration, set show=True
gaze_network = calibration.fine_tune(calibration_data,
                                     device,
                                     gaze_network,
                                     k,
                                     steps=1000,
                                     lr=1e-5)

#################################
# Run on live webcam feed and
# show point of regard on screen
#################################

cap = cv2.VideoCapture(TRACKING_VIDEO_PATH)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if cap.isOpened():
    data = processor.process_video(cap, monitor, device, gaze_network)
else:
    print("Can not open video file")

print(data)
# if data is not None:
# with open('./output/Pre-study calibration.gaze', 'w') as of:
# of.write('timestamp,x,y,z\n')
# if gaze is not None:
# of.write(
# f'{timestamp},{g_t[0] / 2.0},{g_t[1] / 2.0},{0}\n')
# else:
# of.write(f'{timestamp},,,\n')
