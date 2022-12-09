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
from pathlib import Path
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
k = 13

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
print(full_maml_parameters_path)
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

STIMULI_NAME_COLUMN = 'SourceStimuliName'
PRE_CALIBRATION = 'Pre-study calibration'
POST_CALIBRATION = 'Post-study calibration'

CALIBRATION_EVENTS_PATH = './calibration/Native_CalibrationEvents.csv'
PRE_CALIBRATION_VIDEO_PATH = './calibration/Pre-study calibration.webm'
POST_CALIBRATION_VIDEO_PATH = './calibration/Post-study calibration.webm'

TRACKING_VIDEO_PATH = []
# TRACKING_VIDEO_PATH.append('./calibration/Pre-study calibration.webm')
TRACKING_VIDEO_PATH.append('./calibration/Post-study calibration.webm')
# TRACKING_VIDEO_PATH.append('./calibration/SandalCat emoji.webm')
# TRACKING_VIDEO_PATH.append('./calibration/ShyCat emoji.webm')
# TRACKING_VIDEO_PATH.append('./calibration/WashingCat emoji.webm')
# TRACKING_VIDEO_PATH.append('./calibration/WavingCat emoji.webm')
# TRACKING_VIDEO_PATH.append('./calibration/WiggleCat emoji.webm')

# Initialize monitor and frame processor
monitor = Monitor()
processor = FrameProcessor(camera_calibration)
calibration = PersonCalibration(monitor, processor)

#################################
# CALIBRATION DATA PREPARATION
#################################

def read_callibration_data(calibration_events: pd.DataFrame) -> pd.DataFrame:
    """
    # calibration video
    cap = cv2.VideoCapture(PRE_CALIBRATION_VIDEO_PATH)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print('Can not open the calibration video')
        sys.exit(-1)
    # collect calibration data from the video
    pre_calibration_data = calibration.collect_data(cap, calibration_events.loc[calibration_events[STIMULI_NAME_COLUMN] == PRE_CALIBRATION])

    return pre_calibration_data
    """
    # calibration video
    cap = cv2.VideoCapture(POST_CALIBRATION_VIDEO_PATH)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print('Can not open the calibration video')
        sys.exit(-1)
    # collect calibration data from the video
    post_calibration_data = calibration.collect_data(cap, calibration_events.loc[calibration_events[STIMULI_NAME_COLUMN] == POST_CALIBRATION])
    
    return post_calibration_data

    return pd.concat([pre_calibration_data, post_calibration_data], axis=0, ignore_index=True)

# calibration events
calibration_events = pd.read_csv(CALIBRATION_EVENTS_PATH, float_precision='round_trip')

#################################
# TRAINING ITERATION 1 (PRE)
#################################
print('####### TRAINING ITERATION 1: CALIBRATION #######')

# find the monitor screen-to-camera transform
# updates monitor transforms
calibration_data = read_callibration_data(calibration_events)
calibration_data = processor.process_monitor(calibration_data, monitor, device, gaze_network)

# fine-tune gaze network
# adjust steps and lr for best results
# To debug calibration, set show=True
gaze_network = calibration.fine_tune(calibration_data,
                                     device,
                                     gaze_network,
                                     k,
                                     steps=2000,
                                     lr=1e-4)
"""
#################################
# TRAINING ITERATION 2
#################################
print('####### TRAINING ITERATION 2: CALIBRATION #######')

# find the monitor screen-to-camera transform
# updates monitor transforms
calibration_data = read_callibration_data(calibration_events)
calibration_data = processor.process_monitor(calibration_data, monitor, device, gaze_network)

# fine-tune gaze network
# adjust steps and lr for best results
# To debug calibration, set show=True
gaze_network = calibration.fine_tune(calibration_data,
                                     device,
                                     gaze_network,
                                     k,
                                     steps=2000,
                                     lr=1e-4)

#################################
# TRAINING ITERATION 3
#################################
print('####### TRAINING ITERATION 3: CALIBRATION #######')

# fine-tune gaze network
# adjust steps and lr for best results
# To debug calibration, set show=True
gaze_network = calibration.fine_tune(calibration_data,
                                     device,
                                     gaze_network,
                                     k,
                                     steps=2000,
                                     lr=1e-5)
"""

#################################
# Run on live webcam feed and
# show point of regard on screen
#################################
print('####### INFERENCE #######')

for path in TRACKING_VIDEO_PATH:
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        data = processor.process_video(cap, monitor, device, gaze_network, convert_to_monitor=True)
    else:
        print("Can not open video file")

    print(data)

    if data is not None:
        gaze_path = Path(path).with_suffix('.gaze')
        with open(gaze_path, 'w') as of:
            of.write('timestamp,x,y,z\n')
            for _, row in data.iterrows():
                of.write(f"{row['timestamp']},{row['x']},{row['y']},{0}\n")
