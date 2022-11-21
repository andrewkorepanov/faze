#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

from subprocess import call
import cv2
import numpy as np
from os import path

from camera import cam_calibrate


#################################
# Start camera
#################################

calibration_path = './calibration/pattern.mp4'

cap = cv2.VideoCapture(calibration_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if cap.isOpened():
    # calibrate camera
    calib = {'mtx': np.eye(3), 'dist': np.zeros((1, 5))}
    print("Calibrate camera once.")
    cam_calibrate(cap, calib)
else:
    print(f"Error opening video file {calibration_path}")

cap.release()