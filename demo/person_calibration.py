#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import cv2
import numpy as np
import pandas as pd
import random
import threading
import pickle
import sys

import torch
sys.path.append("../src")
from losses import GazeAngularLoss

directions = ['l', 'r', 'u', 'd']
keys = {'u': 82,
        'd': 84,
        'l': 81,
        'r': 83}

global THREAD_RUNNING
global frames

def create_image(mon, direction, i, color, target='E', grid=True, total=9):

    h = mon.h_pixels
    w = mon.w_pixels
    if grid:
        if total == 9:
            row = i % 3
            col = int(i / 3)
            x = int((0.02 + 0.48 * row) * w)
            y = int((0.02 + 0.48 * col) * h)
        elif total == 16:
            row = i % 4
            col = int(i / 4)
            x = int((0.05 + 0.3 * row) * w)
            y = int((0.05 + 0.3 * col) * h)
    else:
        x = int(random.uniform(0, 1) * w)
        y = int(random.uniform(0, 1) * h)

    # compute the ground truth point of regard
    x_cam, y_cam, z_cam = mon.monitor_to_camera(x, y)
    g_t = (x_cam, y_cam)

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = np.ones((h, w, 3), np.float32)
    if direction == 'r' or direction == 'l':
        if direction == 'r':
            cv2.putText(img, target, (x, y), font, 0.5, color, 2, cv2.LINE_AA)
        elif direction == 'l':
            cv2.putText(img, target, (w - x, y), font, 0.5, color, 2, cv2.LINE_AA)
            img = cv2.flip(img, 1)
    elif direction == 'u' or direction == 'd':
        imgT = np.ones((w, h, 3), np.float32)
        if direction == 'd':
            cv2.putText(imgT, target, (y, x), font, 0.5, color, 2, cv2.LINE_AA)
        elif direction == 'u':
            cv2.putText(imgT, target, (h - y, x), font, 0.5, color, 2, cv2.LINE_AA)
            imgT = cv2.flip(imgT, 1)
        img = imgT.transpose((1, 0, 2))

    return img, g_t


def collect_data(mon) -> dict:

    CALIBRATION_EVENTS_PATH = './calibration/calibration.csv'
    CALIBRATION_VIDEO_PATH = './calibration/calibration.webm'

    calib_data = {'frames': [], 'g_t': []}

    # calibration events
    calibration_events = pd.read_csv(CALIBRATION_EVENTS_PATH, float_precision='round_trip')
    print(calibration_events)

    # open calibration video
    cap = cv2.VideoCapture(CALIBRATION_VIDEO_PATH)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print('Can not open the calibration video')
        return calib_data

    timestamp = 0

    for i, row in calibration_events.iterrows():
        # raw data
        start_time, end_time = row['StartTimestamp'], row['EndTimestamp'] 
        x, y = row['Left'], row['Top'] 
        # in the camera coordinate system
        x_cam, y_cam, z_cam = mon.monitor_to_camera(x, y)
        g_t = (x_cam, y_cam)

        print(f'START: {start_time}, END: {end_time}')

        frames = []
        while timestamp < end_time: 
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp > start_time:
                frames.append(frame)

        if len(frames) > 0:
            calib_data['frames'].append(frames)
            calib_data['g_t'].append(g_t)
        
    return calib_data


def fine_tune(data, frame_processor, mon, device, gaze_network, k, steps=1000, lr=1e-4, show=False):

    # collect person calibration data

    data = frame_processor.process_calibration(data, mon, device, gaze_network, por_available=True, show=show)

    n = len(data['image_a'])
    print('N: ', n)
    #assert n==130, "Face not detected correctly. Collect calibration data again."
    _, c, h, w = data['image_a'][0].shape
    img = np.zeros((n, c, h, w))
    gaze_a = np.zeros((n, 2))
    head_a = np.zeros((n, 2))
    R_gaze_a = np.zeros((n, 3, 3))
    R_head_a = np.zeros((n, 3, 3))
    for i in range(n):
        img[i, :, :, :] = data['image_a'][i]
        gaze_a[i, :] = data['gaze_a'][i]
        head_a[i, :] = data['head_a'][i]
        R_gaze_a[i, :, :] = data['R_gaze_a'][i]
        R_head_a[i, :, :] = data['R_head_a'][i]

    # create data subsets
    train_indices = []
    for i in range(0, k*10, 10):
        train_indices.append(random.sample(range(i, i + 10), 3))
    train_indices = sum(train_indices, [])

    valid_indices = []
    for i in range(k*10, n - 10, 10):
        valid_indices.append(random.sample(range(i, i + 10), 1))
    valid_indices = sum(valid_indices, [])

    input_dict_train = {
        'image_a': img[train_indices, :, :, :],
        'gaze_a': gaze_a[train_indices, :],
        'head_a': head_a[train_indices, :],
        'R_gaze_a': R_gaze_a[train_indices, :, :],
        'R_head_a': R_head_a[train_indices, :, :],
    }

    input_dict_valid = {
        'image_a': img[valid_indices, :, :, :],
        'gaze_a': gaze_a[valid_indices, :],
        'head_a': head_a[valid_indices, :],
        'R_gaze_a': R_gaze_a[valid_indices, :, :],
        'R_head_a': R_head_a[valid_indices, :, :],
    }

    for d in (input_dict_train, input_dict_valid):
        for k, v in d.items():
            d[k] = torch.FloatTensor(v).to(device).detach()

    #############
    # Finetuning
    #################

    loss = GazeAngularLoss()
    optimizer = torch.optim.SGD(
        [p for n, p in gaze_network.named_parameters() if n.startswith('gaze')],
        lr=lr,
    )

    gaze_network.eval()
    output_dict = gaze_network(input_dict_valid)
    valid_loss = loss(input_dict_valid, output_dict).cpu()
    print('%04d> , Validation: %.2f' % (0, valid_loss.item()))

    for i in range(steps):
        # zero the parameter gradient
        gaze_network.train()
        optimizer.zero_grad()

        # forward + backward + optimize
        output_dict = gaze_network(input_dict_train)
        train_loss = loss(input_dict_train, output_dict)
        train_loss.backward()
        optimizer.step()

        if i % 100 == 99:
            gaze_network.eval()
            output_dict = gaze_network(input_dict_valid)
            valid_loss = loss(input_dict_valid, output_dict).cpu()
            print('%04d> Train: %.2f, Validation: %.2f' %
                  (i+1, train_loss.item(), valid_loss.item()))
    torch.save(gaze_network.state_dict(), 'gaze_network.pth.tar')
    torch.cuda.empty_cache()

    return gaze_network