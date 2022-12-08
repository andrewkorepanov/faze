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
from monitor import Monitor
from frame_processor import FrameProcessor
from pathlib import Path

import torch

sys.path.append("../src")
from losses import GazeAngularLoss
from losses import GazeMSELoss

CUTOFF_TIME = 1000

class PersonCalibration:
    """Makes person calibration and the network fine tune"""

    def __init__(self, monitor: Monitor, processor: FrameProcessor) -> None:

        self._monitor = monitor
        self._processor = processor

    def collect_data(self, cap: cv2.VideoCapture, events: pd.DataFrame) -> pd.DataFrame:
        """Collect clibration frames along with the marker points"""

        calibration_data = {'frame': [], 'marker_x': [], 'marker_y': []}

        timestamp = 0
        for _, row in events.iterrows():
            # raw data
            start_time, end_time = row['StartTimestamp'] + CUTOFF_TIME, row['EndTimestamp'] - CUTOFF_TIME
            # x, y, _ = self._monitor.monitor_to_camera(row['Left'], row['Top'])
            x, y = row['Left'], row['Top']

            print(f'START: {start_time}, END: {end_time}')

            while timestamp < end_time:
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                if timestamp > start_time:
                    calibration_data['frame'].append(frame)
                    calibration_data['marker_x'].append(x)
                    calibration_data['marker_y'].append(y)

        return pd.DataFrame(calibration_data)

    def fine_tune(self, calibration_data: pd.DataFrame, device, gaze_network, k, steps=1000, lr=1e-4):
        """Makes a few-shot training on the calibration data"""

        # process person calibration data
        data: pd.DataFrame = self._processor.process_calibration(calibration_data)

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
            img[i, :, :, :] = data.loc[i,'image_a']
            gaze_a[i, :] = data.loc[i,'gaze_a']
            head_a[i, :] = data.loc[i,'head_a']
            R_gaze_a[i, :, :] = data.loc[i,'R_gaze_a']
            R_head_a[i, :, :] = data.loc[i,'R_head_a']

        # create data subsets
        step = n // k
        train_indices = []
        for i in range(0, k * step, step):
            train_indices.append(random.sample(range(i, i + step), 12))
        train_indices = sum(train_indices, [])
        
        print('TRAIN INDICES: ', train_indices)

        valid_indices = []
        for i in range(0, k * step, step):
            valid_indices.append(random.sample(range(i, i + step), 3))
        valid_indices = sum(valid_indices, [])

        for i in train_indices:
            image = (img[i,:,:,:] + 1) * 255 / 2
            image = np.transpose(image, [1, 2, 0])  # CxHxW
            cv2.imwrite(f'./output/{i}.png', image)

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
            [
                p for n, p in gaze_network.named_parameters()
                if n.startswith('gaze')
            ],
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
                      (i + 1, train_loss.item(), valid_loss.item()))
        torch.save(gaze_network.state_dict(), 'gaze_network.pth.tar')
        torch.cuda.empty_cache()

        return gaze_network