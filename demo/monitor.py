#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import numpy as np

class Monitor:

    def __init__(self):
        # default screen 13.3": 11.3" x 7.1", 1 inch = 25.4 mm
        self.h_mm = 7.1 * 25.4
        self.w_mm = 11.3 * 25.4

        self.h_pixels = 1080
        self.w_pixels = 1920

    def set_transform(self, monitor_to_camera_matrix: np.array):
        # right transform matrix: x * M = y
        self._monitor_to_camera = monitor_to_camera_matrix
        self._camera_to_monitor = np.linalg.inv(monitor_to_camera_matrix)
        print(self._monitor_to_camera)
        print(self._camera_to_monitor)

    def monitor_to_camera(self, x, y):

        coords = np.matmul(np.array([x, y, 1]), self._monitor_to_camera)
        return coords[0], coords[1], 1

    def camera_to_monitor(self, x, y):

        coords = np.matmul(np.array([x, y, 1]), self._camera_to_monitor)
        return coords[0], coords[1]