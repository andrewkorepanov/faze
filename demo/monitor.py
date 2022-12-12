#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import numpy as np
from scipy.interpolate import RBFInterpolator

class Monitor:

    def __init__(self):
        # default screen 13.3": 11.3" x 7.1", 1 inch = 25.4 mm
        self.h_mm = 7.1 * 25.4
        self.w_mm = 11.3 * 25.4

        self.h_pixels = 1080
        self.w_pixels = 1920

    def set_transform(self, monitor_to_camera: RBFInterpolator,
                      camera_to_monitor: RBFInterpolator):
        # right transform matrix: x * M = y
        self._monitor_to_camera = monitor_to_camera
        self._camera_to_monitor = camera_to_monitor

    def monitor_to_camera(self, gazes:np.array) -> np.array:
        return self._monitor_to_camera(gazes)

    def camera_to_monitor(self, markers:np.array) -> np.array:
        return self._camera_to_monitor(markers)
