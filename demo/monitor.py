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

    def set_transform(self, rotation: np.array,
                      translation: np.array):
        # right transform matrix: x * M = y
        self._rotation = rotation
        self._translation = translation
        self._inv_rotation = np.linalg.inv(rotation)
        self._n = np.squeeze(self._rotation[:,2])
        print('Normal: ', self._n)

    def camera_to_monitor(self, data:np.array) -> np.array:
        gaze_origins = data[:, 0:3]
        gaze_vectors = data[:, 3:6]

        a = np.einsum('ij,j->i', self._translation - gaze_origins, self._n)
        b = np.einsum('ij,j->i', gaze_vectors, self._n) 
        t = a / b

        g = gaze_origins + t[:, np.newaxis] * gaze_vectors - self._translation
        g = np.einsum('ij,kj->ki', self._inv_rotation, g)

        return g[:, 0:2]
