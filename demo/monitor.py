#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import numpy as np

class monitor:

    def __init__(self):
        # default screen 13.3": 11.3" x 7.1", 1 inch = 25.4 mm
        self.h_mm = 7.1 * 25.4
        self.w_mm = 11.3 * 25.4

        self.h_pixels = 1080
        self.w_pixels = 1920

    def monitor_to_camera(self, x_pixel, y_pixel):

        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        x_cam_mm = ((int(self.w_pixels/2) - x_pixel)/self.w_pixels) * self.w_mm
        y_cam_mm = 10.0 + (y_pixel/self.h_pixels) * self.h_mm
        z_cam_mm = 0.0

        return x_cam_mm, y_cam_mm, z_cam_mm

    def camera_to_monitor(self, x_cam_mm, y_cam_mm):
        # assumes in-build laptop camera, located centered and 10 mm above display
        # update this function for you camera and monitor using: https://github.com/computer-vision/takahashi2012cvpr
        x_mon_pixel = np.ceil(int(self.w_pixels/2) - x_cam_mm * self.w_pixels / self.w_mm)
        y_mon_pixel = np.ceil((y_cam_mm - 10.0) * self.h_pixels / self.h_mm)

        return x_mon_pixel, y_mon_pixel
