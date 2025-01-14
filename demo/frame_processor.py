#!/usr/bin/env python3

# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Shalini De Mello.
# --------------------------------------------------------

import cv2
from subprocess import call
import numpy as np
import pandas as pd
from os import path
import pickle
import sys
import os
import torch
import datetime
from pathlib import Path
from scipy.interpolate import RBFInterpolator

sys.path.append("ext/eth")
from undistorter import Undistorter
from KalmanFilter1D import Kalman1D

from monitor import Monitor
from face import face
from landmarks import Landmarks
from head import PnPHeadPoseEstimator
from normalization import normalize


class FrameProcessor:

    def __init__(self, cam_calib):

        self.cam_calib = cam_calib

        #######################################################
        #### prepare Kalman filters, R can change behaviour of Kalman filter
        #### play with it to get better smoothing, larger R - more smoothing and larger delay
        #######################################################
        self.kalman_filters = list()
        for point in range(2):
            # initialize kalman filters for different coordinates
            # will be used for face detection over a single object
            self.kalman_filters.append(Kalman1D(sz=100, R=0.01**2))

        self.kalman_filters_landm = list()
        for point in range(68):
            # initialize Kalman filters for different coordinates
            # will be used to smooth landmarks over the face for a single face tracking
            self.kalman_filters_landm.append(Kalman1D(sz=100, R=0.005**2))

        # initialize Kalman filter for the on-screen gaze point-of regard
        self.kalman_filter_gaze = list()
        self.kalman_filter_gaze.append(Kalman1D(sz=100, R=0.01**2))

        self.undistorter = Undistorter(self.cam_calib['mtx'],
                                       self.cam_calib['dist'])
        self.landmarks_detector = Landmarks()
        self.head_pose_estimator = PnPHeadPoseEstimator()

    def process_calibration(self, calibration_data: pd.DataFrame) -> pd.DataFrame:

        data: pd.Series = calibration_data.apply(lambda row: np.array(
            self._process_calibration_frame(row['frame'], row[['marker_x', 'marker_y', 'marker_z']].to_numpy())
        ),
                                                 axis=1)
        data = np.stack(data.to_list(), axis=0)

        return pd.DataFrame(
            data,
            columns=['image_a', 'gaze_a', 'head_a', 'R_gaze_a', 'R_head_a']).dropna()

    def _process_calibration_frame(self, image, gaze):
        # image = self.undistorter.apply(image)
        # detect face
        face_location = face.detect(image, scale=0.25, use_max='SIZE')
        # print('FACES: ', face_location)

        # No face detected
        if len(face_location) == 0:
            return

        # use kalman filter to smooth bounding box position
        # assume work with complex numbers:
        output_tracked = self.kalman_filters[0].update(face_location[0] +
                                                       1j * face_location[1])
        face_location[0], face_location[1] = np.real(output_tracked), np.imag(
            output_tracked)
        output_tracked = self.kalman_filters[1].update(face_location[2] +
                                                       1j * face_location[3])
        face_location[2], face_location[3] = np.real(output_tracked), np.imag(
            output_tracked)

        # detect facial points
        pts = self.landmarks_detector.detect(face_location, image)
        # run Kalman filter on landmarks to smooth them
        for i in range(68):
            kalman_filters_landm_complex = self.kalman_filters_landm[i].update(
                pts[i, 0] + 1j * pts[i, 1])
            pts[i, 0], pts[i,
                           1] = np.real(kalman_filters_landm_complex), np.imag(
                               kalman_filters_landm_complex)

        # compute head pose
        fx, _, cx, _, fy, cy, _, _, _ = self.cam_calib['mtx'].flatten()
        camera_parameters = np.asarray([fx, fy, cx, cy])
        rvec, tvec = self.head_pose_estimator.fit_func(pts, camera_parameters)

        ######### GAZE PART #########

        # create normalized eye patch and gaze and head pose value,
        # if the ground truth point of regard is given
        head_pose = (rvec, tvec)

        entry = {
            'full_frame': image,
            'full_frame_size': (image.shape[0], image.shape[1]),
            '3d_gaze_target': np.array([gaze[0], gaze[1], gaze[2]]).reshape(3,1),
            'camera_parameters': camera_parameters,
            'face_bounding_box': (int(face_location[0]), int(face_location[1]),
                                  int(face_location[2] - face_location[0]),
                                  int(face_location[3] - face_location[1]))
        }
        [patch, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target] = normalize(entry, head_pose)

        def preprocess_image(image):
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            image = np.transpose(image, [2, 0, 1])  # CxHxW
            image = 2.0 * image / 255.0 - 1
            return image

        # estimate the PoR using the gaze network
        processed_patch = preprocess_image(patch)
        processed_patch = processed_patch[np.newaxis, :, :, :]

        # Functions to calculate relative rotation matrices for gaze dir. and head pose
        def R_x(theta):
            sin_ = np.sin(theta)
            cos_ = np.cos(theta)
            return np.array([[1., 0., 0.], [0., cos_, -sin_],
                             [0., sin_, cos_]]).astype(np.float32)

        def R_y(phi):
            sin_ = np.sin(phi)
            cos_ = np.cos(phi)
            return np.array([[cos_, 0., sin_], [0., 1., 0.],
                             [-sin_, 0., cos_]]).astype(np.float32)

        def calculate_rotation_matrix(e):
            return np.matmul(R_y(e[1]), R_x(e[0]))

        # compute the ground truth POR if the
        # ground truth is available
        R_head_a = calculate_rotation_matrix(h_n)
        R_gaze_a = calculate_rotation_matrix(g_n)

        return np.array([processed_patch, g_n, h_n, R_gaze_a, R_head_a])

    # MONITOR

    def fit_func(self, landmarks, camera_parameters):
        landmarks = np.array([
            landmarks[i - 1, :]
            for i in self.ibug_ids_to_use
        ], dtype=np.float64)
        fx, fy, cx, cy = camera_parameters

        # print('SOLVE PNP POINTS: ', self.sfm_points_for_pnp)
        # print('SOLVE PNP LANDMARKS: ', landmarks)
        # Initial fit
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(self.sfm_points_for_pnp, landmarks,
                                                          camera_matrix, None, flags=cv2.SOLVEPNP_EPNP)

        # Second fit for higher accuracy
        success, rvec, tvec = cv2.solvePnP(self.sfm_points_for_pnp, landmarks, camera_matrix, None,
                                           rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

        return rvec, tvec

    def project_model(self, rvec, tvec, camera_parameters):
        fx, fy, cx, cy = camera_parameters
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        points, _ = cv2.projectPoints(self.sfm_points_ibug_subset, rvec, tvec, camera_matrix, None)
        return points

    def process_monitor(self, calibration_data: pd.DataFrame, monitor: Monitor, device, gaze_network) -> pd.DataFrame:

        #
        # calculate gazes
        #
        data: pd.Series = calibration_data.apply(lambda row: np.array(
            self._process_video_frame(row['frame'], device, gaze_network)
        ),
                                                 axis=1)
        
        data = np.stack(data.to_list(), axis=0)
        data = pd.DataFrame(
            data,
            columns=['gaze_origin_x', 'gaze_origin_y', 'gaze_origin_z', 'gaze_vector_x', 'gaze_vector_y', 'gaze_vector_z'])

        data = pd.concat([data, calibration_data[['marker_x', 'marker_y']]], axis=1)

        print('Gazes & Markers')
        print(data[['marker_x', 'marker_y', 'gaze_vector_x', 'gaze_vector_y', 'gaze_vector_z']])

        grouped_data = data.groupby(['marker_x', 'marker_y'], as_index=False).median()
        print('Grouped gazes')
        print(grouped_data[['marker_x', 'marker_y', 'gaze_vector_x', 'gaze_vector_y', 'gaze_vector_z']])

        markers = grouped_data[['marker_x','marker_y']]
        markers['marker_z'] = 0
        gaze_vectors = grouped_data[['gaze_vector_x','gaze_vector_y','gaze_vector_z']]
        gaze_origins = grouped_data[['gaze_origin_x','gaze_origin_y','gaze_origin_z']]

        # Move origin to the gaze centers, inverse z-axis 
        gazes = gaze_vectors
        gazes['gaze_vector_z'] = -gazes['gaze_vector_z']

        # project gazes to the plane z=1 (in the gaze coordinate system)
        d = 1.0 / gazes.loc[:, 'gaze_vector_z'].to_numpy()
        d = d.reshape((-1,1))

        gazes = d * gazes
        
        #
        # Fit the screen plane to the found gazes configuration
        #
        camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(markers.to_numpy(), gazes[['gaze_vector_x','gaze_vector_y']].to_numpy(),
                                                          camera_matrix, None, flags=cv2.SOLVEPNP_EPNP)

        # Second fit for higher accuracy
        success, rvec, tvec = cv2.solvePnP(markers.to_numpy(), gazes[['gaze_vector_x','gaze_vector_y']].to_numpy(), camera_matrix, None,
                                           rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.squeeze(tvec)
        # convert to the camera coordinate system (z = -z)
        rotation[2, :] *= -1
        translation[2] *= -1
        print('ROTATION: ', rotation)
        print('TRANSLATION: ', translation)

        # calculate gaze vectors from the gaze origin 
        # to the markers on the screen plane
        markers = calibration_data[['marker_x', 'marker_y']]
        markers['marker_z'] = 0

        # gaze vectors
        gaze_vectors = np.einsum('ij,kj->ki', rotation, markers.to_numpy()) + translation
        gaze_vectors = gaze_vectors / np.linalg.norm(gaze_vectors, axis=-1)[:, np.newaxis]

        # screen plane translation
        # in the camera coordinate system
        gaze_origin = data[['gaze_origin_x','gaze_origin_y','gaze_origin_z']].median().to_numpy()
        translation = gaze_origin + translation
        print('Gaze origin: ', gaze_origin)
        print('Translation: ', translation)

        monitor.set_transform(rotation, translation)

        calibration_data[['marker_x', 'marker_y', 'marker_z']] = gaze_vectors
        print('Calibration data')
        print(calibration_data[['marker_x', 'marker_y', 'marker_z']])

        return calibration_data

    def process_video(self, cap: cv2.VideoCapture, monitor: Monitor, device, gaze_network):

        data = []

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp > 0:
                gaze = self._process_video_frame(image, device, gaze_network)
                if gaze is not None:
                    # Kalman filtration
                    # x, y = gaze[0], gaze[1]
                    # fxy = self.kalman_filter_gaze[0].update(x + 1j * y)
                    # x, y = np.ceil(np.real(fxy)), np.ceil(np.imag(fxy))
                    gaze = np.insert(gaze, 0, timestamp)
                    data.append(gaze)

        data = np.stack(data, axis=0)
        data[:, 1:3] = monitor.camera_to_monitor(data[:, 1:7])

        return pd.DataFrame(data[:, 0:3], columns=['timestamp', 'x', 'y'])

    def _process_video_frame(self,
                             image: np.array,
                             device,
                             gaze_network):
        # image = self.undistorter.apply(image)
        # detect face
        face_location = face.detect(image, scale=0.25, use_max='SIZE')
        # print('FACES: ', face_location)

        # No face detected
        if len(face_location) == 0:
            return None

        # use kalman filter to smooth bounding box position
        # assume work with complex numbers:
        output_tracked = self.kalman_filters[0].update(face_location[0] +
                                                       1j * face_location[1])
        face_location[0], face_location[1] = np.real(output_tracked), np.imag(
            output_tracked)
        output_tracked = self.kalman_filters[1].update(face_location[2] +
                                                       1j * face_location[3])
        face_location[2], face_location[3] = np.real(output_tracked), np.imag(
            output_tracked)

        # detect facial points
        landmarks = self.landmarks_detector.detect(face_location, image)
        # run Kalman filter on landmarks to smooth them
        for i in range(68):
            kalman_filters_landm_complex = self.kalman_filters_landm[i].update(
                landmarks[i, 0] + 1j * landmarks[i, 1])
            landmarks[i, 0], landmarks[i,
                           1] = np.real(kalman_filters_landm_complex), np.imag(
                               kalman_filters_landm_complex)

        # compute head pose
        fx, _, cx, _, fy, cy, _, _, _ = self.cam_calib['mtx'].flatten()
        camera_parameters = np.asarray([fx, fy, cx, cy])
        rvec, tvec = self.head_pose_estimator.fit_func(landmarks, camera_parameters)

        ######### GAZE PART #########

        # create normalized eye patch and gaze and head pose value,
        # if the ground truth point of regard is given
        head_pose = (rvec, tvec)
        entry = {
            'full_frame': image,
            'full_frame_size': (image.shape[0], image.shape[1]),
            '3d_gaze_target': None,
            'camera_parameters': camera_parameters,
            'face_bounding_box': (int(face_location[0]), int(face_location[1]),
                                  int(face_location[2] - face_location[0]),
                                  int(face_location[3] - face_location[1]))
        }
        [patch, h_n, _, inverse_M, gaze_cam_origin, _] = normalize(entry, head_pose)

        # cv2.imshow('raw patch', patch)

        def preprocess_image(image):
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
            # cv2.imshow('processed patch', image)

            image = np.transpose(image, [2, 0, 1])  # CxHxW
            image = 2.0 * image / 255.0 - 1
            return image

        # estimate the PoR using the gaze network
        processed_patch = preprocess_image(patch)
        processed_patch = processed_patch[np.newaxis, :, :, :]

        # Functions to calculate relative rotation matrices for gaze dir. and head pose
        def R_x(theta):
            sin_ = np.sin(theta)
            cos_ = np.cos(theta)
            return np.array([[1., 0., 0.], [0., cos_, -sin_],
                             [0., sin_, cos_]]).astype(np.float32)

        def R_y(phi):
            sin_ = np.sin(phi)
            cos_ = np.cos(phi)
            return np.array([[cos_, 0., sin_], [0., 1., 0.],
                             [-sin_, 0., cos_]]).astype(np.float32)

        def calculate_rotation_matrix(e):
            return np.matmul(R_y(e[1]), R_x(e[0]))

        def pitchyaw_to_vector(pitchyaw):
            vector = np.zeros((3, 1))
            vector[0, 0] = np.cos(pitchyaw[0]) * np.sin(pitchyaw[1])
            vector[1, 0] = np.sin(pitchyaw[0])
            vector[2, 0] = np.cos(pitchyaw[0]) * np.cos(pitchyaw[1])
            return vector

        # compute the ground truth POR if the
        # ground truth is available
        R_head_a = calculate_rotation_matrix(h_n)
        R_gaze_a = np.zeros((1, 3, 3))

        input_dict = {
            'image_a': processed_patch,
            'gaze_a': [],
            'head_a': h_n,
            'R_gaze_a': R_gaze_a,
            'R_head_a': R_head_a,
        }

        # compute eye gaze and point of regard
        for k, v in input_dict.items():
            input_dict[k] = torch.FloatTensor(v).to(device).detach()

        gaze_network.eval()
        output_dict = gaze_network(input_dict)
        output = output_dict['gaze_a_hat']
        g_cnn = output.data.cpu().numpy()
        g_cnn = g_cnn.reshape(3, 1)
        g_cnn /= np.linalg.norm(g_cnn)

        # print('g_cnn: ', g_cnn)

        # compute the POR on z=0 plane
        g_n_forward = -g_cnn
        # print('g_cnn: ', g_cnn)
        g_cam_forward = inverse_M * g_n_forward
        g_cam_forward = g_cam_forward / np.linalg.norm(g_cam_forward)

        gaze_cam_origin = gaze_cam_origin.squeeze()
        # print('gaze_cam_origin: ', gaze_cam_origin)

        g_cam_forward = np.asarray(g_cam_forward).squeeze()
        # print('g_cam_forward: ', g_cam_forward)

        return np.concatenate((gaze_cam_origin, g_cam_forward))

