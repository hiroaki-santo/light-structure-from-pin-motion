#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os
from cv2 import aruco

import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Camera calibration')
parser.add_argument("--data_path", "-i", type=str)
ARGS = parser.parse_args()

img_paths = glob.glob(os.path.join(ARGS.data_path, "*.png"))
img = cv2.imread(img_paths[0])[:, :, ::-1]
m, n, _ = img.shape

board_size = np.loadtxt(os.path.join(ARGS.data_path, "board_size.txt"))
print("[*] board_size : {}".format(board_size))

markerLength = board_size[0]
markerSeparation = board_size[1]

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
board = aruco.GridBoard_create(5, 7, markerLength, markerSeparation, aruco_dict)
arucoParams = aruco.DetectorParameters_create()
##########################

all_corners = []
all_ids = []

for path in img_paths:
    try:
        print(path, end=" ")
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        corners, ids, rej, recovered_ids = aruco.refineDetectedMarkers(img_gray, board, corners, ids, rejectedImgPoints)
        print(len(ids))
        if len(ids) > 5:
            all_corners.append(np.array(corners))
            all_ids.append(ids)

    except Exception as e:
        print("[!] ERROR: {}".format(e))

print("[*] Complete Marker Detection OK/ALL: {}/{}".format(len(all_corners), len(img_paths)))

markerCounterPerFrame = []
allCornersConcatenated = []
allIdsConcatenated = []
for i in range(len(all_corners)):
    markerCounterPerFrame.append(len(all_corners[i]))
    for j in range(len(all_corners[i])):
        allCornersConcatenated.append(all_corners[i][j])
        allIdsConcatenated.append(all_ids[i][j])

all_corners = np.array(all_corners)
allIdsConcatenated = np.array(allIdsConcatenated)
markerCounterPerFrame = np.array(markerCounterPerFrame)

camera_matrix = np.zeros(shape=(3, 3))
camera_matrix[0:3, 0:3] = np.identity(3)
camera_matrix[0:2, 2] = np.array([n / 2, m / 2])
dist_coeffs = np.zeros(5)

res, intrinsic, dist, rvecs, tvecs = aruco.calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated,
                                                                markerCounterPerFrame, board, (m, n),
                                                                camera_matrix, dist_coeffs, None, None, 2)
res = np.array([res])

from datetime import datetime as dt

tdatetime = dt.now()
tstr = tdatetime.strftime('%Y-%m-%d_%H-%M-%S')

np.savez(os.path.join(ARGS.data_path, "params_{}.npz".format(tstr)), res=res, intrinsic=intrinsic, dist=dist)

print(intrinsic)
print(dist)
