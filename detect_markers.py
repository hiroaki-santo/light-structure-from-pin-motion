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


def detect_markers(img, params, root_dir, file_name):
    o_path = os.path.join(root_dir, "tmp")
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    #######

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)

    path_board_size = os.path.join(root_dir, "board_size.txt")
    assert os.path.exists(path_board_size), path_board_size

    board_size = np.loadtxt(path_board_size)
    print("[*] board_size : {}".format(board_size))
    markerLength = board_size[0]
    markerSeparation = board_size[1]
    board = aruco.GridBoard_create(5, 7, markerLength, markerSeparation, aruco_dict)

    ####
    intrinsic = params["intrinsic"]
    dist = params["dist"]
    arucoParams = aruco.DetectorParameters_create()

    img = cv2.undistort(img, intrinsic, dist)
    img_gray = cv2.cvtColor(img[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
    corners, ids, rej, recovered_ids = aruco.refineDetectedMarkers(img_gray, board, corners, ids, rejectedImgPoints)

    dist[:] = 0
    retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, intrinsic, dist)
    R, _ = cv2.Rodrigues(rvec)

    tmp_img = aruco.drawDetectedMarkers(img[:, :, ::-1].copy(), corners, ids, (0, 255, 0))
    tmp_img = aruco.drawAxis(tmp_img, intrinsic, dist, rvec, tvec, 100)
    cv2.imwrite(os.path.join(o_path, "{}_marker_detected.png".format(file_name)), tmp_img)

    ########
    marker_coordinates = np.zeros(shape=(5 * 7, 4, 2))

    for index in range(len(ids)):
        try:
            marker_coordinates[ids[index], :, :] = corners[index].reshape(4, 2)
        except Exception as e:
            print("ERROR: ", end=" ")
            print(ids[index], corners[index])

    board_objPoints = np.array(board.objPoints)
    return marker_coordinates, board_objPoints, R, tvec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate pose of board via Aruco')
    parser.add_argument("--data_path", "-i", type=str)
    ARGS = parser.parse_args()

    root_dir = ARGS.data_path
    img_paths = glob.glob(os.path.join(root_dir, "*.png"))
    print("[*] {} images in {}".format(len(img_paths), root_dir))

    camera_params = np.load(glob.glob(os.path.join(root_dir, "params_*.npz"))[-1])
    for path in img_paths:
        print(path, end=" ")

        file_name, ext = os.path.splitext(os.path.basename(path))
        o_path = os.path.join(root_dir, "{}_marker_coordinates.npz".format(file_name))

        try:
            img = cv2.imread(path)[:, :, ::-1]
            marker_coordinates, board_objPoints, R, tvec = detect_markers(img, camera_params,
                                                                          root_dir=root_dir, file_name=file_name)
            np.savez(o_path, marker_coordinates=marker_coordinates, board_objPoints=board_objPoints, R=R, tvec=tvec)
        except Exception as e:
            print("[!] Error in detection: {}".format(e), path)
