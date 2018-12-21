#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import glob

import cv2
import numpy as np

import os


def ang_error_deg(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    assert len(a) == 3
    assert len(b) == 3

    r = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    r = np.arccos(np.clip(r, -1, 1))
    return np.rad2deg(r)


def polar2xyz(theta, phi, r):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([x, y, z])


def find_homography(marker_coordinates, board_obj_points):
    corresponds = []

    assert len(marker_coordinates) == len(board_obj_points)

    for i in range(len(marker_coordinates)):
        for j in range(4):
            img_point = marker_coordinates[i, j]
            board_point = board_obj_points[i, j][:2]

            if np.sum(img_point) == 0:
                continue

            corresponds.append((img_point, board_point))

    src = np.float32([pair[0] for pair in corresponds]).reshape(-1, 1, 2)
    dst = np.float32([pair[1] for pair in corresponds]).reshape(-1, 1, 2)

    retval, mask = cv2.findHomography(src, dst, cv2.RANSAC)
    return retval


def project_distant(L, P):
    pose_num, _ = L.shape
    pin_num, _ = P.shape

    projected_points = np.zeros(shape=(pose_num, pin_num, 3))
    for l in range(pose_num):
        for p in range(pin_num):
            light_vec = L[l]
            mat_L = np.zeros((3, 4))
            mat_P = np.ones(4)
            mat_L[0, 0] = light_vec[2]
            mat_L[1, 1] = light_vec[2]
            mat_L[2, 3] = light_vec[2]
            mat_L[0, 2] = -light_vec[0]
            mat_L[1, 2] = -light_vec[1]
            mat_P[0:3] = P[p]

            s = mat_L.dot(mat_P)
            projected_points[l, p,] = s[:] / s[2]

    return projected_points


def project_near(L, P):
    pose_num, _ = L.shape
    pin_num, _ = P.shape

    projected_points = np.zeros(shape=(pose_num, pin_num, 3))
    for l in range(pose_num):
        for p in range(pin_num):
            light_vec = L[l]
            mat_L = np.zeros((3, 4))
            mat_P = np.ones(4)
            mat_L[0, 0] = -light_vec[2]
            mat_L[1, 1] = -light_vec[2]
            mat_L[2, 3] = -light_vec[2]
            mat_L[0, 2] = light_vec[0]
            mat_L[1, 2] = light_vec[1]
            mat_L[2, 2] = 1.
            mat_P[0:3] = P[p]

            s = mat_L.dot(mat_P)
            projected_points[l, p,] = s[:] / s[2]

    return projected_points


def error_reprojection_near(projected_points, light_source_coordinates, pin_coordinates):
    pose_num, pin_num, _ = projected_points.shape
    reprojected = project_near(light_source_coordinates, pin_coordinates)

    reprojection_error = np.linalg.norm(
        reprojected.reshape(pose_num * pin_num, 3) - projected_points.reshape(pose_num * pin_num, 3), axis=1)
    return np.average(reprojection_error)


def error_reprojection_distant(projected_points, light_source_coordinates, pin_coordinates):
    pose_num, pin_num, _ = projected_points.shape
    reprojected = project_distant(light_source_coordinates, pin_coordinates)

    reprojection_error = np.linalg.norm(
        reprojected.reshape(pose_num * pin_num, 3) - projected_points.reshape(pose_num * pin_num, 3), axis=1)
    return np.average(reprojection_error)


def ransac_wrapper(projected_points, Rs, tvecs, func, ransac_num, iter, init_P=None, init_L=None):
    """
    Run RANSAC and return all results
    :param projected_points:
    :param Rs:
    :param tvecs:
    :param callable func: methods.solve_near() or method.solve_distant()
    :param int ransac_num: number of sampling
    :param int iter: number of iteration
    :param init_P:
    :param init_L:
    :return: list of results
    """
    pose_num, pin_num, _ = projected_points.shape

    assert Rs.shape == (pose_num, 3, 3)
    assert tvecs.shape == (pose_num, 3)

    if init_L is not None:
        assert init_L.shape == (pose_num, 3)

    if init_P is not None:
        assert init_P.shape == (pin_num, 3)

    if pose_num <= ransac_num:
        ransac_num = pose_num
        iter = 1

    results = []
    for _ in range(iter):
        # sampling
        pose_num_ = min(pose_num, ransac_num)
        indices = sorted(np.random.permutation(pose_num)[0:pose_num_])
        projected_points_ = projected_points[indices, :, :].copy()
        Rs_, tvecs_ = Rs[indices], tvecs[indices]
        init_L_ = init_L[indices, :] if init_L is not None else None

        result = func(projected_points_, Rs_, tvecs_, init_P=init_P, init_L=init_L_)
        result["indices"] = indices
        results.append(result)

    return results


def ransac_find_best_near(results, projected_points, Rs, tvecs):
    """
    calculate reprojection error for all results and return best one.

    :param results: list of results
    :param projected_points:
    :param Rs:
    :param tvecs:
    :return: one of results
    """
    best_res = np.float("inf")
    for result in results:
        P = result["P"]
        glight = result["best_global_position"]

        L = np.zeros((len(Rs), 3))
        for l in range(len(Rs)):
            L[l, :] = Rs[l].T.dot(glight) - Rs[l].T.dot(tvecs[l])
        res = error_reprojection_near(projected_points, L, P)
        result["res_ba"] = res
        if res < best_res:
            best_res = res
            best_result = result

    return best_result


def ransac_find_best_distant(results, projected_points, Rs, tvecs=None):
    best_res = np.float("inf")
    for result in results:
        P = result["P"]
        glight = result["best_global_position"]

        L = np.zeros((len(Rs), 3))
        for l in range(len(Rs)):
            L[l, :] = Rs[l].T.dot(glight)
        res = error_reprojection_distant(projected_points, L, P)
        result["res_ba"] = res
        if res < best_res:
            best_res = res
            best_result = result

    return best_result


def load_data(dir_path, pin_num):
    """

    :param str dir_path:
    :param int pin_num:
    :return:
    """
    print("[*] load_data()")

    assert os.path.exists(dir_path), dir_path
    assert pin_num > 0, pin_num

    camera_matrix, camera_dist = load_camera_params(dir_path)

    #############
    img_paths = glob.glob(os.path.join(dir_path, "*.png"))
    img_paths = sorted(img_paths)
    img_paths = np.array(img_paths, dtype=str)

    def __file_name(p):
        file_name, ext = os.path.splitext(p)
        return file_name

    file_names = [__file_name(path) for path in img_paths]
    detected_shadow_paths = [os.path.join(dir_path, "{}_detected_label.txt".format(file_name))
                             for file_name in file_names]
    marker_coordinates_paths = [os.path.join(dir_path, "{}_marker_coordinates.npz".format(path)) for path in file_names]

    marker_coordinates_paths = np.array(marker_coordinates_paths, dtype=str)
    detected_shadow_paths = np.array(detected_shadow_paths, dtype=str)

    ############
    imgs = [cv2.imread(path)[:, :, ::-1] for path in img_paths]
    imgs = [cv2.undistort(i, camera_matrix, camera_dist) for i in imgs]

    detected_shadow_points = [np.loadtxt(path) for path in detected_shadow_paths]
    projected_points_detected = np.zeros(shape=(len(detected_shadow_points), pin_num, 3))
    if len(detected_shadow_points) != 0:
        for l, p in enumerate(detected_shadow_points):
            projected_points_detected[l, :, :] = p

    marker_coordinates = [np.load(path)["marker_coordinates"] for path in marker_coordinates_paths]
    board_objPoints = [np.load(path)["board_objPoints"] for path in marker_coordinates_paths]
    Rs = np.array([np.load(path)["R"] for path in marker_coordinates_paths])
    tvecs = np.array([np.load(path)["tvec"] for path in marker_coordinates_paths])
    tvecs = tvecs.reshape(-1, 3)

    print("[*] load_data() complete.")
    return {"img_paths": img_paths, "imgs": imgs,
            "marker_coordinates": marker_coordinates, "board_objPoints": board_objPoints,
            "projected_points_detected": projected_points_detected, "Rs": Rs, "tvecs": tvecs}


def load_camera_params(dir_path):
    path = glob.glob(os.path.join(dir_path, "params_*.npz"))[-1]
    params = np.load(path)
    camera_matrix = params["intrinsic"]
    camera_dist = params["dist"]

    return camera_matrix, camera_dist


def tracking(unsorted_projected_points):
    pose_num, pin_num, _ = unsorted_projected_points.shape

    projected_points = np.zeros_like(unsorted_projected_points) - 1
    projected_points[0, :, :] = unsorted_projected_points[0, :, :]

    failed_indices = []
    for i in range(1, pose_num):
        points_ = projected_points[i - 1, :, :]
        points = unsorted_projected_points[i, :, :]

        used_indices = []
        for j in range(pin_num):
            prev_point = points_[j, :]
            min_d, index = np.finfo(float).max, -1
            for j_ in range(pin_num):
                point = points[j_, :]
                d = np.linalg.norm(prev_point[:2] - point[:2])
                if d < min_d:
                    min_d = d
                    index = j_
            projected_points[i, j, :] = unsorted_projected_points[i, index, :]
            used_indices.append(index)
        if len(np.unique(used_indices)) != len(used_indices):
            failed_indices.append(i)

    return projected_points, failed_indices
