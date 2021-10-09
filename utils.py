#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import glob
import itertools
import os
import warnings
from collections import defaultdict

import cv2
import joblib
import numpy as np


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


def project_unified(L, P):
    light_num, _ = L.shape
    pin_num, _ = P.shape

    projected_points = np.zeros(shape=(light_num, pin_num, 3), dtype=L.dtype)
    for l in range(light_num):
        for p in range(pin_num):
            light_vec = L[l]
            mat_L = np.zeros((3, 4))
            mat_P = np.ones(4)
            mat_L[0, 0] = 1.
            mat_L[1, 1] = 1.
            mat_L[2, 3] = 1.
            mat_L[0, 2] = light_vec[0] / -light_vec[2]
            mat_L[1, 2] = light_vec[1] / -light_vec[2]
            mat_L[2, 2] = 1. / -light_vec[2]
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


def error_reprojection(projected_points, light_source_coordinates, pin_coordinates):
    pose_num, pin_num, _ = projected_points.shape
    reprojected = project_unified(light_source_coordinates, pin_coordinates)

    reprojection_error = np.linalg.norm(
        reprojected.reshape(pose_num * pin_num, 3) - projected_points.reshape(pose_num * pin_num, 3), axis=1)
    return reprojection_error


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

    def __run():
        # sampling
        pose_num_ = min(pose_num, ransac_num)
        indices = sorted(np.random.permutation(pose_num)[0:pose_num_])
        projected_points_ = projected_points[indices, :, :].copy()
        Rs_, tvecs_ = Rs[indices], tvecs[indices]
        init_L_ = init_L[indices, :] if init_L is not None else None

        result = func(projected_points_, Rs_, tvecs_, init_P=init_P, init_L=init_L_)
        result["indices"] = indices
        return result

    results = joblib.Parallel(n_jobs=-1, verbose=1)([joblib.delayed(__run)() for i in range(iter)])

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


def shadow_correspondence(unsorted_projected_points, seed=None):
    print("[*] shadow_correspondence()")
    import fmatrix

    if seed is not None:
        np.random.seed(seed)

    pose_num, pin_num, _ = unsorted_projected_points.shape
    sorted_projected_points = unsorted_projected_points.copy()

    ########################################################
    CANDIDATE_NUM = 3
    assert CANDIDATE_NUM >= 2

    axis_indices = []
    MAX_ITER = 100
    iter_count = 0
    while len(axis_indices) < max(pose_num * 0.5, 5):
        iter_count += 1
        if (iter_count > MAX_ITER) and (len(axis_indices) >= 2):
            warnings.warn("reaching max number of iterations")
            break

        if len(axis_indices) == 0:
            axis_indices.append(np.random.permutation(pose_num)[0])

        axis_index = np.random.choice(axis_indices)
        axis_pts = sorted_projected_points[axis_index, :, 0:2]
        pts = np.zeros(shape=(CANDIDATE_NUM, pin_num, 2))

        rindices = [i for i in np.random.permutation(pose_num) if i not in axis_indices]
        for i in range(len(rindices) - CANDIDATE_NUM + 1):
            cand_indices = rindices[i:i + CANDIDATE_NUM]

            # print("pair: ", axis_index, cand_indices, end=" ")
            pts[:] = unsorted_projected_points[cand_indices, :, 0:2]

            ress = []
            indices0_last, res, F = _corr_one_pair(axis_pts.T, pts[-1].T)
            ress.append(res)
            #
            indices01, res, F = _corr_one_pair(axis_pts.T, pts[0].T)
            ress.append(res)
            #
            if np.isinf(ress).any():
                continue

            indices_list = [indices01]
            for j in range(CANDIDATE_NUM - 1):
                indices_tmp, res, F = _corr_one_pair(pts[j, indices_list[-1], :].T, pts[j + 1, :, :].T)
                ress.append(res)
                indices_list.append(indices_tmp)
                if np.isinf(res):
                    break

            res = np.mean(ress)
            if (res == np.inf) or (indices0_last != indices_list[-1]):
                continue

            print("accept:", axis_index, cand_indices, "res:", res)
            for j in range(CANDIDATE_NUM):
                sorted_projected_points[cand_indices[j], :, 0:2] = pts[j, indices_list[j], :]
            axis_indices.extend(cand_indices)

        axis_indices = list(np.unique(axis_indices))
        if len(axis_indices) < 2:
            # axis_indices has only initial data. Then reset the list.
            axis_indices = []

    sorted_projected_points[[i for i in range(pose_num) if i not in axis_indices]] = np.nan

    ########################################################
    # Evaluate the residual to extract reliable correspondence established data
    #
    scores = []
    for axis_index in axis_indices:
        pts0 = sorted_projected_points[axis_index, :, 0:2].T
        scores_tmp = []

        # calculate residuals between all pts in axis_indices
        for i in [i for i in axis_indices if i != axis_index]:
            pts1 = sorted_projected_points[i, :, 0:2].T
            F = fmatrix.estimate(pts0.T, pts1.T)
            if F is None:
                res = 10.
            else:
                res = fmatrix.residuals_for_known_corresp(pts0, pts1, F)
                if np.isnan(res).any():
                    res = 10.

            scores_tmp.append(np.mean(res))

        scores.append(np.mean(scores_tmp))

    # sort scores and only use the lowest n% data
    scores = np.array(scores)
    indices = np.argsort(scores)[:int(len(axis_indices) * 0.5)]
    # print("all scores:", np.sort(scores))

    axis_indices = list(np.array(axis_indices)[indices])
    print("selected poses:", axis_indices, len(axis_indices))
    ########################################################

    for i in [i for i in range(pose_num) if i not in axis_indices]:
        sorted_projected_points[i, :, :] = unsorted_projected_points[i, :, :]
    sorted_projected_points, no_use_indices = _voting(sorted_projected_points, axis_indices)
    ########################################################

    sorted_projected_points[[i for i in range(pose_num) if i in no_use_indices]] = np.nan
    ########################################################

    sorted_projected_points[:, :, 2] = 1.
    return sorted_projected_points, no_use_indices


def _corr_one_pair(pts_unknown_corr_0, pts_unknown_corr_1, homogeneity_threshold=5e-2,
                   num_pts_for_unknown_estimation=-1):
    import fmatrix
    _, pin_num = pts_unknown_corr_0.shape
    assert pts_unknown_corr_0.shape == (2, pin_num), pts_unknown_corr_0.shape
    assert pts_unknown_corr_1.shape == (2, pin_num), pts_unknown_corr_1.shape

    if num_pts_for_unknown_estimation <= 0:
        num_pts_for_unknown_estimation = pin_num
    assert 0 < num_pts_for_unknown_estimation <= pin_num, num_pts_for_unknown_estimation

    pts_unknown_corr_0_subset = pts_unknown_corr_0[:, :num_pts_for_unknown_estimation].copy()
    best_residual = np.inf
    best_F = None
    best_indices = []
    for i, permutation in enumerate(itertools.permutations(np.arange(pin_num), r=num_pts_for_unknown_estimation)):
        pts_unknown_corr_1_subset = pts_unknown_corr_1[:, permutation]
        try:
            F = fmatrix.estimate(pts_unknown_corr_0_subset.T, pts_unknown_corr_1_subset.T,
                                 homogeneity_threshold=homogeneity_threshold)
        except:
            print("[!]: ERR in three_point()", end=" ")
            print(pts_unknown_corr_0, pts_unknown_corr_1)
            F = None

        if F is None:  # the function returns None if the eq. system is not homogeneous
            continue
        residuals, indices = fmatrix.residuals_for_unknown_corresp(pts_unknown_corr_0, pts_unknown_corr_1, F)
        if np.mean(residuals) < best_residual:
            best_residual = np.mean(residuals)
            best_F = F
            best_indices = indices

    if len(best_indices) != len(np.unique(best_indices)):
        best_residual = np.inf

    return best_indices, best_residual, best_F


def _voting(projected_points, axis_poses, cpu=-1):
    import joblib
    pose_num, pin_num, _ = projected_points.shape
    no_use_poses = []
    projected_points = projected_points.copy()

    #
    # indices, res, F = _corr_one_pair()
    def __wrapper(*args, **kwargs):
        i = kwargs.pop("i")
        axis_pose = kwargs.pop("axis_pose")
        return _corr_one_pair(*args, **kwargs), i, axis_pose

    rvals = joblib.Parallel(n_jobs=cpu, verbose=1)([
        joblib.delayed(__wrapper)(projected_points[axis_pose, :, 0:2].T, projected_points[i, :, 0:2].T,
                                  homogeneity_threshold=1e-2, i=i, axis_pose=axis_pose)
        for i in range(pose_num) if i not in axis_poses if not np.isnan(projected_points[i]).any()
        for axis_pose in axis_poses
    ])

    for i in [i for i in range(pose_num) if i not in axis_poses]:
        index_count_dict = defaultdict(lambda: 0)
        for indices, res, F in [rval[0] for rval in rvals if rval[1] == i]:
            if res != np.inf:
                index_count_dict[tuple(indices)] += 1

        scores = [v for k, v in index_count_dict.items()]
        best_indices = [k for k, v in index_count_dict.items() if v == np.max(scores)]

        if (len(scores) == 0) or (len(best_indices) != 1):
            no_use_poses.append(i)
            continue

        if (np.sum(scores) < max(3., len(axis_poses) * 0.3)) or (np.max(scores) < np.sum(scores) * 0.51):
            # We only accept reliable score, which means n% of established data supports.
            no_use_poses.append(i)
            continue

        projected_points[i, :, :] = projected_points[i, best_indices[0], :]
    return projected_points, no_use_poses
