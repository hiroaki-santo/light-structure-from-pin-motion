#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

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
