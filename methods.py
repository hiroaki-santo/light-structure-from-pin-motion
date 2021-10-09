#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings

import numpy as np

import utils

XTOL_BA = 1.0e-8
MAXFEV = int(1e5)
IRLS_TOL = 1.0e-8
IRLS_ITER = int(1e5)


def solution_near_linear(projected_points, Rs, tvecs):
    """
    Solve Eq. (7) and (10)

    :param np.ndarray projected_points: shadow positions
    :param np.ndarray Rs: board poses
    :param np.ndarray tvecs: board poses
    :return: dict of estimated result
    """

    pose_num, pin_num, _ = projected_points.shape
    assert Rs.shape == (pose_num, 3, 3)
    assert tvecs.shape == (pose_num, 3)

    def __Ab_ij(R_, t_, s_):
        """

        :param np.ndarray R_: board pose
        :param np.ndarray t_: board pose
        :param np.ndarray s_: shadow position
        :return: A_ij anb b_ij in Eq. (7)
        """
        assert R_.shape == (3, 3)
        assert s_.shape == (3,)
        assert t_.shape == (3,)
        R_ = R_.T
        t_ = -R_.dot(t_)
        #
        R_ = R_.reshape(-1)
        A_ = np.zeros(shape=(3, 15))
        b_ = np.zeros(shape=3)

        A_[:, 0] = (R_[6] * s_[1], -R_[6] * s_[0], R_[3] * s_[0] - R_[0] * s_[1])
        A_[:, 1] = (R_[7] * s_[1], -R_[7] * s_[0], R_[4] * s_[0] - R_[1] * s_[1])
        A_[:, 2] = (R_[8] * s_[1], -R_[8] * s_[0], R_[5] * s_[0] - R_[2] * s_[1])

        A_[:, 3] = (0, t_[2], -t_[1] + s_[1])
        A_[:, 4] = (-t_[2], 0, t_[0] - s_[0])
        A_[:, 5] = (t_[1] - s_[1], -t_[0] + s_[0], 0)

        A_[:, 6] = (0, R_[6], -R_[3])
        A_[:, 7] = (-R_[6], 0, R_[0])
        A_[:, 8] = (R_[3], -R_[0], 0)

        A_[:, 9] = (0, R_[7], -R_[4])
        A_[:, 10] = (-R_[7], 0, R_[1])
        A_[:, 11] = (R_[4], -R_[1], 0)

        A_[:, 12] = (0, R_[8], -R_[5])
        A_[:, 13] = (-R_[8], 0, R_[2])
        A_[:, 14] = (R_[5], -R_[2], 0)

        b_[:] = (-t_[2] * s_[1],
                 t_[2] * s_[0],
                 -t_[1] * s_[0] + t_[0] * s_[1])
        return A_, b_

    def __x2light_pin(x):
        """
        extract light coordinates and pin positions from \theta_j in Eq. (7)
        :param x:
        :return: light and pin positions
        """
        x = np.array(x).flatten()
        assert x.shape == (3 + 12 * pin_num,), (x.shape, (3 + 12 * pin_num))
        lc = np.ones(3)
        lc[0:3] = x[0:3]
        #
        pins = np.zeros(shape=(pin_num, 3))
        for p in range(pin_num):
            pins[p, :] = x[3 + 12 * p:3 + 12 * p + 3]

        return lc, pins

    Qc = np.zeros(shape=(3 * pose_num * pin_num, 3 + 12 * pin_num))
    Bc = np.zeros(shape=(3 * pose_num * pin_num))
    for i in range(pose_num):
        for p in range(pin_num):
            A, b = __Ab_ij(Rs[i], tvecs[i], projected_points[i, p])
            Qc[3 * pose_num * p + 3 * i:3 * pose_num * p + 3 * i + 3, 0:3] = A[:, 0:3]
            Qc[3 * pose_num * p + 3 * i:3 * pose_num * p + 3 * i + 3, 3 + 12 * p:3 + 12 * p + 12] = A[:, 3:]
            Bc[3 * pose_num * p + 3 * i:3 * pose_num * p + 3 * i + 3] = b

    from L1_solver.L1_residual_min import L1_residual_min
    X = L1_residual_min(np.matrix(Qc), np.matrix(Bc).T, MAX_ITER=IRLS_ITER, tol=IRLS_TOL)
    X = np.array(X)
    lc, P = __x2light_pin(X)

    # calculate l_i from l, R_i and t_i
    L = np.zeros(shape=(pose_num, 3))
    for l in range(pose_num):
        R = np.matrix(Rs[l]).T
        t = -R.dot(tvecs[l]).flatten()

        L[l, :] = R.dot(lc.flatten()) + t

    res = utils.error_reprojection_near(projected_points, L, P)
    result = {}
    result["best_global_position"] = lc
    result["L"] = L
    result["P"] = P
    result["indices"] = np.arange(pose_num)
    result["res"] = res

    return result


def solution_distant_linear(projected_points, Rs, tvecs=None):
    pose_num, pin_num, _ = projected_points.shape
    assert Rs.shape == (pose_num, 3, 3), Rs.shape

    def __Ab_ij(R_, s_):
        assert R_.shape == (3, 3), R_.shape
        assert s_.shape == (3,), s_.shape
        R_ = R_.T.reshape(-1)
        A_ = np.zeros(shape=(3, 11))
        b_ = np.zeros(shape=3)
        A_[0, :] = (R_[6] * s_[1], R_[7] * s_[1], 0, -R_[8], R_[5], 0, -R_[6], R_[3], 0, -R_[7], R_[4],)
        A_[1, :] = (
            -R_[6] * s_[0], -R_[7] * s_[0], R_[8], 0, -R_[2], R_[6], 0, -R_[0], R_[7], 0, -R_[1],)
        A_[2, :] = (
            R_[3] * s_[0] - R_[0] * s_[1], R_[4] * s_[0] - R_[1] * s_[1], -R_[5], R_[2], 0, -R_[3], R_[0], 0, -R_[4],
            R_[1], 0)
        b_[:] = (R_[8] * s_[1], -R_[8] * s_[0], R_[5] * s_[0] - R_[2] * s_[1])
        b_ = -b_
        return A_, b_

    def __x2light_pin(x):
        assert x.shape == (2 + 9 * pin_num,), (x.shape, (2 + 9 * pin_num))
        lc = np.ones(3)
        lc[0:2] = x[0:2]
        #
        pins = np.zeros(shape=(pin_num, 3))
        for p in range(pin_num):
            pins[p, :] = x[2 + 9 * p:2 + 9 * p + 3]

        return lc, pins

    Qc = np.zeros(shape=(3 * pose_num * pin_num, 2 + 9 * pin_num))
    Bc = np.zeros(shape=(3 * pose_num * pin_num))
    for i in range(pose_num):
        for p in range(pin_num):
            A, b = __Ab_ij(Rs[i], projected_points[i, p])
            Qc[3 * pose_num * p + 3 * i:3 * pose_num * p + 3 * i + 3, 0:2] = A[:, 0:2]
            Qc[3 * pose_num * p + 3 * i:3 * pose_num * p + 3 * i + 3, 2 + 9 * p:2 + 9 * p + 9] = A[:, 2:]
            Bc[3 * pose_num * p + 3 * i:3 * pose_num * p + 3 * i + 3] = b

    from L1_solver.L1_residual_min import L1_residual_min
    X = L1_residual_min(np.matrix(Qc), np.matrix(Bc).T, MAX_ITER=IRLS_ITER, tol=IRLS_TOL)
    X = np.array(X).flatten()
    lc, P = __x2light_pin(X)
    #
    L = np.zeros(shape=(pose_num, 3))
    for l in range(pose_num):
        L[l, :] = Rs[l].T.dot(lc)
        L[l, :] /= L[l, 2]

    res = utils.error_reprojection_distant(projected_points, L, P)

    result = {}
    result["best_global_position"] = lc
    result["L"] = L
    result["P"] = P
    result["res"] = res

    return result


def solve_unified(projected_points, Rs, tvecs, init_L, init_P):
    pose_num, pin_num, _ = projected_points.shape
    assert Rs.shape == (pose_num, 3, 3)
    assert tvecs.shape == (pose_num, 3)

    def _fit_func(params, _):

        param_P = params[:pin_num * 3].reshape(pin_num, 3)
        param_global_light_position = params[pin_num * 3:]

        L = np.zeros(shape=(pose_num, 3), dtype=params.dtype)
        for l in range(pose_num):
            L[l, :] = Rs[l].T.dot(param_global_light_position) - Rs[l].T.dot(tvecs[l])

        res = projected_points.astype(params.dtype) - utils.project_unified(L, param_P)

        return np.r_[res.flatten(),].flatten()

    init_param = np.ones(shape=(pin_num * 3 + 3))
    P = init_param[:pin_num * 3].reshape(pin_num, 3)
    global_light_position = init_param[pin_num * 3:]

    ######################################################
    P[:] = init_P
    global_light_position[:] = Rs[0].dot(init_L[0, :]) + tvecs[0]
    #######################################################

    from scipy.optimize import leastsq
    x, cov_x, infodict, mesg, ier = leastsq(_fit_func, init_param.reshape(-1),
                                            args=[],
                                            xtol=XTOL_BA, maxfev=MAXFEV, full_output=True)

    if ier not in [1, 2, 3, 4]:
        warnings.warn("Solution not found: ier:{}, {}".format(ier, mesg))

    res = np.linalg.norm(infodict["fvec"])
    P = x[:pin_num * 3].reshape(pin_num, 3).astype(np.float64)
    global_light_position = x[pin_num * 3:].astype(np.float64)

    L = np.zeros(shape=(pose_num, 3))
    for l in range(pose_num):
        L[l, :] = Rs[l].T.dot(global_light_position) - Rs[l].T.dot(tvecs[l])

    res = utils.error_reprojection(projected_points, L, P)
    res = np.mean(res)

    result = {}
    result["best_global_position"] = global_light_position
    result["L"] = L
    result["P"] = P
    result["res"] = res

    return result


def solve_near(projected_points, Rs, tvecs, init_L, init_P):
    """
    Solve Eq. (5) in near light case

    :param np.ndarray projected_points:
    :param np.ndarray Rs:
    :param np.ndarray tvecs:
    :param np.ndarray init_L: initial estimation for light positions (all of l_i)
    :param np.ndarray init_P:initial estimation for pin positions
    :return: dict of estimated result
    """
    pose_num, pin_num, _ = projected_points.shape

    assert Rs.shape == (pose_num, 3, 3)
    assert tvecs.shape == (pose_num, 3)

    def _fit_func(params, _):
        """
        objective function for bundle adjustment
        """
        param_P = params[:pin_num * 3].reshape(pin_num, 3)
        param_global_light_position = params[pin_num * 3:]

        L = np.zeros(shape=(pose_num, 3))
        for l in range(pose_num):
            L[l, :] = Rs[l].T.dot(param_global_light_position) - Rs[l].T.dot(tvecs[l])

        res = projected_points - utils.project_near(L, param_P)

        return np.r_[res.flatten(),].flatten()

    # variables for optimization
    init_param = np.ones(shape=(pin_num * 3 + 3))
    P = init_param[:pin_num * 3].reshape(pin_num, 3)
    global_light_position = init_param[pin_num * 3:]

    # set initial estimation
    P[:] = init_P
    global_light_position[:] = Rs[0].dot(init_L[0, :]) + tvecs[0]

    # run optimization
    from scipy.optimize import leastsq
    x, cov_x, infodict, mesg, ier = leastsq(_fit_func, init_param.reshape(-1),
                                            args=[],
                                            xtol=XTOL_BA, maxfev=MAXFEV, full_output=True)

    if ier not in [1, 2, 3, 4]:
        warnings.warn("Solution not found: ier:{}, {}".format(ier, mesg))

    res = np.linalg.norm(infodict["fvec"])
    P = x[:pin_num * 3].reshape(pin_num, 3)
    global_light_position = x[pin_num * 3:]

    # calculate l_i from l, R_i and t_i
    L = np.zeros(shape=(pose_num, 3))
    for l in range(pose_num):
        L[l, :] = Rs[l].T.dot(global_light_position) - Rs[l].T.dot(tvecs[l])

    # calculate reprojection error
    res = utils.error_reprojection_near(projected_points, L, P)

    # store result to dict
    result = {}
    result["best_global_position"] = global_light_position
    result["L"] = L
    result["P"] = P
    result["res"] = res

    return result


def solve_distant(projected_points, Rs, tvecs, init_L, init_P):
    pose_num, pin_num, _ = projected_points.shape
    assert Rs.shape == (pose_num, 3, 3), Rs.shape
    assert tvecs.shape == (pose_num, 3), tvecs.shape
    assert init_L.shape == (pose_num, 3), init_L
    assert init_P.shape == (pin_num, 3), init_P

    def _fit_func(params, _):
        param_P = params[:pin_num * 3].reshape(pin_num, 3)
        param_global_light_position = params[pin_num * 3:]

        L = np.zeros(shape=(pose_num, 3))
        for l in range(pose_num):
            L[l, :] = Rs[l].T.dot(param_global_light_position)

        res = projected_points - utils.project_distant(L, param_P)

        return np.r_[res.flatten()].flatten()

    init_param = np.ones(shape=(pin_num * 3 + 3))
    P = init_param[:pin_num * 3].reshape(pin_num, 3)
    global_light_position = init_param[pin_num * 3:]

    P[:] = init_P
    global_light_position[:] = Rs[0].dot(init_L[0])

    from scipy.optimize import leastsq
    x, cov_x, infodict, mesg, ier = leastsq(_fit_func, init_param.reshape(-1),
                                            args=[],
                                            xtol=XTOL_BA, maxfev=MAXFEV, full_output=True)
    if ier not in [1, 2, 3, 4]:
        warnings.warn("Solution not found: ier:{}, {}".format(ier, mesg))

    P = x[:pin_num * 3].reshape(pin_num, 3)
    global_light_position = x[pin_num * 3:]

    global_light_position /= global_light_position[2]
    L = np.zeros(shape=(pose_num, 3))
    for l in range(pose_num):
        L[l, :] = Rs[l].T.dot(global_light_position)
        L[l, :] /= L[l, 2]

    res = utils.error_reprojection_distant(projected_points, L, P)

    result = {}
    result["best_global_position"] = global_light_position
    result["L"] = L
    result["P"] = P
    result["res"] = res

    return result
