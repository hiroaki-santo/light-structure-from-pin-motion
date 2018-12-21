#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import glob
import os
import time

import cv2
import joblib
import numpy as np

import utils

SHADOW_DETECTION_LAMS = [1.5, 0.6, 2.3, 1.2]


def detect(img, mask):
    m, n, _ = img.shape
    assert mask.shape == (m, n), "{}, ({},{})".format(mask.shape, m, n)

    bin_img = img.copy()
    bin_img = cv2.adaptiveThreshold(cv2.cvtColor(bin_img, cv2.COLOR_RGB2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 31, 3)
    bin_img[mask == 0] = 255
    kernel = np.ones(shape=(2, 2), dtype=np.uint8)
    bin_img = cv2.dilate(bin_img, kernel, iterations=3)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    template_size = 100
    template = np.ones(shape=(template_size, template_size), dtype=np.uint8) * 255
    result = cv2.matchTemplate(bin_img, template, method=cv2.TM_CCORR_NORMED)
    result[:] = 0
    print("[*] Matching start...")
    stime = time.time()
    for s in [int(template_size * 0.4), int(template_size * 0.3), int(template_size * 0.2)]:
        for l in np.linspace(0, template_size * 4, num=16).astype(np.int):
            l_ = l // template_size
            x = l % template_size
            if l_ == 0:
                l_origin = (x, 0)
            elif l_ == 1:
                l_origin = (x, template_size - 1)
            elif l_ == 2:
                l_origin = (template_size - 1, x)
            elif l_ == 3:
                l_origin = (0, x)

            template[:] = 255
            cv2.circle(template, (template_size // 2, template_size // 2), s, (0, 0, 0), -1)
            cv2.line(template, l_origin, (template_size // 2, template_size // 2), (0, 0, 0), s)

            result_ = cv2.matchTemplate(bin_img, template, method=cv2.TM_CCOEFF_NORMED)
            result += result_

    print("[*] Matching complete... {} s".format(time.time() - stime))
    detected_points = []
    detected_img = img.copy()
    result_ = result[:]
    while len(detected_points) == 0:
        for _ in range(25):
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_)
            img_coord = (max_loc[0] + template_size // 2, max_loc[1] + template_size // 2)

            if mask[img_coord[1], img_coord[0]] == 0:
                continue
            cv2.circle(detected_img, img_coord, 20, (0, 0, 0), 5)
            cv2.circle(result_, max_loc, template_size // 2, (0, 0, 0), -1)
            detected_points.append({"pt": img_coord, "raw_score": max_val})

    score_max = np.max([p["raw_score"] for p in detected_points])
    score_min = np.min([p["raw_score"] for p in detected_points])
    for p in detected_points:
        s = p["raw_score"] - score_min
        p["score"] = s / score_max

    return detected_points, bin_img


def select_shadow_point(candidate_points, img, bin_img, point_num, lams):
    if len(candidate_points) <= point_num:
        return np.array([np.array(p["pt"]) for p in candidate_points])

    r = 30
    for p in candidate_points:
        img_ = img.copy()
        bin_img_ = bin_img.copy()

        mask = np.zeros(shape=img_.shape, dtype=np.uint8)
        cv2.circle(mask, p["pt"], r, (1, 1, 1), -1)
        num_mask_pix = np.count_nonzero(mask[:, :, 0])

        img_ *= mask
        bin_img_ *= mask[:, :, 0]

        rgb_sum = np.sum(img_.reshape(-1, 3), axis=0)
        bin_sum = np.sum(bin_img_)
        p["rgb_sum"] = rgb_sum / num_mask_pix
        p["bin_sum"] = bin_sum / num_mask_pix

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=point_num, init='k-means++', n_init=10, max_iter=1000, random_state=0)
    c = km.fit_predict(np.array([p["pt"] for p in candidate_points]))

    dst_points = []
    for index in range(point_num):
        arg_candidates = np.argwhere(c == index).flatten().astype(np.uint8)
        bin_sums = np.array([candidate_points[i]["bin_sum"] for i in arg_candidates])
        rgb_sums = np.array([candidate_points[i]["rgb_sum"] for i in arg_candidates])
        scores = np.array([candidate_points[i]["score"] for i in arg_candidates])

        # our pin head is red.
        gray_sums = np.array([v[1] + v[2] for v in rgb_sums])
        rgb_ratios = np.array([v[0] / np.sum(v) for v in rgb_sums])
        bin_sums = bin_sums - np.min(bin_sums)
        bin_sums /= np.max(bin_sums)
        gray_sums = (gray_sums - np.min(gray_sums))
        gray_sums /= np.max(gray_sums)

        eval_scores = np.zeros(len(arg_candidates))
        lam0, lam1, lam2, lam3 = lams

        for i in range(len(arg_candidates)):
            eval_scores[i] = lam0 * scores[i] + lam1 * (1 - bin_sums[i]) + lam2 * 1. / rgb_ratios[i] + lam3 * (
                    1 - gray_sums[i])

        index = np.argmax(eval_scores)
        dst_points.append(candidate_points[arg_candidates[index]])

    ####################
    dst_img = img.copy()
    for pt in dst_points:
        cv2.circle(dst_img, pt["pt"], 25, (255, 0, 0), 2)

    return dst_points, dst_img


def detect_shadow(img, marker_coordinates, objPoints, pin_num, lams):
    assert len(marker_coordinates) == len(objPoints)

    objPoints_ = objPoints.copy() * 10
    H = utils.find_homography(marker_coordinates, objPoints_)
    o_size = (int(np.max(objPoints_[:, :, 0])), int(np.max(objPoints_[:, :, 1])))
    warped_img = cv2.warpPerspective(img, H, o_size)

    mask = np.ones(shape=(o_size[1], o_size[0]))

    left_top = objPoints_[30, 1, :].astype(np.int)
    right_bottom = objPoints_[4, 3, :].astype(np.int)
    margin = int(left_top[0] * 0.1)
    mask[:left_top[0] + margin, :] = 0
    mask[:, :left_top[1] + margin] = 0
    mask[right_bottom[1] - margin:, :] = 0
    mask[:, right_bottom[0] - margin:] = 0

    detected_points, bin_img = detect(warped_img, mask)
    points, debug_img = select_shadow_point(detected_points, warped_img, bin_img, point_num=pin_num, lams=lams)

    [cv2.circle(bin_img, p["pt"], 30, 128, 4) for p in points]

    dst_points = np.array([p["pt"] for p in points]).astype(float)
    dst_points /= 10.

    assert dst_points.shape == (pin_num, 2)

    return dst_points, debug_img, bin_img


def __file_name(path):
    file_name, ext = os.path.splitext(path)
    return file_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Shadow detection')
    parser.add_argument("--data_path", "-i", type=str)
    parser.add_argument("--pin_num", type=int, default=5)

    ARGS = parser.parse_args()

    o_path = os.path.join(ARGS.data_path, "tmp")
    if not os.path.exists(o_path):
        os.makedirs(o_path)

    camera_matrix, camera_dist = utils.load_camera_params(ARGS.data_path)

    img_paths = glob.glob(os.path.join(ARGS.data_path, "*.png"))
    img_paths = sorted(img_paths)
    img_paths = np.array(img_paths, dtype=str)
    imgs = [cv2.imread(path)[:, :, ::-1] for path in img_paths]
    imgs = [cv2.undistort(i, camera_matrix, camera_dist) for i in imgs]

    file_names = [__file_name(path) for path in img_paths]
    marker_coordinates_paths = [os.path.join(ARGS.data_path, "{}_marker_coordinates.npz".format(path))
                                for path in file_names]
    marker_coordinates = [np.load(path)["marker_coordinates"] for path in marker_coordinates_paths]
    board_objPoints = [np.load(path)["board_objPoints"] for path in marker_coordinates_paths]
    Rs = np.array([np.load(path)["R"] for path in marker_coordinates_paths])
    tvecs = np.array([np.load(path)["tvec"] for path in marker_coordinates_paths])
    tvecs = tvecs.reshape(-1, 3)

    pose_num = len(imgs)
    pin_num = ARGS.pin_num


    def __wrapper(*args, **kwargs):
        # points, debug_img, bin_img = detect_shadow()
        pose = kwargs.pop("pose")

        points, debug_img, bin_img = detect_shadow(*args, **kwargs)

        file_name, ext = os.path.splitext(os.path.basename(img_paths[pose]))
        cv2.imwrite(os.path.join(o_path, "{}_shadow_detected.png".format(file_name)), debug_img[:, :, ::-1])
        return (points, debug_img, bin_img), pose


    rvals = joblib.Parallel(n_jobs=-1, verbose=1)([joblib.delayed(__wrapper)(
        imgs[pose], marker_coordinates[pose], board_objPoints[pose], pin_num, lams=SHADOW_DETECTION_LAMS, pose=pose) for
        pose in range(pose_num)])

    projected_points_detected = np.ones(shape=(pose_num, pin_num, 3))
    for pose in range(pose_num):
        markers = marker_coordinates[pose]
        objPoints = board_objPoints[pose]
        img = imgs[pose]
        (points, debug_img, bin_img), pose = rvals[pose]
        if points is None:
            continue

        points_ = np.ones(shape=(pin_num, 3))
        points_[:, :2] = points
        projected_points_detected[pose, :, :] = points_

    ##
    # OUTPUT
    ##
    for pose in range(pose_num):
        file_name, ext = os.path.splitext(os.path.basename(img_paths[pose]))
        np.savetxt(os.path.join(ARGS.data_path, "{}_detected_label.txt".format(file_name)),
                   projected_points_detected[pose])
