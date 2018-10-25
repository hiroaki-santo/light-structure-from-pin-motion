#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import utils


def gen_rotation_matrix(rx, ry, rz):
    Rx = np.matrix([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])

    Ry = np.matrix([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.matrix([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

    return Rx.dot(Ry).dot(Rz)


def gen_simulation_data(pin_coordinates, global_light_position, types, pose_num=10, light_board_distance=[400, 600],
                        seed=None):
    """

    :param pin_coordinates: [pin_num, 3]
    :param global_light_position: [3,] 3d position or direction in the world coordinates system
    :param pose_num:
    :param light_board_distance: [2, ] range of t_z
    :param types: light source type
    :param seed: seed for rand
    :return:
    """
    if seed is not None:
        np.random.seed(seed)

    pin_num = len(pin_coordinates)
    assert len(global_light_position) == 3
    assert pin_coordinates.shape == (pin_num, 3)
    assert types in ["near", "distant"]

    sim_data = []
    for l in range(pose_num):
        rx, ry, rz = np.deg2rad(np.random.uniform(-30, 30, size=3))
        R = gen_rotation_matrix(rx, ry, rz)
        if types == "near":
            t = np.random.uniform(-300, 300, size=3)
            t[2] = np.random.uniform(light_board_distance[0], light_board_distance[1])
        else:
            t = np.zeros(shape=3)

        board_light_position = R.dot(global_light_position) + t
        board_light_position = np.array(board_light_position).flatten()

        shadow_positions = np.ones(shape=(pin_num, 3))
        for p in range(pin_num):
            if types == "near":
                s = utils.project_near(board_light_position.reshape(1, 3), pin_coordinates[p].reshape(1, 3))
            elif types == "distant":
                s = utils.project_distant(board_light_position.reshape(1, 3), pin_coordinates[p].reshape(1, 3))
            s = s.reshape(3)
            shadow_positions[p, 0:2] = s[0:2]

        data = {"global_light_position": global_light_position, "R": R.T, "tvec": -R.T.dot(t),
                "board_light_position": board_light_position, "pin_coordinates": pin_coordinates,
                "shadow_position": shadow_positions}
        sim_data.append(data)

    projected_points = np.zeros(shape=(pose_num, pin_num, 3))
    for l in range(pose_num):
        data = sim_data[l]
        for p in range(pin_num):
            projected_points[l, p, :] = data["shadow_position"][p]

    return projected_points, sim_data
