#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import print_function
from __future__ import unicode_literals

import argparse

import numpy as np

import methods
import utils
import utils_simulation


def simulation(pin_num, pose_num, light_board_distance=[400., 600.], pin_height=[20., 50.], types="near",
               noise_pose=0., noise_shadow=0.,
               ransac_num=10, ransac_iter=30,
               seed=None):
    assert types in ["near", "distant"], types

    if seed is not None:
        np.random.seed(seed)

    pin_coordinates = np.random.uniform(0, 200, size=(pin_num, 3))
    pin_coordinates[:, 2] = np.random.uniform(pin_height[0], pin_height[1], size=pin_num)

    if types == "near":
        global_light_position = np.random.uniform(-100., 100., size=3)
        global_light_position[2] = 0.

    elif types == "distant":
        light_theta = np.random.uniform(0, np.deg2rad(45))
        light_phi = np.random.uniform(0, np.pi * 2)
        global_light_position = utils.polar2xyz(light_theta, light_phi, 1.)
        global_light_position /= global_light_position[2]  # normalize the directional vector

    else:
        raise ValueError(types)

    projected_points, sim_data = utils_simulation.gen_simulation_data(pin_coordinates, global_light_position,
                                                                      types, pose_num=pose_num,
                                                                      light_board_distance=light_board_distance,
                                                                      seed=seed)

    Rs = np.array([data["R"] for data in sim_data])
    tvecs = np.array([data["tvec"] for data in sim_data])
    tvecs = tvecs.reshape(pose_num, 3)

    if noise_pose > 0.:
        for l in range(len(Rs)):
            noise_x, noise_y, noise_z = np.deg2rad(np.random.normal(0, noise_pose, size=3))
            noise_R = utils_simulation.gen_rotation_matrix(noise_x, noise_y, noise_z)
            Rs[l] = Rs[l].dot(noise_R)

    if noise_shadow > 0.:
        projected_points += np.random.normal(0, noise_shadow, size=projected_points.shape)
        projected_points[:, :, 2] = 1.

    if types == "near":
        init_result = methods.solution_near_linear(projected_points, Rs, tvecs)
        all_results = utils.ransac_wrapper(projected_points, Rs, tvecs, methods.solve_near,
                                           ransac_num=min(ransac_num, pose_num),
                                           iter=ransac_iter,
                                           init_P=init_result["P"],
                                           init_L=init_result["L"])
        result = utils.ransac_find_best_near(all_results, projected_points, Rs, tvecs)

    elif types == "distant":
        init_result = methods.solution_distant_linear(projected_points, Rs, tvecs)
        all_results = utils.ransac_wrapper(projected_points, Rs, tvecs, methods.solve_distant,
                                           ransac_num=min(ransac_num, pose_num),
                                           iter=ransac_iter,
                                           init_P=init_result["P"],
                                           init_L=init_result["L"])
        result = utils.ransac_find_best_distant(all_results, projected_points, Rs, tvecs)
    else:
        raise ValueError

    print("GT:")
    print("Pin head positions:")
    print(pin_coordinates)
    print("Light position:", global_light_position)

    print("=====")
    if types == "near":
        print("Convex")
        print("MAE:", np.linalg.norm(init_result["best_global_position"] - global_light_position))
        print("BA")
        print("MAE:", np.linalg.norm(result["best_global_position"] - global_light_position))
        print(result["best_global_position"])

    elif types == "distant":
        print("Convex")
        print("MAngE:", utils.ang_error_deg(init_result["best_global_position"], global_light_position))
        print("BA")
        print("MAngE:", utils.ang_error_deg(result["best_global_position"], global_light_position))
        print(result["best_global_position"])

    print("Pin head positions:")
    print(result["P"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Light Structure from Pin Motion')

    parser.add_argument("--sim_type", type=str, default="near", help="Type of synthetic data: near or distant.")
    parser.add_argument("--sim_noise_shadow", type=float, default=0.0,
                        help="The noise's standard deviation for shadow position.")
    parser.add_argument("--sim_noise_pose", type=float, default=0.0,
                        help="The noise's standard deviation for pose of board.")
    parser.add_argument("--sim_pose_num", type=int, default=10)
    parser.add_argument("--sim_pin_num", type=int, default=5)
    parser.add_argument("--sim_board_distance", type=float, default=500, help="t_z in Fig. 6.")
    parser.add_argument("--seed", type=int, default=-1, help="for np.random.seed()")

    parser.add_argument("--ransac_num", type=int, default=10)
    parser.add_argument("--ransac_iter", type=int, default=30)

    ARGS = parser.parse_args()

    seed = ARGS.seed if ARGS.seed >= 0 else None

    simulation(pin_num=ARGS.sim_pin_num, pose_num=ARGS.sim_pose_num,
               noise_shadow=ARGS.sim_noise_shadow, noise_pose=ARGS.sim_noise_pose,
               light_board_distance=[ARGS.sim_board_distance - 100., ARGS.sim_board_distance + 100.],
               ransac_num=ARGS.ransac_num, ransac_iter=ARGS.ransac_iter, types=ARGS.sim_type,
               seed=seed)
