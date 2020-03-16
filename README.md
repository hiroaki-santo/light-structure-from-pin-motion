# Light Structure from Pin Motion
This is the project page for our IJCV paper 'Light Structure from Pin Motion: Geometric point light source calibration' by Hiroaki Santo, Michael Waechter, Wen-Yan Lin, Yusuke Sugano, and Yasuyuki Matsushita.
Here you can find a link to our [preprint version](http://www-infobiz.ist.osaka-u.ac.jp/wp-content/uploads/2020/02/santo2020light.pdf).

An earlier version was presented at ECCV 2018: 'Light Structure from Pin Motion: Simple and Accurate Point Light Calibration for Physics-based Modeling' by Hiroaki Santo, Michael Waechter, Masaki Samejima, Yusuke Sugano, and Yasuyuki Matsushita.
Here you can find links to our [paper](http://www-infobiz.ist.osaka-u.ac.jp/wp-content/uploads/paper/pdf/Santo_Light_Structure_from_Pin_Motion_ECCV_2018_paper.pdf), [supplemental material](http://www-infobiz.ist.osaka-u.ac.jp/wp-content/uploads/paper/pdf/Santo_Light_Structure_from_Pin_Motion_ECCV_2018_supplemental.pdf), and [Youtube video](https://www.youtube.com/watch?v=WWcVqY4XqLM).

If you use our papers or code for research purposes, please cite our papers:
```
@article{santo2020light,
	title = {Light structure from pin motion: Geometric point light source calibration},
	author = {Hiroaki Santo and Michael Waechter and Wen-Yan Lin and Yusuke Sugano and Yasuyuki Matsushita},
	journal = {International Journal of Computer Vision (IJCV)},
	doi = {10.1007/s11263-020-01312-3},
	year = {2020}
}

@inproceedings{santo2018light,
	title = {Light Structure from Pin Motion: Simple and Accurate Point Light Calibration for Physics-based Modeling},
	author = {Hiroaki Santo and Michael Waechter and Masaki Samejima and Yusuke Sugano and Yasuyuki Matsushita},
	booktitle = {European Conference on Computer Vision (ECCV)},
	year = {2018}
}
```

# How to Run

You can get help on how to run the code:
```
$ python calibration.py --help
```

## Simulation

One example call for generating a synthetic dataset and solving it would be
```
$ python calibration.py --sim_type near --pin_num 5 --sim_pose_num 10 --sim_noise_shadow 0.01 --sim_board_distance 500
```
The program outputs the ground truth as well as the estimated result for the light position/direction and the pin head positions.

## Real-world data

### Data Preparation

You need to prepare following files:
 * Images (``*.png``)
 * ``board_size.txt`` (Information of 2D marker. Two lines text file: Length of a marker and separation.)
 * ``params_*.npz`` (Result of camera calibration. Keywords with ``intrinsic`` and ``dist`` (distortion) should be provided.)

Please put all of them into one directory (``DATA_PATH``).

### (Camera Calibration)

We provide ``calibration_camera_aruco.py`` for camera calibration with ArUco markers.
You can use images of our calibration target, but we recommend to use the calibration target with full markers for the camera.

```
$ python calibration_camera_aruco.py -i IMAGES_DIR
```

### Pre-processing

Detecting 2D markers and shadows by using following two codes. Output files are stored to ``DATA_PATH``.

```
$ python detect_markers.py -i DATA_PATH
```

```
$ python detect_shadows.py -i DATA_PATH
```

Our shadow detection method uses a simple template-matching technique.
The program outputs the images to ``./DATA_PATH/tmp`` which illustrate the detected shadow points.

### Estimation

```
$ python calibration.py -i DATA_PATH
```

The program outputs the estimated light source position/direction and the pin head positions.

The file ``methods.py`` contains the functions that estimate the light source and pin head positions
from ``projected_points`` (the shadow positions), ``Rs`` (the various rotations of the calibration board), and ``tvecs`` (translations of the calibration board).
