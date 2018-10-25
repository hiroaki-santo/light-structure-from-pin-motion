# Light Structure from Pin Motion
This is the project page for our ECCV 2018 paper 'Light Structure from Pin Motion: Simple and Accurate Point Light Calibration for Physics-based Modeling'
by Hiroaki Santo, Michael Waechter, Masaki Samejima, Yusuke Sugano, and Yasuyuki Matsushita.
You can find links to our [paper](http://www-infobiz.ist.osaka-u.ac.jp/wp-content/uploads/paper/pdf/Santo_Light_Structure_from_Pin_Motion_ECCV_2018_paper.pdf),
[supplemental material](http://www-infobiz.ist.osaka-u.ac.jp/wp-content/uploads/paper/pdf/Santo_Light_Structure_from_Pin_Motion_ECCV_2018_supplemental.pdf),
and [Youtube video](https://www.youtube.com/watch?v=WWcVqY4XqLM).

If you use our paper or code for research purposes, please cite our paper:
```
@inproceedings{Santo_2018_ECCV,
	title = {Light Structure from Pin Motion: Simple and Accurate Point Light Calibration for Physics-based Modeling},
	author = {Hiroaki Santo and Michael Waechter and Masaki Samejima and Yusuke Sugano and Yasuyuki Matsushita},
	booktitle = {European Conference on Computer Vision (ECCV)},
	year = {2018},
}
```

# How to Run

Currently, this repository only contains the code for generating simulation data and the mathematical part of our solution method.

You can get help on how to run the code:
```
$ python calibration.py --help
```

One example call for generating a synthetic dataset and solving it would be
```
$ python calibration.py --sim_type near --sim_pin_num 5 --sim_pose_num 10 --sim_noise_shadow 0.01 --sim_board_distance 500
```
The program outputs the ground truth as well as the estimated result for the light position/direction and the pin head positions.

The file ``methods.py`` contains the functions that estimate the light source and pin head positions
from ``projected_points`` (the shadow positions), ``Rs`` (the various rotations of the calibration board), and ``tvecs`` (translations of the calibration board).