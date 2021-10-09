import bisect
import math

import numpy as np

min_correspondences = 2


def normalize(F):
    """fixes F's norm and sign to remove the scaling ambiguity"""
    return F / (np.linalg.norm(F) * np.sign(F[2, 0]))


def homogenize(v):
    if v.ndim == 1:
        v = np.expand_dims(v, 1)
    return np.vstack((v, np.ones((1, v.shape[1]))))


def estimate(pts1, pts2, homogeneity_threshold=1e-1, return_singular_values=None, degeneracy_threshold=1e-5):
    """estimates a fundamental shadow matrix from >=2 point correspondences"""
    num_unknowns = 3
    num_points = pts1.shape[0]

    if not (pts1.shape == pts2.shape and num_points >= min_correspondences and pts1.shape[1] == 2):
        raise ValueError(("pt1 and pt2 should have the same shape, namely (>={}, 2), but they have "
                          "pt1.shape == {} and pt2.shape == {}").format(min_correspondences, pts1.shape, pts2.shape))

    # Hartley normalization
    all_points = np.vstack((pts1, pts2))
    shift = np.mean(all_points, axis=0)  # shift and scale have shape (1, 2) and normalize x and y differently
    all_points -= shift
    scale = np.sqrt(np.mean(all_points ** 2, axis=0))
    all_points /= scale
    pts1, pts2 = all_points[:num_points, :], all_points[num_points:, :]

    # prepare and solve homogeneous system
    u, v = pts1[:, 0], pts1[:, 1]
    u_prime, v_prime = pts2[:, 0], pts2[:, 1]
    A = np.column_stack((u * v_prime - v * u_prime, u - u_prime, v - v_prime))
    assert A.shape == (pts1.shape[0], num_unknowns), 'shape of A is {}'.format(A.shape)
    U, S, V = np.linalg.svd(A, full_matrices=True)
    assert V.shape == (num_unknowns, num_unknowns)
    if return_singular_values is not None:
        return_singular_values.resize(S.shape, refcheck=False)
        np.copyto(return_singular_values, S)

    # check if the system had too many eigenvalues a) too far from zero or b) too close to zero
    if S.size > num_unknowns - 1 and abs(S[num_unknowns - 1]) > homogeneity_threshold:
        return None
    if S[num_unknowns - 2] < degeneracy_threshold:
        print('S={} has too many very small singular value. The point configuration may be near-degenerate\n'.format(S))

    # extract solution
    t = V[-1, :]
    F = np.array([[0, t[0], t[1]],
                  [-t[0], 0, t[2]],
                  [-t[1], -t[2], 0]])

    # Hartley denormalization
    condition_mat = np.array([[1 / scale[0], 0, -shift[0] / scale[0]],
                              [0, 1 / scale[1], -shift[1] / scale[1]],
                              [0, 0, 1]])
    F = condition_mat.T @ F @ condition_mat

    return normalize(F)


def residuals_for_known_corresp(pts1, pts2, F, permutation_matrix=None):
    """
    compute abs(pt2.T @ F @ pt1) for all points
    """
    assert pts1.shape == pts2.shape and pts1.shape[0] == 2 and F.shape == (3, 3) \
           and (permutation_matrix is None or permutation_matrix.shape == (pts1.shape[1], pts1.shape[1])), \
        'pts1.shape: {}, pts2.shape: {}, F.shape: {}'.format(pts1.shape, pts2.shape, F.shape)
    pts1, pts2 = homogenize(pts1), homogenize(pts2)
    if permutation_matrix is not None:
        pts2 = pts2 @ permutation_matrix
    return np.abs(np.sum((pts2.T @ F) * pts1.T, axis=1))  # vectorized form of abs(pt2.T @ F @ pt1) for all points


def residuals_for_unknown_corresp(pts1, pts2, F, upper_bound=math.inf, num_possible_matches=0):
    """
    :param pts1: points in the left image
    :param pts2: points in the right image
    :param F: fundamental matrix
    :param upper_bound: if the sum of the residuals is larger than upper_bound, None is returned instead.
    this is a short-cut to not compute all residuals in case we already know an upper bound of the correct residual,
    e.g. from a previous RANSAC iteration that gave a better F-matrix. If you don't know what to set here, leave
    the default and there will be no speedup, but also no harm done.
    :param num_possible_matches: the number of point correspondences that the given F-matrix should explain (if it is
    correct). This is needed for the upper_bound short-cut described above. If you don't know what to set here, leave
    the default and there will be no speedup, but also no harm done.
    :return: a list where each entry is the smallest possible residual for each entry in pts1.
    returns None if the F-matrix can't be correct, because the sum of residuals is larger than the upper_bound
    """
    assert pts1.shape[0] == 2 and pts2.shape[0] == 2 and F.shape == (3, 3)
    num_points = pts1.shape[1]

    pts1, pts2 = homogenize(pts1), homogenize(pts2)

    best_residuals = []
    best_residuals_sorted = []
    best_correspondences = []
    for idx1 in range(num_points):
        if num_possible_matches - (num_points - idx1) >= 1 and \
                sum(best_residuals_sorted[:num_possible_matches - (num_points - idx1)]) > upper_bound:
            return None, None
        residuals = np.abs(np.sum((pts2.T @ F) * pts1[:, idx1], axis=1))  # vectorized form of abs(pts2.T @ F @ pt1)
        best_correspondences.append(residuals.argmin())
        best_residuals.append(residuals[best_correspondences[-1]])
        bisect.insort(best_residuals_sorted, best_residuals[-1])

    return best_residuals, best_correspondences
