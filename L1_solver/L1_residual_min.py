import warnings

import numpy as np
import scipy.linalg
import tqdm


def L1_residual_min(A, b, MAX_ITER=1000, tol=1.0e-8):
    """
    L1 residual minimization by iteratively reweighted least squares (IRLS)

        minimize ||Ax - b||_1

    """

    if A.shape[0] != b.shape[0]:
        raise ValueError("Inconsistent dimensionality between A and b")

    eps = 1.0e-8
    m, n = A.shape

    xold = np.matrix(np.ones((n, 1)), copy=False)
    W = np.matrix(np.identity(m), copy=False)

    if np.ndim(b) != 2 and b.shape[1] != 1:
        raise ValueError("b needs to be a vector")

    iter = 0

    with tqdm.tqdm(total=MAX_ITER) as pbar:
        while iter < MAX_ITER:
            pbar.update(1)
            iter = iter + 1
            # Solve the weighted least squares
            x = scipy.linalg.lstsq(W * A, W * b)[0]
            e = b - A * x
            res = scipy.linalg.norm(x - xold)
            # Termination criterion
            pbar.set_description("||x - xold||: {}".format(res))
            if res < tol:
                return x
            else:
                xold = x
            # Update weighting factor
            W = np.diag(np.asarray(1.0 / np.maximum(np.sqrt(np.fabs(e)), eps))[:, 0])
        warnings.warn("Exceeded the maximum number of iterations.")
        return x
