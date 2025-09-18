# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import lap
import numpy as np
import scipy
import torch
from scipy.spatial.distance import cdist
from boxmot.utils.iou import AssociationFunction


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute IoU matrix.
    :param atlbrs: list[tlbr] | np.ndarray (N,4)
    :param btlbrs: list[tlbr] | np.ndarray (M,4)
    :return: (N,M) IoU matrix
    """
    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float32)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float32)
    return AssociationFunction.iou_batch(atlbrs, btlbrs) if len(atlbrs) and len(btlbrs) else np.zeros(
        (len(atlbrs), len(btlbrs)), dtype=np.float32
    )


def d_iou_distance(atracks, btracks):
    """
    Compute cost based on DIoU
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xyxy for track in atracks]
        btlbrs = [track.xyxy for track in btracks]

    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    _ious = AssociationFunction.diou_batch(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xyxy for track in atracks]
        btlbrs = [track.xyxy for track in btracks]

    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    _ious = AssociationFunction.iou_batch(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU of predicted boxes
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]

    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix


# --------- robust embedding distance (fixed) ---------

def _get_feat(obj, prefer):
    """Get a 1D float32 feature vector from an object with fallbacks."""
    for name in ([prefer] + ["curr_feat", "smooth_feat", "feat", "feature", "features"]):
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is None:
                continue
            a = np.asarray(val, dtype=np.float32)
            # squeeze to 1D if it's shape (1, D) or (D, 1)
            if a.ndim == 2 and 1 in a.shape:
                a = a.reshape(-1)
            elif a.ndim > 2:
                a = a.reshape(-1)
            elif a.ndim == 0:
                continue
            if a.size == 0:
                continue
            return a
    return None


def _stack_valid_1d(vectors):
    """Return (indices, matrix) where matrix is 2D [K, D], skipping invalid entries."""
    idx = []
    buf = []
    D = None
    # first pass: decide D from first valid vector
    for i, v in enumerate(vectors):
        if v is not None:
            D = int(v.size)
            break
    if D is None:
        return [], None
    for i, v in enumerate(vectors):
        if v is None:
            continue
        if v.size != D:
            # size mismatch -> skip
            continue
        buf.append(v.reshape(1, D))
        idx.append(i)
    if len(idx) == 0:
        return [], None
    mat = np.vstack(buf).astype(np.float32, copy=False)  # (K, D)
    return idx, mat


def embedding_distance(tracks, detections, metric="cosine"):
    """
    Compute appearance distance between tracks and detections.
    Robust to None/missing/shape-mismatch by returning 1.0 (far) for those cells.
    """
    n_t, n_d = len(tracks), len(detections)
    cost_matrix = np.ones((n_t, n_d), dtype=np.float32)
    if n_t == 0 or n_d == 0:
        return np.zeros((n_t, n_d), dtype=np.float32)

    # collect raw vectors
    t_vecs = [_get_feat(t, "smooth_feat") for t in tracks]
    d_vecs = [_get_feat(d, "curr_feat") for d in detections]

    # build 2D matrices for valid subsets
    t_idx, t_mat = _stack_valid_1d(t_vecs)
    d_idx, d_mat = _stack_valid_1d(d_vecs)

    if t_mat is None or d_mat is None:
        # nothing valid -> keep ones (far)
        return cost_matrix

    try:
        sub = cdist(t_mat, d_mat, metric)  # (len(t_idx), len(d_idx))
        sub = np.maximum(0.0, sub)  # safety clip
    except Exception:
        # if anything goes wrong, just return "far"
        return cost_matrix

    # place back into full matrix
    for ii, i in enumerate(t_idx):
        cost_matrix[i, d_idx] = sub[ii]

    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric="maha"
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_confs = np.array([det.conf for det in detections])
    det_confs = np.expand_dims(det_confs, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # fuse_sim = fuse_sim * (1 + det_confs) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_confs = np.array([det.conf for det in detections])
    det_confs = np.expand_dims(det_confs, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_confs
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2.0 * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0.0, float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`."""
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """Nearest neighbor distance metric (Euclidean)."""
    distances = _pdist(x, y)
    return np.maximum(0.0, torch.min(distances, axis=0)[0].numpy())


def _nn_cosine_distance(x, y):
    """Nearest neighbor distance metric (cosine)."""
    x_ = torch.from_numpy(np.asarray(x))
    y_ = torch.from_numpy(np.asarray(y))
    distances = _cosine_distance(x_, y_)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.
    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
