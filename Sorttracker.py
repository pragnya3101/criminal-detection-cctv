
import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    """
    Minimal linear Kalman filter (numpy-only, no filterpy dependency).

    Standard predict/update equations:
      Predict:  x = F x            P = F P F^T + Q
      Update:   y = z - H x        (innovation)
                S = H P H^T + R
                K = P H^T S^-1     (Kalman gain)
                x = x + K y
                P = (I - K H) P
    """

    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        self.Q = np.eye(dim_x)
        self._I = np.eye(dim_x)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.asarray(z).reshape((self.dim_z, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self._I - K @ self.H) @ self.P


def iou(bb_test, bb_gt):
    """Intersection-over-union of two [x1,y1,x2,y2] boxes."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    intersection = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_test + area_gt - intersection
    return intersection / union if union > 0 else 0.0


def bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [cx, cy, scale(area), aspect_ratio] for the KF state."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h != 0 else 0.0
    return np.array([cx, cy, s, r]).reshape((4, 1))


def z_to_bbox(z):
    """[cx, cy, scale, aspect_ratio] -> [x1, y1, x2, y2]."""
    w = np.sqrt(max(z[2] * z[3], 0.0))
    h = z[2] / w if w != 0 else 0.0
    cx, cy = z[0], z[1]
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]).reshape((1, 4))


class KalmanBoxTracker:
   
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition: constant velocity model
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # Measurement function: we only observe cx, cy, scale, aspect ratio
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Measurement noise — trust position/scale more than we'd trust
        # a noisy aspect ratio estimate from a small face crop
        self.kf.R *= 1.0
        self.kf.R[2:, 2:] *= 10.0

        # Initial state uncertainty — high for unobserved velocities
        self.kf.P *= 10.0
        self.kf.P[4:, 4:] *= 1000.0

        # Process noise — velocities can drift a bit frame to frame
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.Q[-1, -1] *= 0.01

        self.kf.x[:4] = bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
       

    def predict(self):
        """Advance the filter one frame with no new measurement."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0  # don't let predicted scale go negative
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(z_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return z_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    # Hungarian algorithm minimizes cost, so we negate IOU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(row_ind, col_ind)))

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for d, t in matched_indices:
        if iou_matrix[d, t] < iou_threshold:
            unmatched_detections.append(d)
            unmatched_trackers.append(t)
        else:
            matches.append([d, t])

    matches = np.array(matches) if len(matches) > 0 else np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
   
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        
        self.frame_count += 1

        predicted = np.zeros((len(self.trackers), 4))
        to_delete = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            predicted[t] = pos
            if np.any(np.isnan(pos)):
                to_delete.append(t)

        predicted = np.ma.compress_rows(np.ma.masked_invalid(predicted))
        for t in reversed(to_delete):
            self.trackers.pop(t)

        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, predicted, self.iou_threshold
        )

        for d, t in matches:
            self.trackers[t].update(detections[d])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[i]))

        results = []
        for trk in reversed(self.trackers):
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()[0]
                results.append(np.concatenate((d, [trk.id])).reshape(1, -1))

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        return np.concatenate(results) if results else np.empty((0, 5))