"""
Multi-Vehicle Motion Tracking, Optical Area Expansion & Time-to-Collision (TTC) Engine.
Maintains rolling time-series state per vehicle, estimates approach velocity, and determines threat levels.
"""

import time
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional

try:
    from . import config
    from .vehicle_detector import DetectedVehicle
except ImportError:
    import config
    from vehicle_detector import DetectedVehicle

def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)

class VehicleTrack:
    """Represents a tracked vehicle over continuous frames."""
    def __init__(self, track_id: int, initial_detection: DetectedVehicle):
        self.track_id = track_id
        self.class_name = initial_detection.class_name
        self.corridor = initial_detection.corridor
        self.last_seen_time = time.time()
        self.disappeared_frames = 0
        
        # Current frame attributes
        self.current_box = initial_detection.box_xyxy
        self.current_distance = initial_detection.distance_meters
        self.confidence = initial_detection.confidence

        # Rolling history buffers: (timestamp, area, distance, center_x, center_y)
        self.history = deque(maxlen=config.HISTORY_BUFFER_SIZE)
        self.history.append((
            self.last_seen_time,
            initial_detection.area,
            initial_detection.distance_meters,
            initial_detection.center_x,
            initial_detection.center_y
        ))

        # Dynamic motion metrics
        self.expansion_rate = 0.0     # fraction / second
        self.relative_velocity = 0.0   # meters / second (positive = approaching)
        self.ttc_seconds = float("inf")
        self.threat_level = "STATIONARY"  # 'CRITICAL', 'WARNING', 'SAFE_APPROACH', 'STATIONARY', 'RECEDING'

    def update(self, detection: DetectedVehicle):
        now = time.time()
        self.last_seen_time = now
        self.disappeared_frames = 0
        self.current_box = detection.box_xyxy
        self.current_distance = detection.distance_meters
        self.confidence = detection.confidence
        self.corridor = detection.corridor
        self.class_name = detection.class_name

        self.history.append((
            now,
            detection.area,
            detection.distance_meters,
            detection.center_x,
            detection.center_y
        ))

        self._compute_motion_metrics()

    def _compute_motion_metrics(self):
        """Calculates optical area expansion, velocity, TTC, and threat level from history."""
        if len(self.history) < config.MIN_FRAMES_FOR_TTC:
            self.threat_level = "STATIONARY"
            return

        old_time, old_area, old_dist, old_cx, old_cy = self.history[0]
        cur_time, cur_area, cur_dist, cur_cx, cur_cy = self.history[-1]

        dt = max(0.001, cur_time - old_time)

        # 1. Optical Area Growth Rate: E = (Area_cur - Area_old) / (Area_old * dt)
        raw_expansion = (cur_area - old_area) / (max(1.0, old_area) * dt)
        self.expansion_rate = round(float(raw_expansion), 3)

        # 2. Relative Velocity (m/s): Positive means getting closer
        raw_velocity = (old_dist - cur_dist) / dt
        self.relative_velocity = round(float(raw_velocity), 2)

        # 3. Time-to-Collision (TTC) Estimation
        if self.relative_velocity > 0.3:
            raw_ttc = cur_dist / self.relative_velocity
            self.ttc_seconds = round(max(0.2, min(30.0, raw_ttc)), 1)
        elif self.expansion_rate > config.EXPANSION_RATE_THRESHOLD:
            # Fallback TTC derived from optical expansion formula: TTC = 2 / expansion_rate
            raw_ttc = 2.0 / max(0.01, self.expansion_rate)
            self.ttc_seconds = round(max(0.2, min(30.0, raw_ttc)), 1)
        else:
            self.ttc_seconds = float("inf")

        # 4. Threat Level Determination
        is_approaching = (self.expansion_rate >= config.EXPANSION_RATE_THRESHOLD or self.relative_velocity >= 0.5)
        is_receding = (self.expansion_rate <= config.RECEDING_RATE_THRESHOLD or self.relative_velocity <= -0.5)

        if is_approaching:
            if self.ttc_seconds <= config.CRITICAL_TTC_SECONDS or (cur_dist <= config.CRITICAL_DISTANCE_METERS and self.corridor == "Center"):
                self.threat_level = "CRITICAL"
            elif self.ttc_seconds <= config.WARNING_TTC_SECONDS or cur_dist <= config.WARNING_DISTANCE_METERS:
                self.threat_level = "WARNING"
            else:
                self.threat_level = "SAFE_APPROACH"
        elif is_receding:
            self.threat_level = "RECEDING"
        else:
            self.threat_level = "STATIONARY"

class VehicleMotionTracker:
    def __init__(self):
        self.next_track_id = 1
        self.tracks: Dict[int, VehicleTrack] = {}

    def update_tracks(self, detections: List[DetectedVehicle]) -> List[VehicleTrack]:
        """
        Associates current frame detections with existing tracks using IOU and spatial proximity.
        """
        current_time = time.time()

        # Age and remove stale tracks that haven't been seen for > 1.2 seconds
        expired_ids = [
            tid for tid, trk in self.tracks.items()
            if (current_time - trk.last_seen_time) > 1.2 or trk.disappeared_frames > 15
        ]
        for tid in expired_ids:
            del self.tracks[tid]

        if not detections:
            for trk in self.tracks.values():
                trk.disappeared_frames += 1
            return list(self.tracks.values())

        # Build IOU cost matrix between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        matched_detections = set()
        matched_tracks = set()

        if track_ids:
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for t_idx, tid in enumerate(track_ids):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = compute_iou(self.tracks[tid].current_box, det.box_xyxy)

            # Greedy match highest IOU pairs
            while True:
                max_val = np.max(iou_matrix)
                if max_val < config.IOU_TRACK_THRESHOLD:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[t_idx]
                self.tracks[tid].update(detections[d_idx])
                matched_tracks.add(tid)
                matched_detections.add(d_idx)
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                new_track = VehicleTrack(self.next_track_id, det)
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        # Increment disappeared counter for unmatched existing tracks
        for tid in track_ids:
            if tid not in matched_tracks:
                self.tracks[tid].disappeared_frames += 1

        return list(self.tracks.values())
