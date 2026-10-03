"""
Core Path Guidance & Monocular Depth/Distance Corridor Analyzer.
Calculates metric distance for each obstacle, filters out far objects,
and delivers precise directional guidance (Left, In Front, Right) with proximity awareness.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
try:
    from . import config
except ImportError:
    import config

class NavigationGuidance:
    """Represents the computed guidance decision for the current video frame."""
    def __init__(self, command: str, direction_code: str, reason: str, is_critical: bool = False,
                 left_clear: bool = True, center_clear: bool = True, right_clear: bool = True,
                 detected_objects_summary: str = "", primary_object: str = "", primary_position: str = "in front",
                 primary_distance: float = 0.0):
        self.command = command                          # e.g., "Walk Straight", "Steer Left", "Stop"
        self.direction_code = direction_code             # "FORWARD", "LEFT", "RIGHT", "STOP"
        self.reason = reason                            # Detailed context e.g. "Chair on left (1.4m)"
        self.is_critical = is_critical                  # True if immediate collision danger
        self.left_clear = left_clear
        self.center_clear = center_clear
        self.right_clear = right_clear
        self.detected_objects_summary = detected_objects_summary
        self.primary_object = primary_object
        self.primary_position = primary_position        # "on your left", "in front", "on your right"
        self.primary_distance = primary_distance        # Distance in meters

class PathAnalyzer:
    """
    Analyzes objects with monocular distance triangulation and 3-zone spatial corridors (Left, Center, Right).
    Only obstacles within immediate walking range trigger steering or stop alerts.
    """
    def __init__(self):
        self.left_limit = config.ZONE_LEFT_LIMIT
        self.right_limit = config.ZONE_RIGHT_LIMIT
        self.stop_dist = config.DISTANCE_STOP_THRESHOLD
        self.steer_dist = config.DISTANCE_STEER_THRESHOLD
        self.ignore_dist = config.DISTANCE_IGNORE_THRESHOLD
        self.major_classes = config.MAJOR_OBSTACLE_CLASSES

    def estimate_distance(self, y2: int, box_height: int, frame_height: int) -> float:
        """
        Estimates metric distance (meters) to obstacle using ground-plane contact triangulation.
        Objects lower on the sensor (near user's feet) are much closer.
        """
        horizon_y = int(frame_height * 0.35)
        pixel_delta = max(15, y2 - horizon_y)
        dist = (config.CAMERA_HEIGHT_M * config.FOCAL_LENGTH_PIXELS) / float(pixel_delta)
        # Cap distance between 0.4m and 6.0m
        return round(max(0.4, min(6.0, dist)), 1)

    def analyze_frame(self, detections: List[Dict[str, Any]], frame_width: int, frame_height: int) -> Tuple[NavigationGuidance, List[Dict[str, Any]]]:
        processed_obstacles = []

        # Corridors for proximate obstacles only (<= steer_dist)
        zone_proximate_objects = {"left": [], "center": [], "right": []}

        for det in detections:
            box = det['box']
            cls_name = det.get('class_name', 'object').lower()
            conf = det.get('conf', 0.5)

            x1, y1, x2, y2 = box
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            area_ratio = (w * h) / float(frame_width * frame_height)
            center_x_norm = ((x1 + x2) / 2.0) / float(frame_width)
            bottom_y_norm = y2 / float(frame_height)

            # Calculate metric depth/distance
            distance_m = self.estimate_distance(y2, h, frame_height)

            # Determine spatial corridor tag
            if center_x_norm < self.left_limit:
                pos_zone = "LEFT"
                pos_phrase = "on your left"
            elif center_x_norm > self.right_limit:
                pos_zone = "RIGHT"
                pos_phrase = "on your right"
            else:
                pos_zone = "CENTER"
                pos_phrase = "in front"

            # Filter relevance: Major walking obstacle OR large foreground area
            is_relevant = (cls_name in self.major_classes) or (area_ratio >= 0.08) or (distance_m <= 1.5)
            if not is_relevant:
                continue

            # Check if object is in immediate walking range (<= 3.2m)
            is_proximate = distance_m <= self.ignore_dist
            is_critical = distance_m <= self.stop_dist

            obs_info = {
                'box': box,
                'class_name': cls_name,
                'conf': conf,
                'distance_m': distance_m,
                'is_close': is_critical,
                'is_proximate': is_proximate,
                'zone': pos_zone,
                'pos_phrase': pos_phrase
            }
            processed_obstacles.append(obs_info)

            # Assign proximate obstacles to zones
            if is_proximate:
                zone_key = pos_zone.lower()
                zone_proximate_objects[zone_key].append(obs_info)

        # Sort each zone's obstacles by closest distance (ascending)
        for z in zone_proximate_objects:
            zone_proximate_objects[z].sort(key=lambda item: item['distance_m'])

        center_obs = zone_proximate_objects["center"]
        left_obs = zone_proximate_objects["left"]
        right_obs = zone_proximate_objects["right"]

        # Determine true blockage based on nearest distance thresholds
        center_blocked = len(center_obs) > 0 and center_obs[0]['distance_m'] <= self.steer_dist
        left_blocked = len(left_obs) > 0 and left_obs[0]['distance_m'] <= (self.steer_dist * 0.85)
        right_blocked = len(right_obs) > 0 and right_obs[0]['distance_m'] <= (self.steer_dist * 0.85)

        left_clear = not left_blocked
        center_clear = not center_blocked
        right_clear = not right_blocked

        # Build summary of proximate environment
        summary_parts = []
        if center_obs:
            summary_parts.append(f"{center_obs[0]['class_name']} in front ({center_obs[0]['distance_m']}m)")
        if left_obs:
            summary_parts.append(f"{left_obs[0]['class_name']} on left ({left_obs[0]['distance_m']}m)")
        if right_obs:
            summary_parts.append(f"{right_obs[0]['class_name']} on right ({right_obs[0]['distance_m']}m)")
        detected_summary = ", ".join(summary_parts)

        # 1. CRITICAL STOP: Object very close in front (<= 1.25m) or path completely impassable
        if center_obs and center_obs[0]['distance_m'] <= self.stop_dist:
            closest = center_obs[0]
            guidance = NavigationGuidance(
                command=f"Stop - {closest['class_name'].capitalize()} Close",
                direction_code="STOP",
                reason=f"{closest['class_name'].capitalize()} in front ({closest['distance_m']}m)",
                is_critical=True,
                left_clear=left_clear,
                center_clear=False,
                right_clear=right_clear,
                detected_objects_summary=detected_summary,
                primary_object=closest['class_name'],
                primary_position="in front",
                primary_distance=closest['distance_m']
            )
            return guidance, processed_obstacles

        # 2. CLEAR PATH AHEAD: No proximate obstacle in center walking corridor (<= 2.6m)
        if center_clear:
            side_info = ""
            primary_obj = ""
            primary_pos = ""
            primary_dist = 0.0

            if left_obs and right_obs:
                side_info = f"{left_obs[0]['class_name']} on left, {right_obs[0]['class_name']} on right"
                primary_obj = left_obs[0]['class_name']
                primary_pos = "on your left"
                primary_dist = left_obs[0]['distance_m']
            elif left_obs:
                side_info = f"{left_obs[0]['class_name']} on left ({left_obs[0]['distance_m']}m)"
                primary_obj = left_obs[0]['class_name']
                primary_pos = "on your left"
                primary_dist = left_obs[0]['distance_m']
            elif right_obs:
                side_info = f"{right_obs[0]['class_name']} on right ({right_obs[0]['distance_m']}m)"
                primary_obj = right_obs[0]['class_name']
                primary_pos = "on your right"
                primary_dist = right_obs[0]['distance_m']
            else:
                side_info = "Path is clear"

            guidance = NavigationGuidance(
                command="Walk Straight",
                direction_code="FORWARD",
                reason=side_info,
                is_critical=False,
                left_clear=left_clear,
                center_clear=True,
                right_clear=right_clear,
                detected_objects_summary=detected_summary,
                primary_object=primary_obj,
                primary_position=primary_pos,
                primary_distance=primary_dist
            )
            return guidance, processed_obstacles

        # 3. CENTER OBSTACLE WITHIN STEERING RANGE (1.25m < Z <= 2.6m) -> Steer Left or Steer Right
        c_obj = center_obs[0]
        obj_name = c_obj['class_name']
        dist_m = c_obj['distance_m']

        if left_clear and not right_clear:
            guidance = NavigationGuidance(
                command="Steer Left",
                direction_code="LEFT",
                reason=f"{obj_name.capitalize()} ahead ({dist_m}m), clear on left",
                is_critical=False,
                left_clear=True,
                center_clear=False,
                right_clear=False,
                detected_objects_summary=detected_summary,
                primary_object=obj_name,
                primary_position="in front",
                primary_distance=dist_m
            )
        elif right_clear and not left_clear:
            guidance = NavigationGuidance(
                command="Steer Right",
                direction_code="RIGHT",
                reason=f"{obj_name.capitalize()} ahead ({dist_m}m), clear on right",
                is_critical=False,
                left_clear=False,
                center_clear=False,
                right_clear=True,
                detected_objects_summary=detected_summary,
                primary_object=obj_name,
                primary_position="in front",
                primary_distance=dist_m
            )
        elif left_clear and right_clear:
            # Choose side with furthest obstacle or completely empty
            l_dist = left_obs[0]['distance_m'] if left_obs else 99.0
            r_dist = right_obs[0]['distance_m'] if right_obs else 99.0
            steer_left = l_dist >= r_dist

            guidance = NavigationGuidance(
                command="Steer Left" if steer_left else "Steer Right",
                direction_code="LEFT" if steer_left else "RIGHT",
                reason=f"{obj_name.capitalize()} ahead ({dist_m}m), steer {'left' if steer_left else 'right'}",
                is_critical=False,
                left_clear=True,
                center_clear=False,
                right_clear=True,
                detected_objects_summary=detected_summary,
                primary_object=obj_name,
                primary_position="in front",
                primary_distance=dist_m
            )
        else:
            guidance = NavigationGuidance(
                command="Caution - Path Narrow",
                direction_code="STOP",
                reason=f"Narrow passage: {detected_summary}",
                is_critical=True,
                left_clear=False,
                center_clear=False,
                right_clear=False,
                detected_objects_summary=detected_summary,
                primary_object=obj_name,
                primary_position="in front",
                primary_distance=dist_m
            )

        return guidance, processed_obstacles
