from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np


class ActionDetector:

    def __init__(self) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        self.prev_center: Optional[Tuple[float, float]] = None
        self.prev_ear: Optional[float] = None
        self.blink_cooldown = 0
        self.smile_cooldown = 0
        
        self.movement_threshold = 25
        self.blink_threshold = 0.21
        self.smile_threshold = 0.015
        
        self.LEFT_EYE_UPPER = [159, 145]
        self.LEFT_EYE_LOWER = [23, 130]
        self.LEFT_EYE_LEFT = 33
        self.LEFT_EYE_RIGHT = 133
        
        self.RIGHT_EYE_UPPER = [386, 374]
        self.RIGHT_EYE_LOWER = [253, 359]
        self.RIGHT_EYE_LEFT = 362
        self.RIGHT_EYE_RIGHT = 263
        
        self.MOUTH_CORNER_LEFT = 61
        self.MOUTH_CORNER_RIGHT = 291

    def detect_actions(self, frame: cv2.Mat, face_box: Tuple[int, int, int, int]) -> List[str]:
        actions = []
        
        if self.blink_cooldown > 0:
            self.blink_cooldown -= 1
        if self.smile_cooldown > 0:
            self.smile_cooldown -= 1
        
        landmarks = self._get_landmarks(frame, face_box)
        if landmarks is None:
            return actions
        
        movement = self._detect_movement(landmarks, frame.shape)
        if movement:
            actions.append(movement)
        
        if self._detect_blink(landmarks, frame.shape):
            actions.append("blink")
        
        if self._detect_smile(landmarks, frame.shape):
            actions.append("smile")
        
        return actions

    def _get_landmarks(self, frame: cv2.Mat, face_box: Tuple[int, int, int, int]) -> Optional[any]:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        x1, y1, x2, y2 = face_box
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        height, width = frame.shape[:2]
        
        best_landmarks = None
        min_distance = float('inf')
        
        for face_landmarks in results.multi_face_landmarks:
            nose = face_landmarks.landmark[1]
            nose_x = nose.x * width
            nose_y = nose.y * height
            
            distance = np.sqrt((nose_x - box_center_x)**2 + (nose_y - box_center_y)**2)
            
            if x1 <= nose_x <= x2 and y1 <= nose_y <= y2:
                if distance < min_distance:
                    min_distance = distance
                    best_landmarks = face_landmarks
        
        return best_landmarks

    def _detect_movement(self, landmarks: any, frame_shape: Tuple[int, int, int]) -> Optional[str]:
        height, width = frame_shape[:2]
        
        nose_tip = landmarks.landmark[1]
        center_x = nose_tip.x * width
        center_y = nose_tip.y * height
        
        if self.prev_center is not None:
            dx = center_x - self.prev_center[0]
            
            if abs(dx) > self.movement_threshold:
                self.prev_center = (center_x, center_y)
                if dx > 0:
                    return "moved_right"
                else:
                    return "moved_left"
        
        self.prev_center = (center_x, center_y)
        return None

    def _detect_blink(self, landmarks: any, frame_shape: Tuple[int, int, int]) -> bool:
        if self.blink_cooldown > 0:
            return False
        
        height, width = frame_shape[:2]
        
        left_ratio = self._eye_aspect_ratio(landmarks, self.LEFT_EYE_UPPER, self.LEFT_EYE_LOWER, width, height)
        right_ratio = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE_UPPER, self.RIGHT_EYE_LOWER, width, height)
        
        if left_ratio is None or right_ratio is None:
            return False
        
        current_ear = (left_ratio + right_ratio) / 2.0
        
        if self.prev_ear is not None:
            if current_ear < self.blink_threshold and self.prev_ear > self.blink_threshold:
                self.blink_cooldown = 8
                self.prev_ear = current_ear
                return True
        
        self.prev_ear = current_ear
        return False

    def _eye_aspect_ratio(self, landmarks: any, upper_indices: List[int], lower_indices: List[int], width: int, height: int) -> Optional[float]:
        try:
            upper_points = [landmarks.landmark[i] for i in upper_indices]
            lower_points = [landmarks.landmark[i] for i in lower_indices]
            
            vertical_dist = 0.0
            for up, low in zip(upper_points, lower_points):
                dy = abs(up.y - low.y) * height
                vertical_dist += dy
            vertical_dist /= len(upper_points)
            
            if upper_indices == self.LEFT_EYE_UPPER:
                left_idx, right_idx = 33, 133
            else:
                left_idx, right_idx = 362, 263
            
            left_point = landmarks.landmark[left_idx]
            right_point = landmarks.landmark[right_idx]
            horizontal_dist = abs(right_point.x - left_point.x) * width
            
            if horizontal_dist == 0:
                return None
            
            return vertical_dist / horizontal_dist
        except Exception:
            return None

    def _detect_smile(self, landmarks: any, frame_shape: Tuple[int, int, int]) -> bool:
        if self.smile_cooldown > 0:
            return False
            
        height, width = frame_shape[:2]
        
        try:
            left_corner = landmarks.landmark[self.MOUTH_CORNER_LEFT]
            right_corner = landmarks.landmark[self.MOUTH_CORNER_RIGHT]
            upper_lip = landmarks.landmark[13]
            lower_lip = landmarks.landmark[14]
            
            mouth_center_y = (upper_lip.y + lower_lip.y) / 2
            left_elevation = mouth_center_y - left_corner.y
            right_elevation = mouth_center_y - right_corner.y
            avg_elevation = (left_elevation + right_elevation) / 2
            
            if avg_elevation > self.smile_threshold:
                self.smile_cooldown = 8
                return True
            
            return False
        except Exception:
            return False

    def reset(self) -> None:
        self.prev_center = None
        self.prev_ear = None
        self.blink_cooldown = 0
        self.smile_cooldown = 0

    def __del__(self) -> None:
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
