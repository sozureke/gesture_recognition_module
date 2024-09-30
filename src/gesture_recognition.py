import cv2
import mediapipe as mp
from typing import List, Tuple
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from .gestures import Gesture
from . import config
import math
from collections import deque


class GestureRecognition:
    def __init__(self) -> None:
        self.mp_hands: mp.solutions.hands = mp.solutions.hands
        self.hands: mp.solutions.hands.Hands = self.mp_hands.Hands(
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw: mp.solutions.drawing_utils = mp.solutions.drawing_utils
        self.rotate_360_position_history = deque(maxlen=30)
        self.rotate_360_rotation_threshold = 360  
        self.rotate_360_direction = None
        self.rotate_360_rotation_count = 0
        self.rotate_360_cooldown = 0

    def process_frame(self, img: cv2.Mat) -> Tuple[List[Gesture], cv2.Mat]:
        if self.rotate_360_cooldown > 0:
            self.rotate_360_cooldown -= 1
        img_rgb: cv2.Mat = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results: mp.solutions.hands.Processed = self.hands.process(img_rgb)

        gestures = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                gesture = self.get_gesture(hand_landmarks)
                gestures.append(gesture)

        return gestures, img

    def get_gesture(self, hand_landmarks: NormalizedLandmarkList) -> Gesture:
        gesture_checks = [
            (self.is_rotate_360, Gesture.ROTATE_360),
            (self.is_tilt_left, Gesture.TILT_LEFT),
            (self.is_tilt_right, Gesture.TILT_RIGHT),
            (self.is_attention, Gesture.ATTENTION),
            (self.is_dance, Gesture.DANCE),
            (self.is_listen_message, Gesture.LISTEN_MESSAGE),
            (self.is_stop_hand, Gesture.STOP),
            (self.is_pinch, Gesture.PINCH),
            (self.is_fist, Gesture.FIST),
            (self.is_fingers_forward, Gesture.FINGERS_FORWARD),
        ]

        for check_func, gesture in gesture_checks:
            if check_func(hand_landmarks):
                return gesture

        return Gesture.STOP
 
        
    def calculate_angle(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        return math.degrees(math.atan2(delta_y, delta_x))
            

    def is_rotate_360(self, hand_landmarks: NormalizedLandmarkList) -> bool:
        if self.rotate_360_cooldown > 0:
            return False

        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        h, w = 480, 640
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        self.rotate_360_position_history.append((wrist_x, wrist_y))

        if len(self.rotate_360_position_history) < 10:
            return False
        
        angles = []
        points = list(self.rotate_360_position_history)
        for i in range(1, len(points)):
            angle = self.calculate_angle(points[i-1], points[i])
            angles.append(angle)

        total_rotation = 0.0
        for i in range(1, len(angles)):
            delta_angle = angles[i] - angles[i-1]
        
            if delta_angle > 180:
                delta_angle -= 360
            elif delta_angle < -180:
                delta_angle += 360
            total_rotation += delta_angle

        if self.rotate_360_direction is None and total_rotation != 0:
            self.rotate_360_direction = 1 if total_rotation > 0 else -1

        if self.rotate_360_direction is not None:
            if (total_rotation * self.rotate_360_direction) > 0:
                self.rotate_360_rotation_count += abs(total_rotation)
                if self.rotate_360_rotation_count > self.rotate_360_rotation_threshold:
                    self.rotate_360_position_history.clear()
                    self.rotate_360_rotation_count = 0
                    self.rotate_360_direction = None
                    self.rotate_360_cooldown = 30
                    return True
            else:
                self.rotate_360_rotation_count = 0
                self.rotate_360_direction = None

        return False

    def is_tilt_left(self, hand_landmarks: NormalizedLandmarkList) -> bool:
        return False

    def is_tilt_right(self, hand_landmarks: NormalizedLandmarkList) -> bool:
        return False 

    def is_attention(self, hand_landmarks: NormalizedLandmarkList) -> bool:
        return False

    def is_dance(self, hand_landmarks: NormalizedLandmarkList) -> bool:
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

        is_index_pinky_up = (index_finger_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y and
                             pinky_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].y)
        is_middle_ring_down = (middle_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and
                               ring_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP].y)
        is_thumb_down = thumb_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y

        return is_index_pinky_up and is_middle_ring_down and is_thumb_down

    def is_listen_message(self, hand_landmarks: NormalizedLandmarkList) -> bool:
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]

        is_index_up = index_finger_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y
        is_middle_down = middle_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
        is_ring_down = ring_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP].y
        is_pinky_down = pinky_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].y
        is_thumb_down = thumb_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y

        return is_index_up and is_middle_down and is_ring_down and is_pinky_down and is_thumb_down

    def is_pinch(self, hand_landmarks: NormalizedLandmarkList) -> bool:
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        distance_threshold = 0.05

        distance = self.calculate_distance(thumb_tip, index_tip)
        return distance < distance_threshold

    def is_fist(self, hand_landmarks: NormalizedLandmarkList) -> bool:
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        distance_threshold = 0.1

        distances = [
            self.calculate_distance(wrist, thumb_tip),
            self.calculate_distance(wrist, index_tip),
            self.calculate_distance(wrist, middle_tip),
            self.calculate_distance(wrist, ring_tip),
            self.calculate_distance(wrist, pinky_tip)
        ]

        return all(distance < distance_threshold for distance in distances)

    def is_fingers_forward(self, hand_landmarks: NormalizedLandmarkList) -> bool:
        dip_mapping = {
            mp.solutions.hands.HandLandmark.THUMB_TIP: mp.solutions.hands.HandLandmark.THUMB_IP,
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP: mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP,
            mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP: mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP,
            mp.solutions.hands.HandLandmark.RING_FINGER_TIP: mp.solutions.hands.HandLandmark.RING_FINGER_DIP,
            mp.solutions.hands.HandLandmark.PINKY_TIP: mp.solutions.hands.HandLandmark.PINKY_DIP,
        }
        for finger in [self.mp_hands.HandLandmark.THUMB_TIP,
                      self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                      self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                      self.mp_hands.HandLandmark.RING_FINGER_TIP,
                      self.mp_hands.HandLandmark.PINKY_TIP]:
            tip = hand_landmarks.landmark[finger]
            dip = dip_mapping[finger]  
            if tip.y > hand_landmarks.landmark[dip].y:
                return False
        return True
    
    def is_stop_hand(self, hand_landmarks: NormalizedLandmarkList) -> bool:
        dip_mapping = {
            mp.solutions.hands.HandLandmark.THUMB_TIP: mp.solutions.hands.HandLandmark.THUMB_IP,
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP: mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP,
            mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP: mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP,
            mp.solutions.hands.HandLandmark.RING_FINGER_TIP: mp.solutions.hands.HandLandmark.RING_FINGER_DIP,
            mp.solutions.hands.HandLandmark.PINKY_TIP: mp.solutions.hands.HandLandmark.PINKY_DIP,
        }
        for finger in [self.mp_hands.HandLandmark.THUMB_TIP,
                      self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                      self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                      self.mp_hands.HandLandmark.RING_FINGER_TIP,
                      self.mp_hands.HandLandmark.PINKY_TIP]:
            tip = hand_landmarks.landmark[finger]
            dip = dip_mapping[finger]  
            if tip.y > hand_landmarks.landmark[dip].y:
                return False
        return True

    def calculate_distance(self, point1, point2) -> float:
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5
