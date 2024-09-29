import cv2
import time
from typing import List, Tuple
from .gesture_recognition import GestureRecognition
from .command_handler import CommandHandler
from .gestures import Gesture
from . import config
import logging

class Application:
    def __init__(self) -> None:
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
         self.logger: logging.Logger = logging.getLogger(__name__)
         self.gesture_recognition: GestureRecognition = GestureRecognition()
         self.command_handler: CommandHandler = CommandHandler()
         self.cap: cv2.VideoCapture = self.initialize_camera(config.CAMERA_INDEX)
         self.p_time: float = 0.0

    def initialize_camera(self, camera_index: int) -> cv2.VideoCapture:
        cap: cv2.VideoCapture = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"[*] Unable to open camera with index {camera_index}")
        return cap

    def capture_frame(self) -> Tuple[bool, cv2.Mat]:
        return self.cap.read()

    def process_gestures(self, img: cv2.Mat) -> List[str]:
        gestures, processed_img = self.gesture_recognition.process_frame(img)
        return gestures

    def execute_commands(self, gestures: List[str]) -> None:
         for gesture in gestures:
               if gesture != Gesture.STOP:
                self.command_handler.execute_command(gesture)
                self.logger.info(f"Executed command: {gesture.value}")

    def calculate_fps(self) -> float:
        c_time: float = time.time()
        fps: float = 1 / (c_time - self.p_time) if (c_time - self.p_time) > 0 else 0.0
        self.p_time = c_time
        return fps

    def display_fps(self, img: cv2.Mat, fps: float) -> None:
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

    def display_frame(self, img: cv2.Mat) -> None:
        cv2.imshow("Gesture Recognition", img)

    def handle_exit(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        try:
            while True:
                success: bool
                img: cv2.Mat
                success, img = self.capture_frame()
                if not success:
                    logging.error("[*] Failed to get a frame from the camera")
                    break

                gestures: List[Gesture] = self.process_gestures(img)
                self.execute_commands(gestures)
                fps: float = self.calculate_fps()
                self.display_fps(img, fps)
                self.display_frame(img)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        except KeyboardInterrupt:
            logging.info("[!] The programme was interrupted by the user.")
        finally:
            self.handle_exit()
