from .gestures import Gesture
from typing import Callable, Dict
import logging

class CommandHandler:
	def __init__(self) -> None:
		self.command_map: Dict[Gesture, Callable[[], None]] = {
				Gesture.ROTATE_360: self.rotate_360,
				Gesture.TILT_LEFT: self.tilt_left,
				Gesture.TILT_RIGHT: self.tilt_right,
				Gesture.ATTENTION: self.attention,
				Gesture.DANCE: self.dance,
				Gesture.LISTEN_MESSAGE: self.listen_message,
				Gesture.STOP: self.stop,
				Gesture.PINCH: self.pinch,
				Gesture.FIST: self.fist,
				Gesture.FINGERS_FORWARD: self.fingers_forward
		}

	def execute_command(self, Gesture: Gesture) -> None:
		command: Callable[[], None] = self.command_map.get(Gesture, self.unknown_command)
		command()

	def rotate_360(self) -> None:
		logging.info("Rotate 360 degrees")

	def tilt_left(self) -> None:
		logging.info("Tilt left")

	def tilt_right(self) -> None:
		logging.info("Tilt right")

	def attention(self) -> None:
		logging.info("Attention")

	def dance(self) -> None:
		logging.info("Dance")

	def listen_message(self) -> None:
		logging.info("Listen message")

	def stop(self) -> None:
		logging.info("Stop")

	def pinch(self) -> None:
		logging.info("Pinch")

	def fist(self) -> None:
		logging.info("Fist")

	def fingers_forward(self) -> None:
		logging.info("Fingers forward")

	def unknown_command(self) -> None:
		logging.info("Unknown command")
