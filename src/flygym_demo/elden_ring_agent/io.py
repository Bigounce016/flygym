"""Optional screen-capture and gamepad adapters for the Elden Ring demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .policy import Action, ActionCommand


class FrameSource(Protocol):
    def capture(self) -> np.ndarray:
        """Return the current game frame as an RGB array."""


class Controller(Protocol):
    def send(self, command: ActionCommand) -> None:
        """Apply one high-level action to the game controller."""


@dataclass(slots=True)
class ScreenCapturer:
    """Capture a desktop region using Pillow's native Windows capture."""

    region: tuple[int, int, int, int] | None = None

    def capture(self) -> np.ndarray:
        from PIL import ImageGrab

        image = ImageGrab.grab(bbox=self.region, all_screens=True)
        return np.asarray(image.convert("RGB"))


@dataclass(frozen=True, slots=True)
class GamepadState:
    """Normalized state sent to a virtual Xbox controller."""

    left_x: float = 0.0
    left_y: float = 0.0
    attack: bool = False
    dodge: bool = False
    heal: bool = False


def command_to_gamepad_state(command: ActionCommand) -> GamepadState:
    """Map the policy command to left-stick movement and face buttons."""
    if command.action == Action.ADVANCE:
        return GamepadState(left_y=1.0, left_x=command.turn)
    if command.action == Action.RETREAT:
        return GamepadState(left_y=-1.0, left_x=command.turn)
    if command.action == Action.TURN_LEFT:
        return GamepadState(left_x=-max(abs(command.turn), 0.35))
    if command.action == Action.TURN_RIGHT:
        return GamepadState(left_x=max(abs(command.turn), 0.35))
    if command.action == Action.ATTACK:
        return GamepadState(attack=True)
    if command.action == Action.DODGE:
        return GamepadState(dodge=True)
    if command.action == Action.HEAL:
        return GamepadState(heal=True)
    return GamepadState()


class VirtualXboxController:
    """Send commands through the optional ``vgamepad`` Windows package."""

    def __init__(self) -> None:
        try:
            import vgamepad as vg
        except ImportError as error:
            raise RuntimeError(
                "VirtualXboxController requires the optional 'vgamepad' package "
                "and its ViGEmBus driver."
            ) from error

        self._vg = vg
        self._gamepad = vg.VX360Gamepad()

    def send(self, command: ActionCommand) -> None:
        state = command_to_gamepad_state(command)
        gamepad = self._gamepad
        gamepad.left_joystick_float(
            x_value_float=state.left_x,
            y_value_float=state.left_y,
        )
        for button in (
            self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
            self._vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
            self._vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
        ):
            gamepad.release_button(button)
        if state.attack:
            gamepad.press_button(self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        elif state.dodge:
            gamepad.press_button(self._vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        elif state.heal:
            gamepad.press_button(self._vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        gamepad.update()