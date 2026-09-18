"""Live FlyGym motion loop that follows the Elden Ring-style policy contract."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

import mujoco
import numpy as np

from flygym.anatomy import (
    ActuatedDOFPreset,
    AxisOrder,
    ContactBodiesPreset,
    JointPreset,
    Skeleton,
)
from flygym.compose import ActuatorType, FlatGroundWorld, Fly, KinematicPosePreset
from flygym.rendering import launch_interactive_viewer
from flygym.utils.math import Rotation3D
from flygym_demo.elden_ring_agent.io import Controller, FrameSource
from flygym_demo.elden_ring_agent.policy import Action, ActionCommand, FlyGameObservation, RuleBasedFlyPolicy


def action_to_position_targets(
    dofs: Iterable[str],
    neutral_pose: dict[str, float],
    command: ActionCommand,
    *,
    phase: float,
    base_amplitude: float = 0.35,
) -> np.ndarray:
    """Translate a high-level Elden Ring action into leg-target angles."""
    dofs = list(dofs)
    targets = np.array([float(neutral_pose.get(dof, 0.0)) for dof in dofs], dtype=float)
    phase_offsets = np.linspace(0.0, 2.0 * np.pi, len(dofs), endpoint=False)

    if command.action == Action.ATTACK:
        targets = targets + base_amplitude * (1.0 + 0.7 * np.sin(phase + phase_offsets))
    elif command.action == Action.DODGE:
        targets = targets + 0.5 * (1.0 + 0.6 * np.cos(2.0 * phase + phase_offsets))
    elif command.action in (Action.TURN_LEFT, Action.TURN_RIGHT):
        turn_gain = 0.8 if command.turn >= 0 else -0.8
        targets = targets + turn_gain * np.sin(phase + phase_offsets)
    elif command.action == Action.RETREAT:
        targets = targets - base_amplitude * np.sin(phase + phase_offsets)
    elif command.action == Action.HEAL:
        targets = targets + 0.15 * np.cos(phase + phase_offsets)
    else:
        targets = targets + 0.12 * np.sin(phase + phase_offsets)

    return targets


def build_elden_ring_fly():
    """Create a maneuverable standing fly model with active leg DOFs."""
    neutral_pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
        AxisOrder.YAW_PITCH_ROLL
    )
    skeleton = Skeleton(
        joint_preset=JointPreset.ALL_BIOLOGICAL,
        axis_order=AxisOrder.YAW_PITCH_ROLL,
    )
    fly = Fly()
    fly.add_joints(skeleton, neutral_pose)

    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs,
        ActuatorType.POSITION,
        neutral_input=neutral_pose,
        kp=50.0,
        ctrlrange=(-3.14, 3.14),
    )
    fly.colorize()
    dofs = [jointdof.name for jointdof in fly.get_actuated_jointdofs_order(ActuatorType.POSITION)]
    return fly, neutral_pose, dofs


def run_live_elden_ring_demo(
    *,
    render: bool = True,
    sleep_time: float = 0.02,
    frame_source: FrameSource | None = None,
    observation_provider: Callable[[np.ndarray], FlyGameObservation] | None = None,
    controller: Controller | None = None,
    max_steps: int | None = None,
) -> None:
    """Run the fly agent against synthetic or captured Elden Ring observations.

    Supplying ``frame_source`` and ``observation_provider`` enables screen-driven
    control. Supplying ``controller`` additionally sends each policy command to
    a gamepad. With none of these arguments, the deterministic demo loop runs.
    """
    if (frame_source is None) != (observation_provider is None):
        raise ValueError("frame_source and observation_provider must be supplied together")
    fly, neutral_pose, dofs = build_elden_ring_fly()
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        (0.0, 0.0, 0.8),
        Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
        bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
    )
    mj_model, mj_data = world.compile()

    if render:
        launch_interactive_viewer(mj_model, mj_data, run_async=True)

    policy = RuleBasedFlyPolicy()
    phase = 0.0
    observation = FlyGameObservation(
        enemy_visible=True,
        enemy_distance=0.18,
        enemy_direction=-0.6,
        enemy_attacking=True,
        health=0.9,
        stamina=0.8,
        target_direction=0.3,
    )
    print("Fly is now running an Elden Ring-style action loop in the MuJoCo viewer. Close the window to quit.")

    step = 0
    try:
        while max_steps is None or step < max_steps:
            if frame_source is not None and observation_provider is not None:
                observation = observation_provider(frame_source.capture())
            command = policy.decide(observation)
            if controller is not None:
                controller.send(command)
            targets = action_to_position_targets(dofs, neutral_pose, command, phase=phase)
            mj_data.ctrl[:] = targets
            mujoco.mj_step(mj_model, mj_data)

            # Simple pseudo-game loop: alternate attack/dodge states so the fly visibly responds.
            observation = FlyGameObservation(
                enemy_visible=True,
                enemy_distance=0.15 + 0.35 * np.sin(phase / 3.0),
                enemy_direction=np.sin(phase * 0.75),
                enemy_attacking=(phase % 4.0) > 2.0,
                health=max(0.2, 0.75 + 0.15 * np.sin(phase * 0.5)),
                stamina=max(0.1, 0.65 + 0.2 * np.cos(phase * 0.7)),
                target_direction=np.sin(phase * 0.9),
            )
            phase += 0.25
            step += 1
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Closing live Elden Ring fly demo.")


def main() -> None:
    run_live_elden_ring_demo()


if __name__ == "__main__":
    main()
