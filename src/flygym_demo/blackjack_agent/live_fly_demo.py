"""A real FlyGym demo that maps blackjack actions onto fly joint targets."""

from __future__ import annotations

import random
import time
from typing import Iterable

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
from flygym_demo.blackjack_agent.blackjack import (
    HIT,
    STICK,
    BlackjackEnv,
    BlackjackQLearner,
    train,
)


def action_to_position_targets(
    dofs: Iterable[str],
    neutral_pose: dict[str, float],
    *,
    action: int,
    phase: float,
    hit_amplitude: float = 0.6,
    stick_amplitude: float = 0.18,
) -> np.ndarray:
    """Map a blackjack action to target joint angles for a fly leg controller.

    HIT drives the fly into a more dynamic, oscillatory pose; STICK settles into a
    calmer posture. The result is a vector with one target angle per actuated DOF,
    ordered exactly like ``dofs``.
    """
    if action not in (HIT, STICK):
        raise ValueError(f"action must be {HIT} or {STICK}, got {action!r}")

    dofs = list(dofs)
    targets = np.array([float(neutral_pose.get(dof, 0.0)) for dof in dofs], dtype=float)
    phase_offsets = np.linspace(0.0, 2.0 * np.pi, len(dofs), endpoint=False)

    if action == HIT:
        targets = targets + hit_amplitude * np.sin(phase + phase_offsets)
    else:
        targets = targets + stick_amplitude * np.cos(0.75 * phase + phase_offsets)

    return targets


def build_blackjack_fly(*, seed: int | None = None):
    """Create a simple standing fly with position-controlled leg joints."""
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


def run_live_blackjack_demo(
    *,
    episodes: int = 20000,
    seed: int | None = 0,
    render: bool = True,
    sleep_time: float = 0.02,
) -> None:
    """Train a blackjack policy and drive a real fly model with it while the viewer is open."""
    learner = train(episodes=episodes, seed=seed)
    fly, neutral_pose, dofs = build_blackjack_fly(seed=seed)

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

    env = BlackjackEnv(rng=random.Random(seed))
    state = env.reset()
    phase = 0.0
    print("Fly is now playing blackjack in the MuJoCo viewer; close the window to quit.")

    try:
        while True:
            action = learner.choose_action(state, explore=False)
            target_angles = action_to_position_targets(
                dofs,
                neutral_pose,
                action=action,
                phase=phase,
            )
            mj_data.ctrl[:] = target_angles
            mujoco.mj_step(mj_model, mj_data)

            next_state, _, done = env.step(action)
            if next_state is not None:
                state = next_state
            else:
                state = env.reset()

            phase += 0.2
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Closing live blackjack fly demo.")


def main() -> None:
    run_live_blackjack_demo()


if __name__ == "__main__":
    main()
