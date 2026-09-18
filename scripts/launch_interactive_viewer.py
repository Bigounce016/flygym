import time

import mujoco
import numpy as np

from flygym.anatomy import (
    Skeleton,
    JointPreset,
    AxisOrder,
    ActuatedDOFPreset,
    ContactBodiesPreset,
)
from flygym.compose import Fly, ActuatorType, FlatGroundWorld, KinematicPosePreset
from flygym.rendering import launch_interactive_viewer
from flygym.utils.math import Rotation3D

joint_preset = JointPreset.ALL_BIOLOGICAL
axis_order = AxisOrder.YAW_PITCH_ROLL
actuated_dofs = ActuatedDOFPreset.LEGS_ACTIVE_ONLY
actuator_type = ActuatorType.POSITION
actuator_position_gain = 50.0
neutral_pose = KinematicPosePreset.NEUTRAL
spawn_position = (0, 0, 0.8)  # xyz in mm
spawn_rotation = Rotation3D("quat", (1, 0, 0, 0))  # wxyz in quaternion
bodysegs_with_ground_contact = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD
run_async = True  # launch viewer in a separate process so we can keep animating


def main():
    fly = Fly()

    skeleton = Skeleton(joint_preset=joint_preset, axis_order=axis_order)
    fly.add_joints(skeleton, neutral_pose)

    actuated_dofs_list = skeleton.get_actuated_dofs_from_preset(actuated_dofs)
    fly.add_actuators(
        actuated_dofs_list,
        actuator_type,
        neutral_input=neutral_pose,
        kp=actuator_position_gain,
        ctrlrange=(-3.14, 3.14),
    )

    fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())

    fly.colorize()
    fly.add_tracking_camera(
        name="trackingcam",
        mode="track",
        pos_offset=(-0.5, -7.5, 5),
        rotation=Rotation3D("xyaxes", (1, 0, 0, 0, 0.6, 0.8)),
        fovy=30.0,
    )

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        spawn_position,
        spawn_rotation,
        bodysegs_with_ground_contact=bodysegs_with_ground_contact,
    )

    mj_model, mj_data = world.compile()

    # Gently animate the fly while the viewer stays open.
    amp = 0.6
    phase = 0.0
    print("Launching fly viewer... Use the camera controls to orbit/zoom; close the window to quit.")
    launch_interactive_viewer(mj_model, mj_data, run_async=run_async)

    try:
        while True:
            phase += 0.05
            mj_data.ctrl[:] = amp * np.sin(phase + np.linspace(0, 2 * np.pi, mj_model.nu, endpoint=False))
            mujoco.mj_step(mj_model, mj_data)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Viewer closed.")


if __name__ == "__main__":
    main()
