#!/usr/bin/env python3
"""
Reboot and home a single end-of-arm joint (e.g. the tool/gripper motor) without
touching any other robot motor.

Unlike stretch_robot_dynamixel_reboot.py / stretch_robot_home.py, this does not
instantiate a full stretch_body.robot.Robot() (which talks to the base, lift,
arm, head, and pimu). It only opens the end-of-arm Dynamixel chain, and within
that chain only pings/reboots/homes the one joint requested.

Usage:
    python reboot_home_joint.py stretch_gripper
    python reboot_home_joint.py wrist_yaw
"""
import argparse
import importlib
import sys
import time

from stretch_body.robot_params import RobotParams


def build_end_of_arm():
    """
    Instantiate the robot's actual configured end-of-arm tool (e.g.
    eoa_wrist_dw3_tool_sg3), the same way stretch_body.robot.Robot does
    internally (robot.py: self.eoa_name = self.params['tool']), but without
    constructing the rest of the robot (base/lift/arm/head/pimu/wacc).
    """
    _, robot_params = RobotParams.get_params()
    eoa_name = robot_params["robot"]["tool"]
    module_name = robot_params[eoa_name]["py_module_name"]
    class_name = robot_params[eoa_name]["py_class_name"]
    end_of_arm_cls = getattr(importlib.import_module(module_name), class_name)
    return end_of_arm_cls(name=eoa_name)


def main():
    parser = argparse.ArgumentParser(
        description="Reboot and home a single end-of-arm joint without disturbing other robot motors."
    )
    parser.add_argument(
        "joint",
        nargs="?",
        default="stretch_gripper",
        help="Name of the end-of-arm joint to reboot/home (default: stretch_gripper). "
             "Other options depend on the attached tool, e.g. wrist_yaw, wrist_pitch, wrist_roll.",
    )
    args = parser.parse_args()

    end_of_arm = build_end_of_arm()

    # A servo that's latched a hardware error (e.g. overload_error) makes the
    # whole chain's startup() raise/return False, even though the serial port
    # itself opened fine and other joints may be OK. That's expected here, so
    # we don't treat it as fatal -- we still have direct port access to ping
    # and reboot the one faulted joint below.
    end_of_arm.startup(threaded=False)

    if args.joint not in end_of_arm.joints:
        print(f"Unknown joint '{args.joint}'. Available end-of-arm joints: {list(end_of_arm.joints)}")
        end_of_arm.stop()
        sys.exit(1)

    motor = end_of_arm.get_motor(args.joint)

    if motor.motor.do_ping(verbose=False):
        print(f"Rebooting: {args.joint}")
        motor.motor.do_reboot()
        time.sleep(2.0)  # let the servo finish its power-on reset
    else:
        print(f"Warning: {args.joint} did not respond to ping, skipping reboot.")
        end_of_arm.stop()
        sys.exit(1)

    # Re-run startup now that the error should be cleared, so hw_valid gets
    # set True this time -- home() below refuses to run otherwise.
    if not end_of_arm.startup(threaded=False):
        print(f"Still unable to fully initialize the end-of-arm chain after rebooting '{args.joint}'. "
              "Not attempting to home; check for a persistent hardware fault.")
        end_of_arm.stop()
        sys.exit(1)

    # Not end_of_arm.home(joint=...): some tools (e.g. EOA_Wrist_DW3_Tool_SG3)
    # override EndOfArm.home() with a no-arg version that homes/stows every
    # joint on the tool together. Calling the joint's own .home() directly
    # bypasses that and only touches this one motor.
    print(f"Homing: {args.joint}")
    motor.home()

    end_of_arm.stop()
    print(f"Done. '{args.joint}' rebooted and homed; no other motors were touched.")


if __name__ == "__main__":
    main()
