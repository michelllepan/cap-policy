#!/usr/bin/env python3
"""
Standalone wide-angle head-mounted RGB camera snapshot.

For the separate wide-FOV webcam on the head (not the D435i or D405
RealSense cameras -- see capture_head_photo.py for the D435i). This is a
plain UVC camera, accessed directly via OpenCV/V4L2 -- bypasses
robot-server entirely. Only safe to run while nothing else is using the
camera device.

Usage:
    python3 capture_wide_angle_photo.py --list             # find the right device first
    python3 capture_wide_angle_photo.py --device /dev/video2 [output_path]
"""
import argparse
import glob

import cv2

WARMUP_FRAMES = 30


def list_devices():
    devices = sorted(glob.glob("/dev/video*"))
    if not devices:
        print("No /dev/video* devices found.")
        return
    print("Available video devices:")
    for dev in devices:
        cap = cv2.VideoCapture(dev)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  {dev}  (opens ok, default {width}x{height})")
        else:
            print(f"  {dev}  (failed to open -- may be the depth/IR sibling node of a RealSense camera)")
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="/dev/video0", help="camera device path or index (default: /dev/video0)")
    parser.add_argument("--list", action="store_true", help="list available /dev/video* devices and exit")
    parser.add_argument("output", nargs="?", default="wide_angle_snapshot.jpg")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    device = int(args.device) if args.device.isdigit() else args.device

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera device {args.device!r}. Try --list to see available devices.")

    try:
        frame = None
        for _ in range(WARMUP_FRAMES):
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"Failed to read a frame from {args.device!r}.")
    finally:
        cap.release()

    # Camera is mounted rotated 90 degrees relative to upright.
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imwrite(args.output, frame)
    print(f"Saved {args.output}  (shape={frame.shape})")


if __name__ == "__main__":
    main()
