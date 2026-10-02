#!/usr/bin/env python3
"""
Standalone D435i head-camera snapshot.

Bypasses robot-server/ZMQ entirely -- opens the RealSense pipeline directly
and saves one color frame to disk. Only safe to run while nothing else
(start_server.py, etc.) is using the camera, since RealSense devices can't
be opened by two processes at once.

Usage:
    python3 capture_head_photo.py [output_path]
"""
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

COLOR_SIZE = [640, 480]
FPS = 30
# Settle frames to let auto-exposure/white-balance converge before keeping one.
WARMUP_FRAMES = 30


def find_d435i_serial():
    ctx = rs.context()
    for dev in ctx.devices:
        if dev.get_info(rs.camera_info.name) == "Intel RealSense D435I":
            return dev.get_info(rs.camera_info.serial_number)
    return None


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "head_camera_snapshot.jpg"

    serial = find_d435i_serial()
    if serial is None:
        raise SystemExit("No Intel RealSense D435I found -- is it plugged in?")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, COLOR_SIZE[0], COLOR_SIZE[1], rs.format.bgr8, FPS)
    pipeline.start(config)

    try:
        for _ in range(WARMUP_FRAMES):
            frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())
        # Match the orientation robot-server normally publishes the head
        # camera in (see camera/d435i_publisher.py's get_head_image_and_depth).
        image = np.rot90(image, k=-1)
    finally:
        pipeline.stop()

    cv2.imwrite(output_path, image)
    print(f"Saved {output_path}  (shape={image.shape})")


if __name__ == "__main__":
    main()
