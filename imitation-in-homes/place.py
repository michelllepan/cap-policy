"""
Autonomous place: given a target location description, uses Gemini
Robotics-ER to visually servo the robot until the object currently held in
the gripper is positioned over the target, then releases it.

Unlike pick.py, this doesn't run any learned policy -- there's no trained
"place" model, so this is pure VLM-guided primitive control. Each iteration,
Gemini ER (given the wide-angle and wrist camera views) reports either that
the gripper is aligned over the target (release), or one correction to make:
  - A base shift: Gemini ER reports the target location's horizontal pixel
    position in the wide-angle view (same grounded approach as pick.py's
    search), and we derive the move_base direction/distance ourselves using
    the same hardware-confirmed mapping pick.py's search uses, rather than
    trusting the model's own sense of screen direction for this axis.
  - A lift/arm adjustment: Gemini ER picks the action directly (lift_up/
    lift_down/arm_extend/arm_retract), since those axes don't have a
    confusing sign convention the way base movement does.
Either way, Gemini ER also reports how many small steps to take at once.

Assumes:
  - The robot is already holding the object (e.g. after pick.py succeeds).
  - start_server.py is already running, with the fisheye camera feed
    configured (frames are pulled from its web viewer over HTTP, same as
    pick.py, rather than opening the camera device directly).

While running, both the fisheye and wrist cameras are also recorded to MP4s
(videos/ alongside this script by default) for later review, independent of
the sparser frames actually sent to Gemini ER. The wrist recording uses its
own dedicated ZMQ subscriber, separate from the one the main loop reads
from, since ZMQ SUB sockets aren't safe for concurrent use across threads.

Usage:
    python place.py --location "the empty plate"
    python place.py --location "the empty plate" \\
        --robot-host 127.0.0.1 --robot-port 8081 --camera-port 32922 \\
        --wide-camera-url http://127.0.0.1:7860/feed/fisheye/snapshot \\
        --video-path /tmp/my_place.mp4
"""
import argparse
import json
import logging
import os
import re
import sys
import threading
import time

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from utils.rpc import RPCClient
from utils.zmq_utils import ZMQCameraSubscriber

# Load before any other setup (same reasoning as run.py/pick.py).
load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

BASE_STEP = 0.02  # meters per base step
LIFT_STEP = 0.02  # meters per lift step
ARM_STEP = 0.02  # meters per arm step
MAX_STEPS_PER_MOVE = 5  # cap on how many STEPs Gemini ER can request per iteration
MAX_PLACE_STEPS = 15  # give up after this many VLM-queried iterations
SETTLE_TIME = 1.0  # seconds to let lift/arm motion finish before the next frame
RECORD_FPS = 5

WIDE_CAMERA_URL_DEFAULT = "http://127.0.0.1:7860/feed/fisheye/snapshot"
VIDEO_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")

VALID_ACTIONS = {
    "lift_up",
    "lift_down",
    "arm_extend",
    "arm_retract",
}


def _get_vlm_client():
    from google import genai

    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _encode_png(image):
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode image for VLM query")
    return buffer.tobytes()


class WideAngleCamera:
    """
    Pulls single-frame JPEG snapshots from robot-server's camera web viewer
    (camera/web_viewer.py's /feed/<name>/snapshot route) over HTTP, instead
    of opening the wide-angle camera's V4L2 device directly -- so this can
    run at the same time as start_server.py without both processes fighting
    over exclusive access to the same UVC camera. The viewer already applies
    the 90-degree rotation needed to correct for the camera's mounting, so no
    rotation is needed here.
    """

    def __init__(self, snapshot_url, timeout=5.0):
        self.snapshot_url = snapshot_url
        self.timeout = timeout

    def read(self):
        response = requests.get(self.snapshot_url, timeout=self.timeout)
        response.raise_for_status()
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to decode snapshot from {self.snapshot_url}")
        return image

    def close(self):
        pass


class VideoRecorder:
    """
    Continuously calls read_fn() in a background thread and writes frames to
    an MP4, independent of the main loop's own much sparser reads (one every
    several seconds, gated on VLM round-trips) -- so the recording covers the
    whole run smoothly instead of only the handful of frames actually sent to
    Gemini ER.

    read_fn should be backed by a camera/subscriber object not shared with
    the main thread: ZMQ SUB sockets (the wrist camera) aren't safe for
    concurrent use from multiple threads, so the wrist recording must use its
    own dedicated subscriber rather than the one the main logic reads from.
    """

    def __init__(self, read_fn, output_path, fps=RECORD_FPS, label="video"):
        self.read_fn = read_fn
        self.output_path = output_path
        self.fps = fps
        self.label = label
        self._writer = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        logger.info(f"Recording {self.label} to {self.output_path} ({self.fps} fps)...")
        self._thread.start()

    def _run(self):
        interval = 1.0 / self.fps
        while not self._stop_event.is_set():
            start = time.time()
            try:
                frame = self.read_fn()
            except Exception as e:
                logger.warning(f"Video recorder ({self.label}): failed to read a frame: {e}")
                time.sleep(interval)
                continue
            if self._writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            self._writer.write(frame)
            time.sleep(max(0, interval - (time.time() - start)))

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        if self._writer is not None:
            self._writer.release()
            logger.info(f"Saved {self.label} to {self.output_path}")


def _locate_for_place(vlm_client, wide_image, wrist_image, place_location):
    """
    Ask Gemini ER whether the object currently held in the gripper (visible
    in the wrist image) is correctly positioned over the target placement
    location. If not, get back either:
      - a base correction: target_x, the target location's horizontal pixel
        position in the wide-angle image (0-1000) -- the same grounded,
        pixel-position-driven approach as pick.py's search, since we've
        already empirically confirmed the sign mapping from that position to
        a move_base direction. This avoids relying on the model's own
        (unverified) sense of which screen direction "forward"/"backward"
        correspond to.
      - or a lift/arm correction: a direct action choice, since those axes
        don't have a confusing sign convention the way base movement does.
    The prompt also instructs Gemini ER to lift the object clear of its
    pickup surface before making any other move (pick.py no longer verifies
    this itself), and to avoid choosing corrections that would drag the
    object or gripper into other objects visible in the wide-angle view.
    """
    from google.genai import types

    prompt = (
        "You are controlling a mobile robot that is holding an object in "
        "its closed gripper (visible in image 2). It needs to place that "
        f"object at the location described as: '{place_location}'.\n"
        "Image 1 is a wide-angle view of the scene. Image 2 is a close-up "
        "view from a camera mounted at the robot's wrist, right next to the "
        "gripper and held object.\n\n"
        "Safety first: if the held object is still resting on, or very "
        "close to, the surface it was picked up from, do NOT move the base "
        "or arm yet -- first report \"needs_base_correction\": false, "
        "\"action\": \"lift_up\" to raise it clear of that surface.\n\n"
        "Also, before choosing any correction, check image 1 for other "
        "objects near the gripper's path. Never choose a base shift, lift, "
        "or arm move that would drag the held object into, or swing the "
        "gripper into, another object in the scene -- prefer a smaller step, "
        "a different axis, or lifting higher first, to clear any obstacle.\n\n"
        "Determine whether the gripper (and held object) is directly above "
        f"'{place_location}', close enough to release the object there.\n\n"
        "If not yet aligned, decide whether the correction needed is a "
        "sideways base shift, or a lift/arm adjustment:\n\n"
        '- If it\'s a base shift: report "needs_base_correction": true, and '
        f'"target_x": the horizontal pixel position of \'{place_location}\' '
        "in image 1, normalized 0-1000 (0 = left edge, 1000 = right edge).\n"
        '- If it\'s a lift/arm adjustment: report "needs_base_correction": '
        'false, and "action" as one of:\n'
        '  - "lift_up" / "lift_down": raise or lower the gripper.\n'
        '  - "arm_extend" / "arm_retract": move the gripper further from or '
        "closer to the robot's body.\n\n"
        'Either way, report "steps": how many small steps of that '
        f"correction to take at once, as an integer from 1 to "
        f"{MAX_STEPS_PER_MOVE} -- more steps for larger misalignment, fewer "
        "for small (and when avoiding an obstacle, prefer fewer steps).\n\n"
        'Respond with only JSON: {"aligned": bool, "needs_base_correction": bool, '
        '"target_x": int, "action": "lift_up"|"lift_down"|"arm_extend"|"arm_retract"'
        '|null, "steps": int}'
    )

    logger.info("Querying Gemini ER for placement (wide + wrist views)...")
    response = vlm_client.models.generate_content(
        model="gemini-robotics-er-2-preview",
        contents=[
            types.Part.from_bytes(data=_encode_png(wide_image), mime_type="image/png"),
            types.Part.from_bytes(data=_encode_png(wrist_image), mime_type="image/png"),
            prompt,
        ],
    )
    logger.info(f"Gemini ER response: {response.text.strip()}")

    match = re.search(r"\{.*\}", response.text, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not parse VLM response: {response.text}")
    result = json.loads(match.group(0))

    aligned = bool(result["aligned"])
    needs_base_correction = bool(result.get("needs_base_correction"))
    target_x = int(result.get("target_x") or 500)
    action = result.get("action")
    steps = max(1, min(MAX_STEPS_PER_MOVE, int(result.get("steps") or 1)))
    return aligned, needs_base_correction, target_x, action, steps


def _apply_base_correction(robot, target_x, steps):
    """
    Same direction mapping as pick.py's search, confirmed empirically on
    hardware: target right of center -> move backward to bring it toward
    center; left of center -> move forward.
    """
    distance = BASE_STEP * steps
    if target_x > 500:
        logger.info(f"Target right of center (target_x={target_x}) -- moving backward {distance:.3f}m.")
        robot.move_base(-distance)
    else:
        logger.info(f"Target left of center (target_x={target_x}) -- moving forward {distance:.3f}m.")
        robot.move_base(distance)


def _apply_joint_action(robot, action, steps):
    """
    Execute a lift/arm corrective action: read the current joint position and
    command a new absolute target with everything else (wrist orientation,
    gripper) held fixed -- gripper is explicitly kept at 0.0 (closed) since
    the object must stay held while repositioning.
    """
    if action not in VALID_ACTIONS:
        logger.warning(f"Unknown action {action!r} from VLM -- skipping this iteration.")
        return

    lift_pos, _base_pos, arm_pos, roll_pos, pitch_pos, yaw_pos, _gripper_pos = robot.getJointPos()
    if action == "lift_up":
        lift_pos += LIFT_STEP * steps
    elif action == "lift_down":
        lift_pos -= LIFT_STEP * steps
    elif action == "arm_extend":
        arm_pos += ARM_STEP * steps
    elif action == "arm_retract":
        arm_pos -= ARM_STEP * steps

    logger.info(f"{action} x{steps}: lift_pos={lift_pos:.3f} arm_pos={arm_pos:.3f}")
    robot.move_to_position(
        lift_pos=lift_pos,
        arm_pos=arm_pos,
        base_trans=0.0,
        wrist_yaw=yaw_pos,
        wrist_pitch=pitch_pos,
        wrist_roll=roll_pos,
        gripper_pos=0.0,  # stay closed -- object must remain held while repositioning
    )
    time.sleep(SETTLE_TIME)


def place_object(robot, wrist_camera, wide_camera, vlm_client, place_location):
    logger.info(f"Starting place at '{place_location}' (max {MAX_PLACE_STEPS} iterations)...")
    for attempt in range(MAX_PLACE_STEPS):
        logger.info(f"--- Place attempt {attempt + 1}/{MAX_PLACE_STEPS} ---")
        wrist_image, _np_depth, _timestamp = wrist_camera.recv_image_and_depth()
        wide_image = wide_camera.read()

        aligned, needs_base_correction, target_x, action, steps = _locate_for_place(
            vlm_client, wide_image, wrist_image, place_location
        )
        logger.info(
            f"aligned={aligned} needs_base_correction={needs_base_correction} "
            f"target_x={target_x} action={action} steps={steps}"
        )

        if aligned:
            logger.info("Aligned over target -- releasing gripper.")
            robot.move_to_pose(np.zeros(3), np.zeros(3), 1.0)
            return True

        if needs_base_correction:
            _apply_base_correction(robot, target_x, steps)
        else:
            _apply_joint_action(robot, action, steps)

    logger.warning(f"Gave up placing after {MAX_PLACE_STEPS} iterations without confirming alignment.")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", required=True, help="description of where to place it, e.g. 'the empty plate'")
    parser.add_argument("--robot-host", default="127.0.0.1", help="robot-server host/IP")
    parser.add_argument("--robot-port", type=int, default=8081, help="robot-server RPC action port")
    parser.add_argument("--camera-port", type=int, default=32922, help="wrist camera ZMQ port")
    parser.add_argument("--wide-camera-url", default=WIDE_CAMERA_URL_DEFAULT, help="camera web viewer's fisheye snapshot URL")
    parser.add_argument("--video-path", default=None, help="output path for the fisheye recording (default: videos/place_<timestamp>_fisheye.mp4 next to this script)")
    parser.add_argument("--wrist-video-path", default=None, help="output path for the wrist recording (default: videos/place_<timestamp>_wrist.mp4 next to this script)")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = args.video_path or os.path.join(VIDEO_DIR_DEFAULT, f"place_{timestamp}_fisheye.mp4")
    wrist_video_path = args.wrist_video_path or os.path.join(VIDEO_DIR_DEFAULT, f"place_{timestamp}_wrist.mp4")

    robot = RPCClient(args.robot_host, args.robot_port)
    wrist_camera = ZMQCameraSubscriber(args.robot_host, args.camera_port, "RGBD")
    wide_camera = WideAngleCamera(args.wide_camera_url)
    vlm_client = _get_vlm_client()
    fisheye_recorder = VideoRecorder(wide_camera.read, video_path, label="fisheye video")

    # Dedicated subscriber for recording -- never shared with the wrist_camera
    # used by the main loop, since ZMQ SUB sockets aren't safe for concurrent
    # use across threads.
    wrist_record_subscriber = ZMQCameraSubscriber(args.robot_host, args.camera_port, "RGBD")
    wrist_recorder = VideoRecorder(
        lambda: wrist_record_subscriber.recv_image_and_depth()[0], wrist_video_path, label="wrist video"
    )

    try:
        fisheye_recorder.start()
        wrist_recorder.start()
        success = place_object(robot, wrist_camera, wide_camera, vlm_client, args.location)
        logger.info(f"Place {'succeeded' if success else 'failed'} at '{args.location}'.")
    finally:
        fisheye_recorder.stop()
        wrist_recorder.stop()
        wide_camera.close()


if __name__ == "__main__":
    main()
