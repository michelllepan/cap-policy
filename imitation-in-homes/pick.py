"""
Autonomous pick: given an object name, uses Gemini Robotics-ER to search for
the object and then runs the existing VQ-BeT pick policy to grasp it --
no keyboard/mouse input required.

Equivalent to interactively running run.py (typing the object name, "p", and
repeated run commands), but fully autonomous:

  1. Search: repeatedly capture the head-mounted wide-angle camera (always
     sees the object, per the robot's physical layout) and the wrist camera
     (may not yet), and ask Gemini ER whether the object is visible in the
     wrist camera. If not, drive the base forward/backward based on the
     object's position in the wide camera until it is: object right of
     center -> move backward; left of center -> move forward (confirmed
     empirically on hardware).
  2. Grasp: once visible, run the normal use_vlm=true pick policy (Gemini ER
     picks the exact contact point, same as run.py's "p" + run flow) in
     batches of STEPS_PER_BATCH steps, stopping once the gripper closes
     (controller.gripper drops below the pick task's closing_threshold --
     the same signal _run_policy_goals already uses to end an episode).

While running, both the fisheye and wrist cameras are also recorded to MP4s
(videos/ alongside this script by default) for later review, independent of
the sparser frames actually sent to Gemini ER. The wrist recording uses its
own dedicated ZMQ subscriber, separate from the one the main logic reads
from, since ZMQ SUB sockets aren't safe for concurrent use across threads.

Usage:
    python pick.py object="red mug"
    python pick.py object="red mug" wide_camera_url=http://127.0.0.1:7860/feed/fisheye/snapshot
    python pick.py object="red mug" video_path=/tmp/my_pick_fisheye.mp4 wrist_video_path=/tmp/my_pick_wrist.mp4

object, wide_camera_url, video_path, and wrist_video_path aren't in
run_vqbet_pick.yaml, so on the command line they need Hydra's "+" syntax for
new keys:
    python pick.py +object="red mug"

The wide-angle/fisheye frame is pulled from robot-server's camera web viewer
(robot-server/camera/web_viewer.py) over HTTP, rather than opening the
camera's V4L2 device directly, since start_server.py's own viewer process
already holds that device open and only one process can have it at a time.
This means start_server.py (with the fisheye feed configured) must already
be running before this script is started.
"""
import json
import logging
import os
import re
import sys
import threading
import time

import cv2
import hydra
import numpy as np
import requests
from dotenv import load_dotenv
from omegaconf import OmegaConf

from utils.zmq_utils import ZMQCameraSubscriber

# Load before hydra changes the working directory (same reasoning as run.py).
load_dotenv()

logger = logging.getLogger(__name__)

BASE_STEP = 0.02  # meters per search move, matches run.py's interactive bl/br
MAX_SEARCH_STEPS = 10  # give up searching after this many VLM-queried iterations
MAX_STEPS_PER_MOVE = 5  # cap on how many BASE_STEPs Gemini ER can request per iteration
STEPS_PER_BATCH = 5
MAX_TOTAL_POLICY_STEPS = 100
HOME_HEIGHT = 0.85  # lift height for the pre-search home position
RECORD_FPS = 5

WIDE_CAMERA_URL_DEFAULT = "http://127.0.0.1:7860/feed/fisheye/snapshot"
# Anchored to this script's own directory (not cwd), since @hydra.main changes
# the working directory by default -- a relative path here would otherwise
# land inside Hydra's auto-generated per-run output folder.
VIDEO_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")


def _get_vlm_client():
    from google import genai

    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _encode_png(image):
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode image for VLM query")
    return buffer.tobytes()


def _locate_for_search(vlm_client, wide_image, wrist_image, object_name):
    """
    Ask Gemini ER whether the object is visible/centered enough in the wrist
    camera to proceed, and if not, where it is horizontally in the wide
    camera (always visible there) so we know which way to move the base, and
    how many base-move steps to take at once (more when far from center,
    so small offsets don't take many slow single-step iterations).
    """
    from google.genai import types

    prompt = (
        f"You are controlling a mobile robot searching for the object "
        f"'{object_name}' before grasping it.\n"
        "Image 1 is a wide-angle view from the robot's head; the object is "
        "always visible somewhere in image 1.\n"
        "Image 2 is a narrow field-of-view camera mounted at the robot's "
        "wrist, near the gripper; the object may or may not be in view "
        "there yet.\n\n"
        "Report:\n"
        '1. "visible_in_wrist": true only if the object is clearly and '
        "fully visible, reasonably centered, in image 2.\n"
        '2. "wide_x": the object\'s horizontal pixel position in image 1, '
        "normalized to the range 0-1000 (0 = left edge, 1000 = right edge). "
        "Always provide this, even if visible_in_wrist is true.\n"
        f'3. "steps": if visible_in_wrist is false, how many {BASE_STEP * 100:.0f}cm '
        f"base-movement steps to take at once before checking again, as an "
        f"integer from 1 to {MAX_STEPS_PER_MOVE}. Use more steps when the "
        "object is far from center (wide_x far from 500), fewer when it's "
        "close to center. Use 1 if visible_in_wrist is true.\n\n"
        'Respond with only JSON: {"visible_in_wrist": bool, "wide_x": int, "steps": int}'
    )

    logger.info("Querying Gemini ER for search (wide + wrist views)...")
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
    steps = max(1, min(MAX_STEPS_PER_MOVE, int(result.get("steps", 1))))
    return bool(result["visible_in_wrist"]), int(result["wide_x"]), steps


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
        logger.debug(f"Fetched wide-angle snapshot from {self.snapshot_url} shape={image.shape}")
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


def search_for_object(controller, wide_camera, vlm_client, object_name):
    """
    Move the base forward/backward until the object is visible in the wrist
    camera. Returns True once visible, False if MAX_SEARCH_STEPS is reached
    first.
    """
    logger.info(f"Starting search for '{object_name}' (max {MAX_SEARCH_STEPS} iterations)...")
    for attempt in range(MAX_SEARCH_STEPS):
        logger.info(f"--- Search attempt {attempt + 1}/{MAX_SEARCH_STEPS} ---")
        wrist_image, _np_depth, _timestamp = controller.subscriber.recv_image_and_depth()
        wide_image = wide_camera.read()

        visible, wide_x, steps = _locate_for_search(vlm_client, wide_image, wrist_image, object_name)
        logger.info(f"visible_in_wrist={visible} wide_x={wide_x} steps={steps}")
        if visible:
            logger.info(f"'{object_name}' is now visible in the wrist camera.")
            return True

        # Object right of center -> move backward to bring it toward center;
        # left of center -> move forward. (Confirmed empirically on hardware;
        # opposite of the earlier forward=view-shifts-left assumption.)
        # Gemini ER picks how many BASE_STEPs to take at once, so large
        # offsets from center converge in fewer iterations.
        distance = BASE_STEP * steps
        if wide_x > 500:
            logger.info(f"Object right of center (wide_x={wide_x}) -- moving backward {distance:.3f}m ({steps} steps).")
            controller.robot.move_base(-distance)  # backward
        else:
            logger.info(f"Object left of center (wide_x={wide_x}) -- moving forward {distance:.3f}m ({steps} steps).")
            controller.robot.move_base(distance)  # forward

    logger.warning(f"Gave up searching for '{object_name}' after {MAX_SEARCH_STEPS} iterations.")
    return False


def run_pick(controller, object_name):
    """
    Run the existing use_vlm=true pick policy in batches, stopping once the
    gripper closes (controller.gripper below closing_threshold -- the same
    signal _run_policy_goals already uses internally to end an episode
    early).
    """
    logger.info(f"Starting grasp policy for '{object_name}' (batches of {STEPS_PER_BATCH} steps)...")
    controller.reset_experiment(gripper=1.0, object_name=object_name)
    total_steps = 0
    while total_steps < MAX_TOTAL_POLICY_STEPS:
        logger.info(f"Running batch: steps {total_steps}-{total_steps + STEPS_PER_BATCH}...")
        steps_before = controller.step_n
        controller._run(run_for=STEPS_PER_BATCH)
        total_steps += controller.step_n - steps_before
        logger.info(f"Batch done: gripper={controller.gripper:.3f} (closing_threshold={controller.closing_threshold})")
        if controller.gripper < controller.closing_threshold:
            logger.info(f"Grasp detected (gripper={controller.gripper:.3f}) after {total_steps} steps.")
            return True
    logger.warning(f"Gave up after {total_steps} steps without a grasp.")
    return False


@hydra.main(config_path="configs", config_name="run_vqbet_pick", version_base="1.2")
def main(cfg):
    from run import _init_model_loss
    from robot.controller import Controller

    object_name = cfg.get("object")
    if not object_name:
        raise SystemExit("Pass the target object, e.g.: python pick.py +object='red mug'")

    wide_camera_url = cfg.get("wide_camera_url", WIDE_CAMERA_URL_DEFAULT)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = cfg.get("video_path") or os.path.join(VIDEO_DIR_DEFAULT, f"pick_{timestamp}_fisheye.mp4")
    wrist_video_path = cfg.get("wrist_video_path") or os.path.join(VIDEO_DIR_DEFAULT, f"pick_{timestamp}_wrist.mp4")

    logger.info(f"Target object: '{object_name}'")
    logger.info(f"Wide-angle camera URL: {wide_camera_url}")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["use_vlm"] = True

    logger.info("Loading model/checkpoint...")
    model = _init_model_loss(cfg)
    controller = Controller(cfg=cfg_dict)
    controller.setup_model(model)

    wide_camera = WideAngleCamera(wide_camera_url)
    vlm_client = _get_vlm_client()
    fisheye_recorder = VideoRecorder(wide_camera.read, video_path, label="fisheye video")

    # Dedicated subscriber for recording -- never shared with controller's
    # own self.subscriber, since ZMQ SUB sockets aren't safe for concurrent
    # use across threads.
    network_cfg = cfg_dict["network"]
    wrist_record_subscriber = ZMQCameraSubscriber(
        network_cfg.get("remote", "127.0.0.1"), network_cfg["camera_port"], "RGBD"
    )
    wrist_recorder = VideoRecorder(
        lambda: wrist_record_subscriber.recv_image_and_depth()[0], wrist_video_path, label="wrist video"
    )

    try:
        fisheye_recorder.start()
        wrist_recorder.start()

        # Known starting pose, so the search's forward/backward logic isn't
        # thrown off by wherever the arm was left from a previous run.
        logger.info(f"Homing (lift={HOME_HEIGHT})...")
        controller.robot.set_home_position(lift=HOME_HEIGHT)
        controller.robot.home(gripper=1.0, reset_base=True)

        if not search_for_object(controller, wide_camera, vlm_client, object_name):
            return

        success = run_pick(controller, object_name)
        logger.info(f"Pick {'succeeded' if success else 'failed'} for '{object_name}'.")
    finally:
        fisheye_recorder.stop()
        wrist_recorder.stop()
        wide_camera.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and any(arg in ["-h", "--help", "help"] for arg in sys.argv):
        print(__doc__)
        sys.exit(0)
    main()
