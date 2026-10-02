import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, render_template_string

from robot.zmq_utils import ZMQCameraSubscriber

# Placeholder shown for a feed before its first frame arrives.
_BLANK_FRAME = np.zeros((256, 256, 3), dtype=np.uint8)

# Frames to discard after opening a V4L2 device, to let auto-exposure/buffer
# settle before trusting a frame (see capture_wide_angle_photo.py).
V4L2_WARMUP_FRAMES = 30

_PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Robot Camera Feeds</title>
  <style>
    body { background: #111; color: #eee; font-family: sans-serif; margin: 0; padding: 16px; }
    h1 { font-size: 1.2rem; }
    .grid { display: flex; flex-wrap: wrap; gap: 16px; }
    figure { margin: 0; }
    figcaption { text-align: center; margin-top: 4px; }
    img { max-width: 480px; width: 100%; height: auto; background: #000; }
  </style>
</head>
<body>
  <h1>Robot Camera Feeds</h1>
  <div class="grid">
    {% for name in names %}
    <figure>
      <img src="{{ url_for('feed', name=name) }}">
      <figcaption>{{ name }}</figcaption>
    </figure>
    {% endfor %}
  </div>
</body>
</html>
"""


class CameraWebViewer:
    """
    Subscribes to the ZMQ camera feeds published by the robot server's camera
    processes and serves them as a single MJPEG webpage, replacing the
    per-process cv2.imshow popups.
    """

    def __init__(self, feeds, host="0.0.0.0", port=7860, fps=15):
        self.host = host
        self.port = port
        self.frame_interval = 1.0 / fps
        self.feeds = feeds
        self.latest_frames = {feed["name"]: None for feed in feeds}
        self.locks = {feed["name"]: threading.Lock() for feed in feeds}

        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", self._index)
        self.app.add_url_rule("/feed/<name>", "feed", self._feed)
        self.app.add_url_rule("/feed/<name>/snapshot", "snapshot", self._snapshot)

    def _index(self):
        return render_template_string(_PAGE_TEMPLATE, names=list(self.latest_frames.keys()))

    def _read_loop(self, feed):
        if feed.get("type", "zmq") == "v4l2":
            self._read_loop_v4l2(feed)
        else:
            self._read_loop_zmq(feed)

    def _read_loop_zmq(self, feed):
        subscriber = ZMQCameraSubscriber(
            host=feed["host"], port=feed["port"], topic_type=feed["topic_type"]
        )
        name = feed["name"]
        while True:
            if feed["topic_type"] == "RGBD":
                image, _depth, _timestamp = subscriber.recv_image_and_depth()
            else:
                image, _timestamp = subscriber.recv_rgb_image()
            with self.locks[name]:
                self.latest_frames[name] = image

    def _read_loop_v4l2(self, feed):
        """
        Direct V4L2 access for cameras not published over ZMQ by any
        robot-server camera process (e.g. the head-mounted wide-angle/fisheye
        webcam) -- same approach as capture_wide_angle_photo.py.
        """
        name = feed["name"]
        cap = cv2.VideoCapture(feed["device"])
        if not cap.isOpened():
            print(f"Warning: could not open {name} camera device {feed['device']!r}")
            return
        try:
            # This camera's auto-exposure/buffer starts out black; discard
            # the first batch of frames before trusting one (same warm-up
            # capture_wide_angle_photo.py already does).
            for _ in range(V4L2_WARMUP_FRAMES):
                cap.read()
            while True:
                ok, image = cap.read()
                if not ok:
                    continue
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                with self.locks[name]:
                    self.latest_frames[name] = image
        finally:
            cap.release()

    def _get_frame_jpeg(self, name):
        with self.locks[name]:
            image = self.latest_frames[name]
        if image is None:
            image = _BLANK_FRAME
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buffer.tobytes() if ok else None

    def _mjpeg_generator(self, name):
        while True:
            jpeg = self._get_frame_jpeg(name)
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(self.frame_interval)

    def _feed(self, name):
        if name not in self.latest_frames:
            return "Unknown camera feed: {}".format(name), 404
        return Response(
            self._mjpeg_generator(name),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _snapshot(self, name):
        """
        Single-frame JPEG, for programmatic polling (e.g. pick.py) instead of
        the continuous MJPEG stream -- lets another process read the current
        frame without fighting this one for exclusive access to the camera.
        """
        if name not in self.latest_frames:
            return "Unknown camera feed: {}".format(name), 404
        jpeg = self._get_frame_jpeg(name)
        if jpeg is None:
            return "No frame available yet", 503
        return Response(jpeg, mimetype="image/jpeg")

    def stream(self):
        for feed in self.feeds:
            thread = threading.Thread(target=self._read_loop, args=(feed,), daemon=True)
            thread.start()
        self.app.run(host=self.host, port=self.port, threaded=True)
