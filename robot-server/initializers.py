from multiprocessing import Process
from robot.zmq_utils import *
import hydra


#Builds the list of {name, host, port, topic_type} feeds that the camera web
#viewer should subscribe to, derived from the same camera config used to
#start the publisher process(es), so the viewer always matches whatever
#camera=... was selected.
def _derive_camera_feeds(camera_cfg):
    entries = {"camera": camera_cfg} if "_target_" in camera_cfg else camera_cfg
    feeds = []
    for name, sub_cfg in entries.items():
        host = sub_cfg.get("host", "127.0.0.1")
        if host in ("0.0.0.0", "*"):
            host = "127.0.0.1"
        port = sub_cfg.get("port", sub_cfg.get("camera_port"))
        use_depth = sub_cfg.get("use_depth", sub_cfg.get("stream_depth", False))
        feeds.append({
            "name": name,
            "host": host,
            "port": port,
            "topic_type": "RGBD" if use_depth else "RGB",
        })
    return feeds


#Runs in its own process; kept as a module-level function so it can be
#pickled as a Process target.
def _run_camera_web_viewer(feeds, host, port):
    from camera.web_viewer import CameraWebViewer
    CameraWebViewer(feeds, host=host, port=port).stream()


class StartServer(ProcessInstantiator):
    """
    Returns all processes. Start the list of processes
    to run the robot.
    """
    def __init__(self, configs):
        super().__init__()
        self.configs=configs

        self._init_camera_process()
        self._init_viewer_process()
        self._init_robot_process()

    #Function to start the components
    def _start_component(self, configs):
        # print(configs)
        # assert False
        component = hydra.utils.instantiate(configs)
        component.stream()

    #Function to start camera process(es). A camera config is either a single
    #component (top-level _target_, e.g. iphone.yaml) or a mapping of several
    #named components (e.g. cameras.yaml's camera1/camera3), each of which
    #gets its own process.
    def _init_camera_process(self):
        camera_cfg = self.configs.camera
        if "_target_" in camera_cfg:
            self.processes.append(Process(
                target = self._start_component,
                args = (camera_cfg, )
            ))
        else:
            for _, sub_cfg in camera_cfg.items():
                self.processes.append(Process(
                    target = self._start_component,
                    args = (sub_cfg, )
                ))

    #Function to start a single webpage that shows all configured camera
    #feeds, instead of each camera process popping up its own cv2 window.
    def _init_viewer_process(self):
        feeds = _derive_camera_feeds(self.configs.camera)

        # The head-mounted wide-angle/fisheye camera isn't published over ZMQ
        # by any camera process -- it's accessed directly over V4L2 (see
        # capture_wide_angle_photo.py), so it's added here instead of coming
        # from camera_cfg.
        fisheye_device = self.configs.network.get("wide_camera_device", "/dev/video6")
        if fisheye_device:
            feeds.append({"name": "fisheye", "type": "v4l2", "device": fisheye_device})

        if not feeds:
            return
        viewer_port = self.configs.network.get("viewer_port", 7860)
        self.processes.append(Process(
            target = _run_camera_web_viewer,
            args = (feeds, "0.0.0.0", viewer_port)
        ))

    def _init_robot_process(self):
        self.processes.append(Process(
            target = self._start_component,
            args = (self.configs.controller, )
        ))

class StickTeleop(ProcessInstantiator):
    """
    Returns all the teleoperation processes. Start the list of processes 
    to run the teleop.
    """
    def __init__(self, configs):
        super().__init__()
        self.configs=configs
      
        self._init_camera_process()
        
    #Function to start the components
    def _start_component(self, configs):
        component = hydra.utils.instantiate(configs)
        component.stream()

    #Function to start camera process
    def _init_camera_process(self):
        self.processes.append(Process(
            target = self._start_component,
            args = (self.configs.camera, )
        ))

class StartScript(ProcessInstantiator):
    def __init__(self, configs):
        super().__init__()
        self.configs=configs
      
        self._init_camera_process()
        
    #Function to start the components
    def _start_component(self, configs):
        component = hydra.utils.instantiate(configs)
        component.stream()

    #Function to start camera process
    def _init_camera_process(self):
        self._start_component(self.configs.camera)