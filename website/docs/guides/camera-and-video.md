# Run a model on a camera or video

## When you need this

You have a webcam, an IP camera, a video file, an industrial camera or an RGB-D camera
(RealSense, Orbbec, a ROS 2 topic), and you want results continuously, not one photo at a time.
The server stays as it is: the Python SDK reads the camera and sends one ordinary request per
frame, skipping frames it cannot keep up with so the answers stay live.

## The command

```console
$ pip install 'visionserve[opencv]'
$ visionserve watch 0 --model rf-detr --fps 5
```

`0` is the first webcam. Put a video file, `rtsp://...`, `gst:<pipeline>`, `realsense`,
`orbbec` or `ros2:<topic>` there instead (table below). `--fps 5` sends at most five frames a
second; leave it out to go as fast as the server answers. Stop with Ctrl-C, `--max-frames N` or
`--duration SEC`. In Python, the same loop is
`for res in Client().watch(0, "rf-detr", fps=5): ...`.

`watch` belongs to the Python SDK's command (`pip install visionserve`), which talks to a running
server (`visionserve serve`). If the Go binary comes first on your `PATH` (it answers
`unknown command: "watch"`), run `python -m visionserve watch ...`.

## Reading the result

The development PC has no camera, so here the source is `slideshow.mp4`, a 4-second, 10 fps clip
of four COCO photos (one second each). A video file plays at its own speed, like a camera:

```console
$ visionserve watch slideshow.mp4 --model rf-detr --fps 5 --track
frame 0      (3 detections) laptop 0.94 #1, cat 0.81 #2, person 0.56 #3  server 150 ms
frame 8      (4 detections) laptop 0.95 #1, cat 0.83 #2, person 0.54 #3, couch 0.51 #4  server 59 ms
frame 10     (3 detections) broccoli 0.85 #5, bowl 0.73 #6, carrot 0.71 #7  server 167 ms
frame 14     (3 detections) broccoli 0.85 #5, bowl 0.75 #6, carrot 0.71 #7  server 58 ms
...
frame 39     (7 detections) couch 0.96 #14, tv 0.94 #15, chair 0.91 #16, book 0.76 #17, chair 0.73 #18, ...  server 34 ms
watch: 16 frames in 4.6 s (3.5 per second); 24 skipped to stay real time
```

| Part | In plain words |
|---|---|
| `frame 8` | The frame's number in the stream. The jump from 0 to 8 means frames 1 to 7 arrived while the first request ran and were skipped: the next request always takes the newest frame. |
| `(4 detections) laptop 0.95 ...` | What the model found, best first (five at most on the line). |
| `#1`, `#2` | With `--track`: the object's id across frames, from a simple tracker on your side. The same object keeps its id while it moves less than about its own size between two processed frames. |
| `server 59 ms` | The model's own time on the server. The rest of each request is the upload and decoding. |
| `watch: 16 frames in 4.6 s` | On stderr at the end: how many frames were processed and how many were skipped. |

`--json` prints one JSON object per frame instead (`frame_id`, `timestamp`, and the
[result](../reference/api.md)), for scripts.

### Other sources

| Source | Example | Install |
|---|---|---|
| Webcam, V4L2 device | `0`, `/dev/video2` | `pip install 'visionserve[opencv]'` |
| Video file | `clip.mp4` | same |
| IP camera (RTSP, HTTP) | `rtsp://user:pass@192.168.1.20:554/stream1` | same; reconnects when the stream drops |
| GStreamer pipeline | `gst:rtspsrc location=... ! ... ! nvvidconv` | GStreamer from your OS |
| Intel RealSense | `realsense`, `realsense:<serial>` | `pip install 'visionserve[realsense]'` |
| Orbbec (SDK v2) | `orbbec`, `orbbec:1` | `pip install 'visionserve[orbbec]'` |
| ROS 2 topics | `ros2:/camera/color/image_raw,/camera/aligned_depth_to_color/image_raw` | `source /opt/ros/<distro>/setup.bash` |

### GStreamer: hardware decoding and special cameras

`gst:` runs your pipeline with `gst-launch-1.0` and adds the conversion to RGB at the end, so any
camera or decoder GStreamer supports works, without Python bindings. End the pipeline at the
decoded video; on Jetson, end it with `nvvidconv` (it copies frames out of GPU memory).

```bash
# RTSP, H.264, decoded by the Jetson's hardware decoder
visionserve watch "gst:rtspsrc location=rtsp://192.168.1.20:554/stream1 latency=100 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv" --model rf-detr --fps 5
# The same on a PC with VA-API (Intel / AMD), or in software
visionserve watch "gst:rtspsrc location=rtsp://192.168.1.20:554/stream1 ! rtph264depay ! h264parse ! vaapih264dec" --model rf-detr
visionserve watch "gst:rtspsrc location=rtsp://192.168.1.20:554/stream1 ! rtph264depay ! h264parse ! avdec_h264" --model rf-detr
# A Jetson CSI camera
visionserve watch "gst:nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv" --model rf-detr
# A GigE Vision / USB3 Vision industrial camera (Aravis)
visionserve watch "gst:aravissrc camera-name=Basler-12345678 ! bayer2rgb" --model rf-detr
```

These five lines are examples for hardware this PC does not have; they were not run here. The
GStreamer adapter itself was run with `videotestsrc` and with a file. Dropping frames inside
GStreamer is cheaper than decoding them in Python and skipping them, so for a low rate put
`videorate` in the pipeline. Decoding the clip above (as MJPG in an AVI, since this PC's
GStreamer has no MPEG-4 decoder) and keeping two frames a second:

```console
$ visionserve watch "gst:filesrc location=slideshow.avi ! decodebin ! videorate drop-only=true ! video/x-raw,framerate=2/1" \
      --model rf-detr --every-frame
frame 0      (4 detections) laptop 0.94, cat 0.88, person 0.61, couch 0.54  server 261 ms
frame 1      (4 detections) laptop 0.93, cat 0.85, couch 0.53, person 0.51  server 19 ms
frame 2      (3 detections) broccoli 0.87, bowl 0.73, carrot 0.68  server 45 ms
...
frame 7      (7 detections) couch 0.96, tv 0.94, chair 0.91, chair 0.74, book 0.72, ...  server 25 ms
watch: 8 frames in 1.3 s (6.1 per second); 0 skipped to stay real time
```

`--every-frame` sends every frame the pipeline delivers (here, a file decoded faster than real
time). In Python, `GStreamerSource(pipeline, width=W, height=H, sync=True, restart=True)` also
scales in GStreamer, plays a file at its own speed, and restarts a pipeline that ends.

!!! tip "DeepStream and GStreamer apps: VisionServe as the expert on key frames"
    A DeepStream or GStreamer pipeline is the right tool for every frame of many streams: a
    small detector and a tracker at full frame rate, on the hardware decoder's buffers. VisionServe
    fits next to it as the expert you ask now and then: an open-vocabulary detector
    (`grounding-dino`), masks (`grounded-sam`), grasps, on **key frames** only: when the tracker
    sees a new object, once a second, or when an operator asks. Feed it the same stream at a low
    rate (`videorate` as above, or `--fps 1`), or call `Client.predict()` from your app on the
    frames you pick. They run as separate processes, so a slow expert answer does not block the
    real-time pipeline (on one GPU they still share its compute).

### Depth cameras

With a RealSense, an Orbbec or a ROS 2 depth topic, every frame carries a depth image aligned to
the colour image. `--depth auto` (the default) sends it only to models that read it: the server
lists them in `GET /api/models` as `accepts_depth`, and today that is `background`, which then
fits the support surface to your depth instead of estimating depth with MiDaS:

```console
$ visionserve watch realsense --model background --method depth
```

The grasp models do not read depth; in Python, keep it for 3-D on your side with
`watch(..., return_frames=True)` and `grasp_distances(res.frame.depth, res.grasps, ...)` (see
[Watching a camera or video](../clients/python.md#depth)).

## If it goes wrong

| Message | What to do |
|---|---|
| `error: this source needs the 'cv2' module: pip install 'visionserve[opencv]' ...` (or `pyrealsense2`, `pyorbbecsdk`) | Install the extra it names. For ROS 2, run from a shell where you sourced `/opt/ros/<distro>/setup.bash`. |
| `error: OpenCV could not open 7 (no such camera, file or stream, or no backend for it)` | Wrong camera index, the camera is in use by another program, or your user may not read `/dev/video*` (add it to the `video` group). `v4l2-ctl --list-devices` lists the cameras. |
| `error: no such video file or device: 'clip.mp4' ...` | The path is wrong. A camera is a number like `0`, a stream starts with `rtsp://` or `http://`. |
| `error: GStreamer pipeline '...' ended (exit code 1) before its first frame:` then `no element "..."` | That GStreamer element is not installed: `gst-inspect-1.0 <element>` checks; install the plugin package that has it. |
| the same, then `Missing decoder: ...` | No decoder for the video's codec: install `gstreamer1.0-libav` (software) or use the hardware decoder of your platform (`nvv4l2decoder` on Jetson). |
| `error: depth='always' but the source gave a frame without depth (frame 0) ...` | The source has no depth (a webcam or file). Use `--depth auto`, or a depth camera. |
| `Frame.depth is 640x480 but Frame.color is 1280x720: depth must be ALIGNED ...` | Turn on depth-to-colour alignment in the camera driver (RealSense ROS: `align_depth.enable:=true`, then subscribe to `/camera/aligned_depth_to_color/image_raw`). |
| `unknown command: "watch"` | That is the Go binary. Run `python -m visionserve watch ...`, or put the SDK's `visionserve` first on your `PATH`. |
| `error: failed to reach VisionServe at http://localhost:11435/...` | Start the server (`visionserve serve`), or pass `--host`. |
| `error: POST /api/predict -> 404: ... is not in the registry` | Pull the model, or check its name with `visionserve list`. |
| `watch: the source is still blocked in read() ...` (a log line when you stop) | An RTSP stream stalled inside OpenCV, which waits for its own timeout. The program still ends. For flaky streams, a `gst:` pipeline with `GStreamerSource(..., restart=True)` restarts cleanly. |
| Many frames skipped, answers late | The model is slower than the camera; that is what skipping is for. To process more frames: run the server on a GPU, pick a smaller model, try `--in-flight 2` (uploads overlap inference), or shrink the upload with `--resize 960`. To process **every** frame of a file, add `--every-frame`. |

## Want the details?

- Every option, your own sources, the depth wire format and the tracker:
  [Python SDK: Watching a camera or video](../clients/python.md#watching-a-camera-or-video).
- `accepts_depth` and the depth request fields: [HTTP API](../reference/api.md).
- Running the server itself on a Jetson: [Make it smaller and faster for Jetson](jetson.md).

<small>Run on 6 October 2026 with SDK 0.3.0 against a server on an RTX A6000 shared with other
jobs (CUDA, ONNX Runtime 1.26); GStreamer 1.20. Photos: COCO val2017 #177015, #389381, #372819 and
#29596 (CC BY 2.0, see
[CREDITS.md](https://github.com/mtbui2010/visionserve/blob/main/website/docs/assets/img/CREDITS.md)).</small>
