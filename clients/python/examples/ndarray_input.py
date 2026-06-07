"""Run inference on an in-memory ``numpy.ndarray`` frame (e.g. an OpenCV camera grab).

``Client.predict()`` accepts a ``numpy.ndarray`` directly (HWC ``uint8``, or float in
``[0, 1]``; grayscale ``(H, W)`` is promoted to RGB), so you can feed a frame straight
from a camera / video pipeline with NO file round-trip — the client JPEG-encodes it for
you.

IMPORTANT — colour order: the client treats the array as **RGB**. OpenCV delivers frames
in **BGR**, so convert BGR→RGB first (``cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`` or
``frame[:, :, ::-1]``). Skip this and the red/blue channels are swapped server-side, which
quietly wrecks detections and colours.

Usage:
    # Make sure the server is up:  make serve
    #
    # Generic — any loaded model; pass a file that stands in for a captured frame:
    python examples/ndarray_input.py --model rf-detr --image frame.jpg
    python examples/ndarray_input.py --model grounding-dino --image frame.jpg --prompt "cat. dog."
    python examples/ndarray_input.py --model grasp --image frame.jpg --out annotated.jpg
    #
    # No --image → a synthetic gradient frame is generated so it runs headless.

Requires: numpy + pillow  (pip install 'visionserve[images]'). opencv-python is optional —
used to mimic a real BGR camera grab when available; otherwise the frame is read with PIL.
"""

import argparse

import numpy as np

from visionserve import Client, VisionServeError, draw


def grab_frame_rgb(path):
    """Return an ``(HWC uint8 RGB ndarray, source_label)`` tuple — a stand-in for a frame
    grabbed from a camera.

    Prefers OpenCV (which reads/captures as BGR, like a real camera) and converts BGR→RGB;
    falls back to PIL (already RGB); falls back to a synthetic gradient when no path given.
    """
    if path:
        try:
            import cv2

            frame_bgr = cv2.imread(path)  # OpenCV decodes to BGR uint8, like cap.read()
            if frame_bgr is None:
                raise FileNotFoundError("cv2 could not read %r" % path)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # <-- the crucial conversion
            return np.ascontiguousarray(rgb), "opencv (BGR->RGB)"
        except ImportError:
            from PIL import Image

            return np.asarray(Image.open(path).convert("RGB")), "PIL (already RGB)"

    # No file: synthesize a frame so the example runs with no camera and no image on disk.
    h, w = 240, 320
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]  # red ramp →
    frame[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]  # green ramp ↓
    return frame, "synthetic gradient"


def main() -> None:
    ap = argparse.ArgumentParser(description="VisionServe ndarray (in-memory frame) example")
    ap.add_argument("--model", default="rf-detr", help="any loaded model name")
    ap.add_argument("--host", default="http://localhost:11435")
    ap.add_argument("--image", default=None,
                    help="image file used as the 'captured frame' (optional; omit for synthetic)")
    ap.add_argument("--prompt", default=None,
                    help="text prompt (open-vocab / grasp-gd models)")
    ap.add_argument("--out", default=None, help="optional annotated image output path")
    args = ap.parse_args()

    # 1) Obtain a frame as a numpy.ndarray (no file is sent to the server).
    frame, source = grab_frame_rgb(args.image)
    print("frame: shape=%s dtype=%s source=%s" % (frame.shape, frame.dtype, source))
    if frame.dtype != np.uint8 or frame.ndim != 3:
        raise SystemExit("expected an HWC uint8 RGB frame, got %s %s" % (frame.shape, frame.dtype))

    # 2) Predict straight from the ndarray.
    client = Client(host=args.host)
    try:
        client.load(args.model)
        result = client.predict(args.model, frame, prompt=args.prompt)  # <-- ndarray in
    except VisionServeError as e:
        print("server error: %s" % e)
        return

    print("task=%s model=%s device=%s duration_ms=%.1f"
          % (result.task, result.model, result.device or "?", result.duration_ms))
    print("detections: %d  masks: %d  grasps: %d  classifications: %d"
          % (len(result.detections), len(result.masks),
             len(result.grasps), len(result.classifications)))
    for d in result.detections[:10]:
        print("  det %-16s %.2f  bbox=%s" % (d.cls, d.conf, [round(v, 1) for v in d.bbox]))

    # 3) Visualize — draw() takes PIL/path/bytes (not ndarray), so wrap the frame.
    if args.out:
        from PIL import Image

        annotated = draw(result, Image.fromarray(frame))
        annotated.save(args.out)
        print("saved annotated image -> %s" % args.out)


if __name__ == "__main__":
    main()
