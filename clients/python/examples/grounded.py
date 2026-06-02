"""Open-vocab detection/segmentation with a text prompt (Grounded-SAM / Grounding DINO).

Usage:
    # Make sure the server is up:  make serve
    python examples/grounded.py path/to/image.jpg --prompt "cat. remote." [--model grounded-sam]

NOTE: the target model must be available on the server (check `client.list_models()`).
Full Grounding DINO / Grounded-SAM may be a paid tier or not yet shipped — this example
will print a clear error from the server if the model is unavailable.
"""

import argparse

from visionserve import Client, VisionServeError


def main() -> None:
    ap = argparse.ArgumentParser(description="VisionServe open-vocab (text prompt) example")
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("--host", default="http://localhost:11435")
    ap.add_argument("--model", default="grounded-sam")
    ap.add_argument("--prompt", default="cat. remote.", help='text prompt, e.g. "cat. remote."')
    args = ap.parse_args()

    client = Client(host=args.host)

    # Show what's available so the user can pick a real model name.
    print("available models:")
    for m in client.list_models():
        print("  %-16s task=%-12s license=%-12s state=%s" % (m.name, m.task, m.license, m.state))

    try:
        client.load(args.model)
        result = client.predict(args.model, args.image, prompt=args.prompt)
    except VisionServeError as e:
        print("server error: %s" % e)
        return

    print("task=%s model=%s duration_ms=%.1f" % (result.task, result.model, result.duration_ms))
    print("detections: %d" % len(result.detections))
    for d in result.detections:
        print("  %-16s conf=%.3f  bbox=%s" % (d.cls, d.conf, d.bbox))
    print("masks: %d" % len(result.masks))


if __name__ == "__main__":
    main()
