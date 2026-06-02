"""Run RF-DETR detection on an image file and print (optionally draw) the boxes.

Usage:
    # Make sure the server is up:  make serve   (in the repo root)
    python examples/detect.py path/to/image.jpg [--model rf-detr] [--save out.png]
"""

import argparse

from visionserve import Client


def main() -> None:
    ap = argparse.ArgumentParser(description="VisionServe RF-DETR detection example")
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("--host", default="http://localhost:11435")
    ap.add_argument("--model", default="rf-detr")
    ap.add_argument("--save", default=None, help="optional output path to draw boxes")
    args = ap.parse_args()

    client = Client(host=args.host)

    # load() is optional — the server may auto-load — but doing it explicitly gives a
    # clearer error if the model/weights are missing.
    client.load(args.model)

    result = client.predict(args.model, args.image)
    print("task=%s model=%s duration_ms=%.1f" % (result.task, result.model, result.duration_ms))
    print("detections: %d" % len(result.detections))
    for d in result.detections:
        x, y, w, h = d.bbox
        print("  %-16s conf=%.3f  bbox=[%.1f, %.1f, %.1f, %.1f]" % (d.cls, d.conf, x, y, w, h))

    if args.save:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            print("(install pillow to use --save)")
            return
        img = Image.open(args.image).convert("RGB")
        draw = ImageDraw.Draw(img)
        for d in result.detections:
            x, y, w, h = d.bbox
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
            draw.text((x, max(0, y - 10)), "%s %.2f" % (d.cls, d.conf), fill=(255, 0, 0))
        img.save(args.save)
        print("saved annotated image to %s" % args.save)


if __name__ == "__main__":
    main()
