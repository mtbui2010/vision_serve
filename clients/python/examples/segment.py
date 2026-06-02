"""Segment an object with MobileSAM using a box prompt, and decode the mask to numpy.

Usage:
    # Make sure the server is up:  make serve
    python examples/segment.py path/to/image.jpg --box 50,40,120,90 [--save mask.png]

The box is "x,y,w,h" in ORIGINAL image pixels.
"""

import argparse

from visionserve import Client


def main() -> None:
    ap = argparse.ArgumentParser(description="VisionServe MobileSAM box-prompt example")
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("--host", default="http://localhost:11435")
    ap.add_argument("--model", default="mobile-sam")
    ap.add_argument("--box", required=True, help='box "x,y,w,h" in original pixels')
    ap.add_argument("--save", default=None, help="optional path to save the mask PNG")
    args = ap.parse_args()

    box = [float(v) for v in args.box.split(",")]

    client = Client(host=args.host)
    client.load(args.model)

    result = client.predict(args.model, args.image, box=box)
    print("task=%s model=%s masks=%d duration_ms=%.1f"
          % (result.task, result.model, len(result.masks), result.duration_ms))

    if not result.masks:
        print("no masks returned")
        return

    # We need the original image size to decode the column-major RLE.
    try:
        from PIL import Image
    except ImportError:
        print("install pillow/numpy to decode the mask to an ndarray")
        return
    w, h = Image.open(args.image).size

    mask = result.masks[0]
    arr = mask.to_ndarray(width=w, height=h)  # bool (H, W)
    print("mask ndarray shape=%s  positive_pixels=%d  conf=%.3f"
          % (arr.shape, int(arr.sum()), mask.conf))

    if args.save:
        Image.fromarray((arr * 255).astype("uint8")).save(args.save)
        print("saved mask to %s" % args.save)


if __name__ == "__main__":
    main()
