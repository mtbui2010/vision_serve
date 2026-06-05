"""Build a fixed POST body for /api/predict so wrk2/vegeta replay identical bytes.

Both wrk2 (via a Lua script) and vegeta (via a targets file) need a pre-built JSON body. This
script emits that body once from a chosen image, so every system and every run replays the
*same* request bytes (a requirement for a fair W1/W4 comparison).

    python -m eval.loadgen.make_payload --model mobilenet-v3 --image test/testdata/sample.jpg \
        --out /tmp/payload.json
"""

from __future__ import annotations

import argparse
import json

from eval.common.api import build_json_request, encode_image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True, help="path to the image to replay")
    ap.add_argument("--prompt", default=None, help="text prompt (GroundingDINO etc.)")
    ap.add_argument("--box", default=None, help="box prompt 'x,y,w,h' (MobileSAM)")
    ap.add_argument("--out", required=True, help="output JSON body path")
    args = ap.parse_args()

    body = build_json_request(
        args.model, encode_image(args.image), prompt=args.prompt, box=args.box
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(body, f)
    print(f"wrote {args.out} (model={args.model}, image={args.image})")


if __name__ == "__main__":
    main()
