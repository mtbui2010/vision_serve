"""Command-line interface for the VisionServe Python client.

This is a thin CLI over :class:`visionserve.Client` — it performs NO inference
itself, it talks to a running VisionServe server (the Go runtime) over HTTP
(default ``http://localhost:11435``). Start the server first with
``visionserve serve`` (the Go binary), then use this CLI to drive it.

Installed as the ``visionserve`` console command (see ``pyproject.toml``)::

    visionserve predict rf-detr cat.jpg
    visionserve predict grounding-dino kitchen.jpg --prompt "cat. remote."
    visionserve predict mobile-sam dog.jpg --box 50,40,200,180 --save
    visionserve list
    visionserve ps

Design notes:
  * ``predict`` prints the result as JSON to **stdout** (pipe-friendly) and a
    one-line human summary (model/task/device + timings) to **stderr**.
  * The reported duration is split into ``client`` (wall-clock around the
    ``predict()`` HTTP round-trip) and ``server`` (the ``duration_ms`` the
    server measured for inference only). Both are captured BEFORE any image is
    drawn or saved, so visualization cost never inflates the reported latency.
  * ``--save`` writes an annotated image using an auto-generated, self-describing
    name ``<stem>.python.<model>.<task>.png`` so outputs from different clients /
    models / tasks never collide. ``--save-as PATH`` overrides the name.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .client import Client, VisionServeError
from .types import ModelInfo, Result

CLIENT_TYPE = "python"
DEFAULT_HOST = "http://localhost:11435"


# --------------------------------------------------------------------------- #
# Wire serialization (match the server schema in pkg/api/types.go exactly)
# --------------------------------------------------------------------------- #
def _result_to_wire(res: Result) -> Dict[str, Any]:
    """Reconstruct the server's JSON wire shape from a parsed :class:`Result`.

    Field names mirror ``pkg/api/types.go`` (notably ``class``, not ``cls``), and
    empty collections are omitted to match the Go ``omitempty`` tags — so the JSON
    printed here is identical to ``visionserve run`` (the Go CLI).
    """
    out: Dict[str, Any] = {"task": res.task, "model": res.model}
    if res.device:
        out["device"] = res.device
    if res.detections:
        out["detections"] = [
            {"bbox": list(d.bbox), "class": d.cls, "conf": d.conf} for d in res.detections
        ]
    if res.masks:
        out["masks"] = [
            {"rle": m.rle, "bbox": list(m.bbox), "conf": m.conf} for m in res.masks
        ]
    if res.grasps:
        grasps: List[Dict[str, Any]] = []
        for g in res.grasps:
            item: Dict[str, Any] = {
                "x": g.x,
                "y": g.y,
                "theta": g.theta,
                "width": g.width,
                "quality": g.quality,
            }
            if g.cls:
                item["class"] = g.cls
            if g.conf:
                item["conf"] = g.conf
            grasps.append(item)
        out["grasps"] = grasps
    if res.classifications:
        out["classifications"] = [
            {"class": c.cls, "conf": c.conf} for c in res.classifications
        ]
    if res.embeddings:
        out["embeddings"] = [list(v) for v in res.embeddings]
    if res.depth_map:
        out["depth_map"] = list(res.depth_map)
        out["depth_width"] = res.depth_width
        out["depth_height"] = res.depth_height
    out["duration_ms"] = res.duration_ms
    return out


def _auto_name(image_path: str, model: str, task: str, ext: str) -> str:
    """Build a self-describing output filename: ``<stem>.python.<model>.<task>.<ext>``.

    The ``client_type`` segment (``python``) distinguishes outputs from the JS / Go
    CLIs that may run against the same image + model.
    """
    stem = Path(image_path).stem or "image"
    task = task or "result"
    return f"{stem}.{CLIENT_TYPE}.{model}.{task}.{ext}"


# --------------------------------------------------------------------------- #
# Argument parsers
# --------------------------------------------------------------------------- #
def _parse_boxes(spec: Optional[str]) -> Optional[List[List[float]]]:
    """Parse ``"x,y,w,h"`` (boxes separated by ``;``) into a list of float boxes."""
    if not spec:
        return None
    boxes: List[List[float]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [float(v) for v in chunk.split(",")]
        if len(vals) != 4:
            raise ValueError("box %r must have 4 values x,y,w,h" % chunk)
        boxes.append(vals)
    return boxes or None


def _parse_points(spec: Optional[str]) -> Optional[List[List[float]]]:
    """Parse ``"x,y[,label]"`` (points separated by ``;``) into a list of points."""
    if not spec:
        return None
    points: List[List[float]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [float(v) for v in chunk.split(",")]
        if len(vals) not in (2, 3):
            raise ValueError("point %r must have 2 or 3 values x,y[,label]" % chunk)
        points.append(vals)
    return points or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionserve",
        description=(
            "VisionServe Python client CLI — drive a running VisionServe server "
            "over HTTP (run `visionserve serve` first; this CLI does no inference "
            "itself)."
        ),
        epilog=(
            "Examples:\n"
            "  visionserve predict rf-detr cat.jpg\n"
            "  visionserve predict grounding-dino kitchen.jpg --prompt 'cat. remote.'\n"
            "  visionserve predict mobile-sam dog.jpg --box 50,40,200,180 --save\n"
            "  visionserve predict grasp-gd bin.jpg --prompt 'mug.' --save-as out.png\n"
            "  visionserve list\n"
            "  visionserve ps --host http://10.0.0.5:11435\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version="visionserve-client %s" % __version__
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="server base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        metavar="SEC",
        help="per-request timeout in seconds (default: %(default)s)",
    )

    # Connection flags are also accepted AFTER the subcommand (e.g.
    # `predict ... --host URL`). SUPPRESS defaults so a subcommand-level flag only
    # overrides when actually passed — otherwise the top-level value stands.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    common.add_argument(
        "--timeout", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>", parser_class=argparse.ArgumentParser)
    sub.required = True

    # ---- predict ------------------------------------------------------------
    p = sub.add_parser(
        "predict",
        parents=[common],
        help="run inference on an image and print the result as JSON",
        description=(
            "Run inference on IMAGE with MODEL and print the unified result JSON to "
            "stdout. A one-line summary with client + server timings goes to stderr. "
            "Use --save / --save-as to also write an annotated image (requires Pillow)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model", help="model name, e.g. rf-detr / mobile-sam / grounding-dino / grasp-gd")
    p.add_argument("image", help="path to an input image (jpg/png)")
    p.add_argument(
        "--prompt",
        metavar="TEXT",
        help='open-vocab text prompt, e.g. "cat. remote." (GroundingDINO / Grounded-SAM / grasp-gd)',
    )
    p.add_argument(
        "--box",
        metavar="x,y,w,h",
        help="SAM box prompt(s) in ORIGINAL image pixels; multiple boxes separated by ';'",
    )
    p.add_argument(
        "--point",
        metavar="x,y[,l]",
        help="SAM point prompt(s); label 1=fg 0=bg (default 1); multiple separated by ';'",
    )
    p.add_argument(
        "--min-size",
        type=float,
        metavar="PCT",
        help="drop objects whose bbox area is below PCT%% of the image (e.g. 0.1)",
    )
    p.add_argument(
        "--max-size",
        type=float,
        metavar="PCT",
        help="drop objects whose bbox area is above PCT%% of the image (e.g. 90)",
    )
    p.add_argument(
        "--gripper-min",
        type=float,
        metavar="PX",
        help="grasp models only: minimum jaw opening in original-image pixels",
    )
    p.add_argument(
        "--gripper-max",
        type=float,
        metavar="PX",
        help="grasp models only: maximum jaw opening in original-image pixels",
    )
    p.add_argument(
        "--roi",
        metavar="x,y,w,h",
        help="region of interest in ORIGINAL pixels: process only this crop, map results back",
    )
    p.add_argument(
        "--method",
        metavar="NAME",
        help="background model: auto|depth|sam|cv|automask (default auto = depth->cv fallback)",
    )
    p.add_argument(
        "--box-threshold",
        type=float,
        metavar="T",
        help="GroundingDINO query-score threshold (grounding-dino/grounded-sam/grasp-gd)",
    )
    p.add_argument(
        "--text-threshold",
        type=float,
        metavar="T",
        help="GroundingDINO token->label threshold; lower keeps more words per label",
    )
    p.add_argument(
        "--bg-max-area",
        type=float,
        metavar="PCT",
        help="background model (sam/automask): a mask >= PCT%% of the image is background",
    )
    p.add_argument(
        "--fg-min-area",
        type=float,
        metavar="PCT",
        help="background model (sam/automask): drop masks below PCT%% of the image as noise",
    )
    p.add_argument(
        "--grid-size",
        type=int,
        metavar="N",
        help="background/MobileSAM automask grid N (N*N decoder calls); larger=more coverage",
    )
    p.add_argument(
        "--dilate",
        type=int,
        metavar="N",
        help="morph every output mask by |N| px (square kernel): >0 enlarge, <0 shrink",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="save an annotated image with an auto name <stem>.python.<model>.<task>.png",
    )
    p.add_argument(
        "--save-as",
        metavar="PATH",
        help="save the annotated image to this exact path (implies --save; extension picks the format)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="mask overlay opacity for --save (0..1, default: %(default)s)",
    )
    p.add_argument(
        "--max-grasps-per-object",
        type=int,
        default=3,
        metavar="N",
        help="for grasp results, draw at most N grasps per object (<=0 = all; default: %(default)s)",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="print the result JSON on a single line (default: pretty-printed)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the stderr summary line (stdout JSON is unaffected)",
    )
    p.set_defaults(func=cmd_predict)

    # ---- list / models ------------------------------------------------------
    pl = sub.add_parser(
        "list",
        parents=[common],
        aliases=["models", "ls"],
        help="list models in the registry (name, task, license, state)",
    )
    pl.add_argument("--json", action="store_true", help="print as JSON instead of a table")
    pl.set_defaults(func=cmd_list)

    # ---- ps -----------------------------------------------------------------
    pp = sub.add_parser("ps", parents=[common], help="list models currently loaded in server memory")
    pp.add_argument("--json", action="store_true", help="print as JSON instead of a table")
    pp.set_defaults(func=cmd_ps)

    # ---- load / unload ------------------------------------------------------
    pld = sub.add_parser("load", parents=[common], help="load a model into server memory")
    pld.add_argument("model", help="model name")
    pld.set_defaults(func=cmd_load)

    puld = sub.add_parser("unload", parents=[common], aliases=["rm"], help="unload a model from server memory")
    puld.add_argument("model", help="model name")
    puld.set_defaults(func=cmd_unload)

    # ---- health -------------------------------------------------------------
    ph = sub.add_parser("health", parents=[common], help="check that the server is reachable")
    ph.set_defaults(func=cmd_health)

    return parser


# --------------------------------------------------------------------------- #
# Command implementations
# --------------------------------------------------------------------------- #
def cmd_predict(client: Client, args: argparse.Namespace) -> int:
    boxes = _parse_boxes(args.box)
    points = _parse_points(args.point)

    # --- Inference: time ONLY the predict() round-trip (excludes draw + save). ---
    t0 = time.perf_counter()
    res = client.predict(
        args.model,
        args.image,
        prompt=args.prompt,
        box=boxes,
        point=points,
        roi=_parse_boxes(args.roi),
        dilate=args.dilate,
        method=args.method,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        bg_max_area=args.bg_max_area,
        fg_min_area=args.fg_min_area,
        grid_size=args.grid_size,
        min_size=args.min_size,
        max_size=args.max_size,
        gripper_min=args.gripper_min,
        gripper_max=args.gripper_max,
        max_grasps_per_object=args.max_grasps_per_object,
    )
    client_ms = (time.perf_counter() - t0) * 1000.0
    server_ms = res.duration_ms

    # For grasp tasks: select the single best target grasp for JSON output.
    target_grasp = None
    if res.grasps:
        from .postprocess import select_target_grasp as _select_target

        target_grasp = _select_target(res.grasps)
        if target_grasp is not None:
            import dataclasses
            res_out = dataclasses.replace(res, grasps=[target_grasp])
        else:
            res_out = res
    else:
        res_out = res

    # stdout: result JSON (wire-faithful, pipe-friendly).
    wire = _result_to_wire(res_out)
    if args.compact:
        sys.stdout.write(json.dumps(wire, separators=(",", ":")) + "\n")
    else:
        sys.stdout.write(json.dumps(wire, indent=2) + "\n")

    # Optional annotated image (cost NOT counted in the reported durations above).
    # Draw ALL filtered grasps (from res, not res_out), highlighting the target in red.
    saved_path: Optional[str] = None
    if args.save or args.save_as:
        out_path = args.save_as or _auto_name(args.image, args.model, res.task, "png")
        try:
            from .visualize import draw

            annotated = draw(
                res,
                args.image,
                alpha=args.alpha,
                max_grasps_per_object=None,  # already filtered by predict()
                target_grasp=target_grasp,
            )
            annotated.save(out_path)
            saved_path = out_path
        except ImportError as exc:
            print("warning: could not save image: %s" % exc, file=sys.stderr)

    if not args.quiet:
        counts = _summary_counts(res_out)
        device = res.device or "?"
        summary = "predict: model=%s task=%s device=%s  client=%.1fms server=%.1fms  %s" % (
            res.model or args.model,
            res.task or "?",
            device,
            client_ms,
            server_ms,
            counts,
        )
        print(summary, file=sys.stderr)
        if saved_path:
            print("saved: %s" % saved_path, file=sys.stderr)

    return 0


def _summary_counts(res: Result) -> str:
    parts: List[str] = []
    if res.detections:
        parts.append("%d detections" % len(res.detections))
    if res.masks:
        parts.append("%d masks" % len(res.masks))
    if res.grasps:
        parts.append("%d grasps" % len(res.grasps))
    if res.classifications:
        parts.append("%d classes" % len(res.classifications))
    if res.embeddings:
        parts.append("%d embeddings" % len(res.embeddings))
    if res.depth_map:
        parts.append("depth %dx%d" % (res.depth_width, res.depth_height))
    return "(" + ", ".join(parts) + ")" if parts else "(no objects)"


def cmd_list(client: Client, args: argparse.Namespace) -> int:
    models = client.list_models()
    if args.json:
        sys.stdout.write(json.dumps([_model_to_dict(m) for m in models], indent=2) + "\n")
    else:
        _print_models_table(models)
    return 0


def cmd_ps(client: Client, args: argparse.Namespace) -> int:
    models = client.ps()
    if args.json:
        sys.stdout.write(json.dumps([_model_to_dict(m) for m in models], indent=2) + "\n")
    else:
        if not models:
            print("no models loaded", file=sys.stderr)
        _print_models_table(models)
    return 0


def cmd_load(client: Client, args: argparse.Namespace) -> int:
    resp = client.load(args.model)
    sys.stdout.write(json.dumps(resp) + "\n")
    return 0


def cmd_unload(client: Client, args: argparse.Namespace) -> int:
    resp = client.unload(args.model)
    sys.stdout.write(json.dumps(resp) + "\n")
    return 0


def cmd_health(client: Client, args: argparse.Namespace) -> int:
    resp = client.health()
    sys.stdout.write(json.dumps(resp) + "\n")
    return 0


def _model_to_dict(m: ModelInfo) -> Dict[str, str]:
    return {"name": m.name, "task": m.task, "license": m.license, "state": m.state}


def _print_models_table(models: Sequence[ModelInfo]) -> None:
    if not models:
        return
    name_w = max([len("NAME")] + [len(m.name) for m in models])
    task_w = max([len("TASK")] + [len(m.task) for m in models])
    lic_w = max([len("LICENSE")] + [len(m.license) for m in models])
    header = "%-*s  %-*s  %-*s" % (name_w, "NAME", task_w, "TASK", lic_w, "LICENSE")
    print(header + "  STATE")
    for m in models:
        print(
            "%-*s  %-*s  %-*s  %s"
            % (name_w, m.name, task_w, m.task, lic_w, m.license, m.state)
        )


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    host = getattr(args, "host", DEFAULT_HOST)
    timeout = getattr(args, "timeout", 120.0)
    client = Client(host=host, timeout=timeout)
    try:
        return args.func(client, args)
    except VisionServeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
