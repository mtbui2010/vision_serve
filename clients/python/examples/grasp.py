"""Planar grasp detection: predict grasps, pick a target, and visualize.

Usage:
    # Make sure the server is up:  make serve
    #
    # Class-agnostic (whole-image automask -> grasps):
    python examples/grasp.py path/to/image.jpg
    #
    # Class-aware (rf-detr detector -> per-object grasps), restrict to one class:
    python examples/grasp.py path/to/image.jpg --model grasp-rfdetr --cls cup
    #
    # Constrain the gripper opening (pixels) and pick the grasp nearest a target pixel:
    python examples/grasp.py img.jpg --gripper-min 20 --gripper-max 120 --target 320 240

Target selection here uses the 2D image criteria (quality + nearest-to-target-point).
For a TRUE 3D camera->object distance you also need a depth map + camera intrinsics;
see the commented block at the bottom and visionserve.select_target_grasp(..., depth_result=,
intrinsics=, target_distance=).

NOTE: the grasp model must be available on the server (check `client.list_models()`).
"""

import argparse

from visionserve import Client, VisionServeError, draw, select_target_grasp


def main() -> None:
    ap = argparse.ArgumentParser(description="VisionServe planar grasp example")
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("--host", default="http://localhost:11435")
    ap.add_argument("--model", default="grasp", help="grasp model name (e.g. grasp, grasp-rfdetr)")
    ap.add_argument("--prompt", default=None, help='text prompt (only for a grounding-dino detector)')
    ap.add_argument("--cls", default=None, help="restrict the target grasp to this object class")
    ap.add_argument("--gripper-min", type=float, default=None, help="min jaw opening in pixels")
    ap.add_argument("--gripper-max", type=float, default=None, help="max jaw opening in pixels")
    ap.add_argument("--target", type=float, nargs=2, metavar=("X", "Y"),
                    default=None, help="target pixel; pick the grasp nearest it")
    ap.add_argument("--max-per-object", type=int, default=3,
                    help="draw at most N highest-quality grasps per object (0 = all)")
    ap.add_argument("--out", default="grasp_out.jpg", help="annotated image output path")
    args = ap.parse_args()

    client = Client(host=args.host)

    print("available models:")
    for m in client.list_models():
        print("  %-16s task=%-12s license=%-12s state=%s" % (m.name, m.task, m.license, m.state))

    try:
        client.load(args.model)
        result = client.predict(
            args.model,
            args.image,
            prompt=args.prompt,
            gripper_min=args.gripper_min,
            gripper_max=args.gripper_max,
        )
    except VisionServeError as e:
        print("server error: %s" % e)
        return

    print("task=%s model=%s device=%s duration_ms=%.1f"
          % (result.task, result.model, result.device or "?", result.duration_ms))
    print("detections: %d   masks: %d   grasps: %d"
          % (len(result.detections), len(result.masks), len(result.grasps)))
    for g in result.grasps[:10]:
        label = g.cls or "(class-agnostic)"
        print("  grasp %-14s q=%.3f  x=%.1f y=%.1f theta=%.3f width=%.1f"
              % (label, g.quality, g.x, g.y, g.theta, g.width))

    if not result.grasps:
        print("no grasps returned.")
        return

    # --- pick a target grasp (2D criteria) ---
    target_point = tuple(args.target) if args.target else None
    target = select_target_grasp(
        result.grasps,
        cls=args.cls,
        gripper_min=args.gripper_min,
        gripper_max=args.gripper_max,
        target_point=target_point,
        # composite: prefer quality, lightly biased toward the target point if given
        weights={"quality": 1.0, "near": 0.5} if target_point else None,
    )
    if target is None:
        print("no grasp matched the target criteria.")
    else:
        print("TARGET grasp: q=%.3f  x=%.1f y=%.1f theta=%.3f width=%.1f  class=%s"
              % (target.quality, target.x, target.y, target.theta, target.width, target.cls or "-"))

    # --- visualize (top-N grasps per object); save ---
    try:
        annotated = draw(result, args.image, max_grasps_per_object=args.max_per_object)
        annotated.save(args.out)
        print("saved annotated image -> %s" % args.out)
    except ImportError as e:
        print("(skipped visualization: %s)" % e)

    # --- TRUE 3D distance selection (optional) ---
    # If you have a depth model result + camera intrinsics, you can pick the grasp whose
    # real camera distance is closest to a desired target distance (e.g. 0.5 m):
    #
    #   from visionserve import CameraIntrinsics
    #   depth = client.predict("midas", args.image)            # a depth Result (metric depth!)
    #   K = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240)   # your camera's intrinsics
    #   target = select_target_grasp(
    #       result.grasps, depth_result=depth, intrinsics=K, target_distance=0.5,
    #   )


if __name__ == "__main__":
    main()
