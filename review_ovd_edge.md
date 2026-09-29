# Review from `ovd-edge`: what VisionServe would need in order to measure and serve the
# strict-IoU result

> **Nothing in this repository was changed.** This file is added; no existing file was touched.
> The working tree currently holds substantial uncommitted work by someone else (16 modified
> files, 14 untracked paths at the time of writing), so every item below is written as a
> proposal with a location and a sketch, not as an applied patch.
>
> Written from `~/trung_workdir/ovd-edge`, whose paper claims are the reason these six items
> matter. Cross-references of the form "§N" are `ovd-edge/docs/FINDINGS.md`.

---

## 0. What the neighbouring project found, in one paragraph

A distilled head scores a frozen detector's existing object queries against a word supplied at
run time, so no vision-language model runs at inference. The finding is that the standard way of
carrying teacher supervision onto those queries — a threshold on IoU — makes the target constant
across every student query that clears the threshold against one teacher box. Non-maximum
suppression then resolves each cluster by score, and so discards a box that fits better than the
one it keeps. The reused-query proposal pool is measurably *tighter* than GroundingDINO's (median
best IoU 0.939 against 0.927) and *looser* after suppression (0.909 against 0.923).

Two consequences bear directly on this server:

1. **The effect is close to invisible at IoU 0.50 and plain at IoU 0.90.** Three independent
   perturbations moved AP@0.90 by 1.6 to 3.9 points while moving AP@0.50 by at most 0.8, which is
   inside AP@0.50's own between-seed spread.
2. **mAP at a loose threshold is confounded by how many boxes a branch emits.** That confound
   appeared four times in `ovd-edge`, and once it reversed a conclusion that had already been
   written down.

Items 1 and 2 below are those two sentences turned into code changes.

---

## Summary table

| # | what | where | why it matters | risk |
|---|---|---|---|---|
| 1 | AP is only reported at `[.5:.95]`; no strict-IoU number exists | `eval/accuracy/task_eval.py:162` | the whole effect lives at IoU 0.90 | low, additive |
| 2 | boxes-per-image is neither settable nor reported | `pkg/api/types.go:21,87` | the confound that reversed a conclusion | low, additive |
| 3 | arithmetic precision is invisible in the response, and the TRT engine cache is not keyed on it | `pkg/api/types.go:24`, `internal/engine/trt.go:69,94` | FP16 is already wired into the fastpath re-scorer | medium |
| 4 | one `duration_ms` for a three-stage pipeline | `pkg/api/types.go:34` | the cost claim is per-stage | low, additive |
| 5 | `Result.Grasps` must **not** be used for the hardware experiment | `pkg/api/types.go:58`, `docs/grasp-design.md` | it repairs bad boxes and hides the effect | none — a usage note |
| 6 | absolute `/home/trung/...` paths in an untracked manifest | `models/rfdetr-gdino-fastpath/manifest.yaml:40-43,65` | breaks on any other machine and in Docker | low, but fix before committing |

---

## 1. The eval reports no strict-IoU AP

`eval/accuracy/task_eval.py:162` returns one number:

```python
ev.evaluate(); ev.accumulate(); ev.summarize()
return {"task": "detection", "metric": "mAP@[.5:.95]", "value": float(ev.stats[0]), ...}
```

`ev.stats[0]` is mAP over the `0.50:0.05:0.95` grid. `ev.stats[1]` and `ev.stats[2]` would give
AP@0.50 and AP@0.75 for free, and neither is returned. **AP@0.90 is not in `ev.stats` at all** —
`COCOeval.summarize` does not compute it — so it has to be read out of the accumulated precision
array at the IoU index for 0.90:

```python
import numpy as np

def _ap_at(ev, iou):
    """AP at ONE IoU threshold, from the accumulated precision array.

    ev.eval['precision'] has shape [T, R, K, A, M]: IoU thresholds, recall points, categories,
    area ranges, max-dets. A == 0 is 'all' areas and M == -1 is the largest max-dets, which is
    what stats[0..2] use. -1 marks a recall point with no detections and must be dropped, not
    averaged in as a zero.
    """
    t = int(np.argmin(np.abs(ev.params.iouThrs - iou)))
    if abs(ev.params.iouThrs[t] - iou) > 1e-6:
        raise ValueError(f"IoU {iou} is not on the grid {ev.params.iouThrs}")
    p = ev.eval["precision"][t, :, :, 0, -1]
    p = p[p > -1]
    return float(np.mean(p)) if p.size else float("nan")

return {"task": "detection",
        "metrics": {"mAP@[.5:.95]": float(ev.stats[0]),
                    "AP@0.50": _ap_at(ev, 0.50),
                    "AP@0.75": _ap_at(ev, 0.75),
                    "AP@0.90": _ap_at(ev, 0.90)},
        "n": len(img_ids), "device_reported": device}
```

`_ap_at` above was checked against `pycocotools` on a synthetic one-image set before being written
here: it reproduces `ev.stats[1]` and `ev.stats[2]` to the digit, and returns a value at 0.90 where
`stats` has no entry. On that set a box at IoU 1.00 and a box at IoU 0.69 give AP@0.50 = 1.000 and
AP@0.90 = 0.505 — the loose box dropping out is the whole effect, in miniature.

Two things to get right, both of which `ovd-edge` got wrong first:

- **Keep the `metric` string attached to every value.** Six numerical errors in `ovd-edge` came
  from an AP@0.50 number being read against an mAP@[.5:.95] number. The worst claimed a crop
  budget of K=8 matched GroundingDINO at 64.76 against 64.85, when the correct baseline at that
  threshold was 72.57. A dict keyed by metric name makes that mistake require an explicit
  mislabelling rather than a glance.
- **Do not replace `stats[0]` with a single threshold.** Report all four. The point is not that
  AP@0.90 is the right metric; it is that the two disagree, and a reader has to see both.

This is the highest-value item in the file and the cheapest: it is additive, it changes no served
behaviour, and without it this repository cannot see the effect it is serving.

## 2. The crop budget can neither be fixed nor observed

`PredictRequest` (`pkg/api/types.go:87`) carries `MinSize` and `MaxSize` but no `top_k`. In the
textalign path the crop budget is set *implicitly*, by `postprocess.conf_threshold` on the
detector's own objectness — `models/rfdetr-dualhead-dec1/manifest.yaml` documents this at length
and notes that the count lands somewhere between 3 and 15 depending on the scene.

That makes the number of boxes a scene-dependent free variable. In `ovd-edge` the effect of that
is measured and large: AP@0.50 rises with the box count regardless of whether the boxes got any
better, so two configurations compared at different budgets are not comparable at all. Holding
the budget fixed reversed which of AP@0.50 and AP@0.90 correlated better with downstream grasp
feasibility (0.99/0.96–0.98 pooled, against 0.791/0.924 at a fixed budget and the tightest
tolerance).

Two additions, both small:

```go
// pkg/api/types.go, PredictRequest
TopK int `json:"top_k,omitempty"` // cap boxes entering the crop/re-score stage; 0 = manifest default

// pkg/api/types.go, Result
Counts map[string]int `json:"counts,omitempty"` // e.g. {"queries":300,"selected":22,"cropped":22,"after_nms":7}
```

`TopK` is what makes an equal-budget comparison expressible through the API. `Counts` is what
lets a reader detect the confound in results somebody else produced — which matters more, because
the numbers already published through this server were produced without it.

The `ovd-edge` branch uses top-K on the *head score* with K = 22, and its measurement is that K
sets nearly the whole latency (the crop tower is 87% of the branch's graph cost on an AGX Orin)
while K = 22 is already past saturation for AP: going to 44 costs 1.77× the served latency and
returns +0.8 AP@0.50 and 0.0 AP@0.90.

## 3. Precision is invisible, and the engine cache does not know about it

`DeviceString` (`internal/engine/provider.go:39`) reports the execution provider — `gpu:0+trt`,
`gpu:0`, `cpu` — and nothing reports the arithmetic. Two facts make that consequential here
rather than cosmetic:

**(a) FP16 is already in the recommended fast path.**
`models/rfdetr-gdino-fastpath/manifest.yaml` wires `siglip-image-fp16/model.onnx` and
`siglip-text-fp16/model.onnx`. The crop tower is the re-scorer, so its arithmetic decides the
final ranking, and a response served from it is indistinguishable from an FP32 one in the JSON.
Note carefully what is and is not known: `ovd-edge` measured FP16 on the **frozen detector** and
found it costs 1.6–2.1 AP@0.90 while costing under one point of AP@0.50. It did **not** measure
FP16 on the **crop tower**, which is the stage this server actually runs in FP16. So the right
statement is not "this is costing you 2 points"; it is that a stage was moved to FP16 on the
evidence of a loose-threshold metric that has been shown not to see this class of change, and the
strict-threshold cost of that specific decision is unmeasured. Item 1 is what would measure it.

**(b) The TRT engine cache key ignores the provider options.**
`engineKey` (`internal/engine/trt.go:94`) keys the cache on the resolved weights path plus size
and mtime, which is a good answer to the question it was written for (a re-export rewriting a file
in place). It does not include anything from `TRTOptions`, and ONNX Runtime's TensorRT EP also
reads `ORT_TENSORRT_FP16_ENABLE` and friends from the environment independently of the options
this code sets. So if that variable is ever exported — in a systemd unit, a Dockerfile, a
`.bashrc` on a Jetson — an FP16 engine gets built and cached under a key that says nothing about
precision, and afterwards a process with a clean environment reuses it. The response still says
`gpu:0+trt`.

Proposed:

```go
// pkg/api/types.go, Result
Precision string `json:"precision,omitempty"` // "fp32" | "fp16" | "mixed" — arithmetic, not the EP
```

and fold the effective provider options into the cache key, so that changing precision builds a
different engine instead of silently reusing one:

```go
// internal/engine/trt.go
func engineKey(modelPath string, opts map[string]string) string {
    // ... existing path/size/mtime work, then hash the sorted option pairs into the key, plus
    // the ORT_TENSORRT_* variables actually present in the environment. An engine is identified
    // by (weights, arithmetic, build flags); keying on weights alone is the bug.
}
```

`"mixed"` is in the enum deliberately: the fastpath's detector and its SigLIP towers need not
agree, and collapsing that to a single label would be the same kind of lie the field is meant to
prevent. If per-stage reporting (item 4) lands, the honest shape is one precision per stage.

## 4. One `duration_ms` for a three-stage pipeline

`Result.DurationMs` (`pkg/api/types.go:34`) is a single float. For the router and the fastpath the
interesting quantity is the split. Measured on an AGX Orin in `ovd-edge`: detector forward 5.5 ms,
SigLIP crop tower at K=22 39.2 ms (87% of the branch's graph cost), CPU crop extraction 27.3 ms
(61% of served latency at 640×480). That last number is the actionable one and it is not on the
GPU at all — it would be invisible in any profile taken inside the ONNX Runtime session.

```go
// pkg/api/types.go, Result
Stages map[string]float64 `json:"stages,omitempty"` // ms per stage, e.g. {"detect":5.5,"crop_extract":27.3,"crop_embed":39.2,"nms":0.4}
```

Additive, and it is the difference between "the open branch costs 80 ms" and "61% of the open
branch is `image.Crop` on the CPU and can be moved".

## 5. Do not use `Result.Grasps` for the hardware experiment

Not a defect — a warning, written here because this is the obvious field to reach for and
reaching for it would invalidate the experiment.

`pkg/api/types.go:58` defines `Grasp{X, Y, Theta, Width, Quality}`, and `docs/grasp-design.md`
specifies it as an analytic planar planner over the box or mask. The `ovd-edge` hardware protocol
requires the map from box to grasp to be a **deliberately dumb, pure function of the box, byte
identical across conditions**, precisely so that a bad box produces a bad grasp. An analytic
planner that re-centres on the mask, snaps θ to a principal axis, or scores quality will repair a
bad box, and the experiment is built to measure bad boxes.

Two further specifics for the KETI k-care arm the experiment runs on:

- Its command set exposes `grip::close` with **no width argument**, so `Grasp.Width` has nowhere
  to go. The rig can therefore test only the *centring* half of the grasp proxy, not the
  *aperture* half. The analysis has to be compared against the centring-only variant of the proxy
  or it is comparing two different criteria.
- `movet::x=,y=,z=` is translation only, so wrist orientation is fixed by the preceding
  `movej::approach_lying`. `Grasp.Theta` cannot be commanded, which means object placement has to
  be orientation-controlled and that fact has to be stated in the paper.

What the experiment should call is plain `/api/predict` with `prompt`, and use `bbox` — which is
already in original-image `[x, y, w, h]`, so no letterbox inversion is needed on the robot. That
part of the API needs nothing.

## 6. Absolute paths in an untracked manifest

`models/rfdetr-gdino-fastpath/manifest.yaml` hardcodes
`/home/trung/trung_workdir/vision_serve/...` for all four `files:` entries and for `labels:`. It
is the only manifest in `models/` that does, and it is currently untracked, so this is a note for
whoever is about to commit it rather than a report of a shipped bug. It will not resolve on
another machine, in CI, or inside the Docker image. Every other manifest uses paths relative to
its own directory, including `../siglip-text/model.onnx` in the dualhead manifest, so the
convention to follow already exists.

---

## Suggested order

1. **Item 1** first and alone. It is additive, it cannot change served behaviour, and until it
   lands this repository has no way to observe the difference the other items are about.
2. **Item 2**, because every comparison made before it exists is confounded and cannot be
   repaired afterwards from the stored results.
3. **Item 4**, additive, and it answers a question someone will ask immediately (the 61% CPU
   crop cost).
4. **Item 3**, last of the code changes, because it touches the engine cache and a wrong cache
   key is worse than no change. Note that invalidating the key will force a one-time rebuild of
   every cached engine.
5. **Items 5 and 6** need no code: a paragraph in `docs/grasp-design.md` for the first and a path
   fix before commit for the second.

## What this review does not cover

Read for these six questions only, and with the models not loaded and no server running — no
`visionserve` process was started and ports 11435 / 11530 / 11531 were left alone. In particular
this file says nothing about the correctness of the uncommitted work in
`internal/models/hybrid/` (`fastpath.go`, `rescore.go`) or `internal/models/siglip/`
(`tokenizer.go`, `text.go`), which were read only far enough to locate the crop budget and the
precision of the towers. The dualhead manifest records that a SigLIP tokenizer padding bug once
made every text embedding "plausible and wrong" and moved a headline number by 69 points; that is
the neighbourhood those files are in, and it deserves its own pass by whoever is writing them.
