"""Tiers B (end-to-end parity on real images) and the orchestration of B / C / speed.

  B1  preprocessing: the tensor the SERVER feeds the model (POST /api/preprocess) vs the
      REFERENCE preprocessing of the same image (reference.py), max/mean |Δ| in gray levels and,
      when large, a diagnosis (diagnose.py).
  B2  outputs: the SERVER's answer (POST /api/predict) vs the original framework model run
      through its own pre- and postprocessing on the same image (metrics.py).

Everything goes through a real server (serverctl.py): the point is to test what users get —
image decoding, the Go preprocess, ONNX Runtime, the Go postprocess and the HTTP schema.
"""
from __future__ import annotations

import dataclasses
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np

from .common import log, onnx_io, sample_image
from .metrics import (compare_classification, compare_detections, cosine, pearson, phrase_in_merged_label,
                      resize_map, same_label)
from .reference import (ManifestReference, OnnxForward, Reference, TorchModuleReference, UserReference,
                        load_reference_script, resolve_device)
from .report import (ERROR, FAIL, PASS, SKIP, WARN, Report, TierResult, fmt, grade_high, grade_low,
                     parse_thresholds, worst)
from .constants import IMAGE_EXT, OPEN_VOCAB_ARCHS, TEXT_ARCHS

DEFAULT_PHRASES = ["a photo of a cat", "remote control", "water bottle"]


# --------------------------------------------------------------------------------------------
# images
# --------------------------------------------------------------------------------------------

def list_images(d) -> List[Path]:
    d = Path(d)
    if not d.is_dir():
        raise ValueError(f"--images {d}: not a directory")
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXT and p.is_file())


def spread(items: list, n: int) -> list:
    """n items evenly spaced over the list (deterministic; more variety than the first n)."""
    if n <= 0 or len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


@dataclasses.dataclass
class Img:
    label: str
    pil: object
    src: object        # what is sent to the server: a Path (bytes verbatim) or a PIL image
    synthetic: bool = False


def load_images(paths) -> List[Img]:
    from PIL import Image
    out = []
    for p in paths:
        with Image.open(p) as im:
            pil = im.convert("RGB")
        out.append(Img(Path(p).name, pil, Path(p)))
    return out


def synthetic_image() -> Img:
    from PIL import Image
    return Img("synthetic 640x480", Image.fromarray(sample_image(640, 480, seed=7)),
               Image.fromarray(sample_image(640, 480, seed=7)), synthetic=True)


# --------------------------------------------------------------------------------------------
# the reference plan for one bundle
# --------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Plan:
    bundle: object
    input_name: Optional[str]     # the image tensor's ONNX input name (None for text towers)
    b1: Optional[Reference]
    b2: Optional[Reference]
    framework: Optional[Reference]
    onnx_path: str


def image_input_name(onnx_path) -> Optional[str]:
    ins, _ = onnx_io(onnx_path)
    for name, shp, et in ins:
        if len(shp) == 4 and et == 1:  # float32 NCHW
            return name
    return None


def build_plan(bundle, args, onnx_path, device: str) -> Plan:
    inp = image_input_name(onnx_path)
    ref = getattr(bundle, "reference", None)
    official = ref if (ref is not None and ref.kind == "official") else None
    module = ref if isinstance(ref, TorchModuleReference) else None
    if official is not None:
        official.to(device)
    if module is not None:
        module.to(device)
    forward = module.forward if module is not None else OnnxForward(onnx_path)
    fdesc = "the original PyTorch module" if module is not None else "the exported ONNX (CPU)"

    user = None
    script = getattr(args, "reference_script", None)
    pre_fn = getattr(args, "preprocess_fn", None)
    if script:
        s = load_reference_script(script)
        user = UserReference(bundle, inp, s["preprocess"], s["predict"], forward,
                             f"--reference-script {Path(script).name}"
                             + ("" if s["predict"] else f" preprocess + {fdesc}, decoded like the Go side"))
    elif pre_fn is not None:
        user = UserReference(bundle, inp, pre_fn, None, forward,
                             f"your preprocess callable + {fdesc}, decoded like the Go side")
    manifest = ManifestReference(bundle, inp, forward, fdesc) if inp else None

    b1 = user or official or manifest
    if user is not None and user._pred is not None:
        b2 = user
    elif official is not None:
        b2 = official
    else:
        b2 = user or manifest
    framework = official or module
    return Plan(bundle, inp, b1, b2, framework, str(onnx_path))


# --------------------------------------------------------------------------------------------
# B1
# --------------------------------------------------------------------------------------------

def tier_b1(plan: Plan, client, name: str, images: List[Img], th: dict, prompt=None) -> TierResult:
    from .diagnose import compare_tensors, diagnose
    b = plan.bundle
    ref = plan.b1
    title = "preprocessing vs " + (_short(ref, pre=True) if ref else "reference")
    if b.architecture in TEXT_ARCHS:
        return _b1_tokens(plan, client, name, th, prompt)
    if ref is None or plan.input_name is None:
        return TierResult("B1", "preprocessing", SKIP, "no image input / no reference preprocessing", model=name)
    per, int_issues = [], []
    diags = {}  # code -> [images, first message]
    shape_issue = None
    for im in images:
        res = client.preprocess(name, im.src, prompt=prompt)
        rin = ref.preprocess(im.pil, prompt)
        if not rin:
            return TierResult("B1", title, SKIP, "the reference has no preprocessing to compare", model=name)
        s_key = plan.input_name if plan.input_name in res.inputs else _only_image(res.inputs)
        r_key = plan.input_name if plan.input_name in rin else _only_image(rin)
        if s_key is None or r_key is None:
            raise ValueError(f"cannot find the image tensor: server inputs {sorted(res.inputs)}, "
                             f"reference {sorted(rin)}")
        cmp = compare_tensors(res.inputs[s_key], rin[r_key], b.std)
        cmp["image"] = im.label
        if not cmp["same_shape"]:
            shape_issue = f"server {cmp['shape_srv']} vs reference {cmp['shape_ref']}"
        if not cmp["same_shape"] or cmp["mean_levels"] > th["b1_mean_warn"]:
            codes, msgs = diagnose(res.inputs[s_key], rin[r_key], res.meta, b.mean, b.std, im.pil.size)
            cmp["diagnosis"] = codes
            cmp["diagnosis_text"] = msgs
            for c, m in zip(codes, msgs):
                diags.setdefault(c, [0, f"{m} (e.g. {im.label})"])[0] += 1
        per.append(cmp)
        for k, v in res.inputs.items():  # token ids [N, L]: exact (image-shaped masks follow the geometry)
            if v.dtype == np.int64 and v.ndim == 2 and k in rin:
                rv = np.asarray(rin[k])
                if rv.shape != v.shape or not np.array_equal(rv.astype(np.int64), v):
                    int_issues.append(f"{k}: server {list(v.shape)} vs reference {list(rv.shape)}"
                                      + ("" if rv.shape != v.shape else f", {int((rv != v).sum())} values differ"))
    same = [p for p in per if p["same_shape"]]
    metrics = {"images": len(per), "reference": ref.description, "reference_kind": ref.kind, "per_image": per}
    notes = []
    if same:
        ml = float(np.mean([p["mean_levels"] for p in same]))
        mx = float(np.max([p["max_levels"] for p in same]))
        p99 = float(np.max([p["p99_levels"] for p in same]))
        metrics.update(mean_levels=ml, max_levels=mx, p99_levels=p99,
                       mean_abs=float(np.mean([p["mean_abs"] for p in same])),
                       max_abs=float(np.max([p["max_abs"] for p in same])))
        status = grade_low(ml, th["b1_mean_warn"], th["b1_mean_fail"])
        summary = (f"mean|Δ| {ml:.2f} gray levels (tensor {fmt(metrics['mean_abs'])}), max {mx:.1f} levels, "
                   f"p99 {p99:.1f}; {len(same)} image(s)")
    else:
        status, summary = FAIL, f"shape mismatch: {shape_issue}"
    if shape_issue and same:
        status, summary = FAIL, summary + f"; shape mismatch on {len(per) - len(same)} image(s): {shape_issue}"
    if int_issues:
        status = worst(status, FAIL)
        summary += "; token inputs differ: " + "; ".join(sorted(set(int_issues))[:3])
    if images and images[0].synthetic:
        notes.append("B1 ran on a SYNTHETIC image (no --images / --eval given): it checks the arithmetic, not "
                     "real-photo decoding")
    notes += [f"[{k}/{len(per)} images] {m}" for k, m in sorted(diags.values(), key=lambda v: -v[0])]
    metrics["diagnosis"] = {c: v[0] for c, v in diags.items()}
    if status == FAIL and ref.kind == "official" and ref.known_divergence:
        status = WARN
        notes.append(f"documented divergence, not a bug: {ref.known_divergence} — B2/C measure its effect")
    if ref.kind == "manifest":
        notes.append("B1 reference = the manifest's declared preprocessing re-implemented in numpy/PIL: this checks "
                     "the Go implementation of the declared spec, NOT that the spec matches how the model was "
                     "trained (pass --reference-script, or preprocess= in the Python API, for that)")
    return TierResult("B1", title, status, summary, model=name, metrics=metrics, notes=notes)


def _short(ref, pre: bool = False) -> str:
    if pre and getattr(ref, "short_pre", ""):
        return ref.short_pre
    return getattr(ref, "short", "") or ref.description.split(" (")[0].split(" — ")[0]


def _only_image(d: dict) -> Optional[str]:
    keys = [k for k, v in d.items() if np.asarray(v).ndim == 4 and np.asarray(v).dtype != np.int64]
    return keys[0] if len(keys) == 1 else None


def _b1_tokens(plan, client, name, th, prompt) -> TierResult:
    ref = plan.b1 if plan.b1 is not None else plan.bundle.reference
    if ref is None:
        return TierResult("B1", "token ids", SKIP, "no reference tokenizer", model=name)
    phrases = [prompt] if prompt else DEFAULT_PHRASES
    bad = []
    for t in phrases:
        got = np.asarray(client.tokenize(name, t)).reshape(-1)
        want = np.asarray(ref.preprocess(None, t)["input_ids"]).reshape(-1)
        if got.shape != want.shape or not np.array_equal(got, want):
            n = int((got != want).sum()) if got.shape == want.shape else -1
            bad.append(f"{t!r}: " + (f"{n} ids differ" if n >= 0 else f"length {got.size} vs {want.size}"))
    st = FAIL if bad else PASS
    summ = (f"{len(phrases)} phrase(s), token ids identical" if not bad else "; ".join(bad))
    return TierResult("B1", f"token ids vs {_short(ref)}", st, summ, model=name,
                      metrics={"phrases": phrases, "mismatches": bad})


# --------------------------------------------------------------------------------------------
# B2
# --------------------------------------------------------------------------------------------

def tier_b2(plan: Plan, client, name: str, images: List[Img], th: dict, prompt=None) -> TierResult:
    b = plan.bundle
    ref = plan.b2
    if b.architecture in TEXT_ARCHS:
        return TierResult("B2", "outputs", SKIP, "text tower: B1 compared the token ids; text embeddings are not "
                          "served by /api/predict", model=name)
    if ref is None:
        return TierResult("B2", "outputs", SKIP, "no reference predictor", model=name)
    title = "outputs vs " + _short(ref)
    real = [im for im in images if not im.synthetic]
    if not real:
        return TierResult("B2", title, SKIP, "no real images: pass --images DIR (or --eval) — a synthetic image "
                          "gives no meaningful detections/classes", model=name)
    pairs = []
    device = ""
    for im in real:
        r = ref.predict(im.pil, prompt)
        if r is None:
            return TierResult("B2", title, SKIP, "the reference cannot predict", model=name)
        s = client.predict(name, im.src, prompt=prompt, max_grasps_per_object=None)
        device = s.device or device
        pairs.append((im, r, s))
    metrics = {"images": len(pairs), "reference": ref.description, "server_device": device}
    notes = []
    task = b.task
    if task in ("detection", "open_vocab"):
        conf = (b.postprocess or {}).get("conf_threshold")
        # GroundingDINO's official post-process labels a box with every prompt token above
        # text_threshold ("cup water bottle"); VisionServe gives each box its single best phrase.
        open_vocab = b.architecture in OPEN_VOCAB_ARCHS
        m = compare_detections([(r.detections or [], s.detections, im.pil.size) for im, r, s in pairs],
                               conf_threshold=conf, iou_thr=th["b2_iou"], boundary=th["b2_boundary"],
                               label_match=phrase_in_merged_label if open_vocab else same_label)
        metrics.update(m, conf_threshold=conf)
        if m["matched_frac"] is None:
            return TierResult("B2", title, WARN, f"vacuous: no detections on either side at conf_threshold {conf} "
                              f"on {len(pairs)} image(s) — use images that contain the model's classes",
                              model=name, metrics=metrics)
        st = worst(grade_high(m["matched_frac"], th["b2_match_warn"], th["b2_match_fail"]),
                   grade_low(m["mean_dconf"], th["b2_conf_warn"], th["b2_conf_fail"]),
                   grade_low(m["mean_box_rel"], th["b2_box_warn"], th["b2_box_fail"]))
        eff = max(m["matched"] + m["unmatched_ref"], m["matched"] + m["unmatched_srv"])
        summ = (f"{m['matched']}/{eff} matched ({100 * m['matched_frac']:.1f}%), |Δconf| mean {fmt(m['mean_dconf'])} "
                f"max {fmt(m['max_dconf'])}, box err mean {fmt(m['mean_box_px'])} px max {fmt(m['max_box_px'])} px "
                f"(ref {m['n_ref']} / served {m['n_srv']} dets)")
        if m["boundary_ref"] or m["boundary_srv"]:
            notes.append(f"{m['boundary_ref']} reference-only and {m['boundary_srv']} served-only detections scored "
                         f"within {th['b2_boundary']} of conf_threshold {conf}: threshold-boundary flips, not counted")
        if m.get("matched_via_merged_label"):
            notes.append(f"{m['matched_via_merged_label']} match(es) where the reference label merged several "
                         "prompt phrases (transformers' grounded post-process) and the served label is one of them")
        if m["class_confusions"]:
            notes.append("overlapping boxes with DIFFERENT classes (reference, served): "
                         + ", ".join(f"{a}->{c}" for a, c in m["class_confusions"]))
    elif task == "classification":
        m = compare_classification([(r.probs or {}, [(c.cls, c.conf) for c in s.classifications])
                                    for im, r, s in pairs])
        metrics.update(m)
        st = worst(grade_high(m["top1_agree"], th["b2_top1_warn"], th["b2_top1_fail"]),
                   grade_low(m["max_dprob"], th["b2_prob_warn"], th["b2_prob_fail"]))
        summ = (f"top-1 agreement {100 * (m['top1_agree'] or 0):.1f}% ({m['n']} images), |Δprob| max "
                f"{fmt(m['max_dprob'])} mean {fmt(m['mean_dprob'])}")
    elif task == "embed":
        cs = [cosine(np.asarray(s.embeddings[0]), r.embedding) for im, r, s in pairs if s.embeddings]
        metrics.update(min_cosine=min(cs) if cs else None, mean_cosine=float(np.mean(cs)) if cs else None)
        st = grade_high(metrics["min_cosine"], th["b2_cos_warn"], th["b2_cos_fail"])
        summ = f"cosine min {fmt(metrics['min_cosine'], 5)} mean {fmt(metrics['mean_cosine'], 5)} ({len(cs)} images)"
    elif task == "depth":
        rs = []
        for im, r, s in pairs:
            if not s.depth_map:
                continue
            sm = np.asarray(s.depth_map, np.float32).reshape(s.depth_height, s.depth_width)
            rs.append(pearson(sm, resize_map(r.depth, s.depth_width, s.depth_height)))
        metrics.update(min_pearson=min(rs) if rs else None, mean_pearson=float(np.mean(rs)) if rs else None)
        st = grade_high(metrics["min_pearson"], th["b2_depth_warn"], th["b2_depth_fail"])
        summ = f"Pearson r min {fmt(metrics['min_pearson'], 4)} mean {fmt(metrics['mean_pearson'], 4)} ({len(rs)} maps)"
        notes.append("depth maps compared at the served map size; Pearson is invariant to the Go min-max "
                     "normalisation")
    else:
        return TierResult("B2", title, SKIP, f"no B2 metric for task {task!r}", model=name)
    if ref.kind == "manifest":
        notes.append("B2 reference = the exported model on the manifest-declared preprocessing (no official "
                     "pipeline exists for this format)")
    return TierResult("B2", title, st, summ, model=name, metrics=metrics, notes=notes)


# --------------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------------

def _guard(tier: str, title: str, name: str, fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — a broken CHECK is reported, never hidden
        log(f"verify: {tier} raised {type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}")
        return TierResult(tier, title, ERROR, f"{type(e).__name__}: {e}"[:300], model=name)


def default_prompt(bundle, evalset=None, given=None):
    if given:
        return given
    if bundle.task != "open_vocab":
        return None
    if evalset is not None and evalset.kind == "detection" and evalset.categories:
        return ". ".join(list(evalset.categories.values())[:20]) + "."
    return "object."


def run_verification(bundles, args, report: Report, models_dir: Path, workdir: Path,
                     fresh_server: bool = False) -> Report:
    """Tiers B / C / speed for every installed bundle; appends TierResults to `report`.
    fresh_server: the reachable server had these names before this run replaced them — verify on
    a temporary server (serverctl.acquire(fresh=True))."""
    from . import serverctl
    th = parse_thresholds(getattr(args, "threshold", None))
    report.thresholds = th
    names = [b.name for b in bundles]
    skip_all = None
    if getattr(args, "dry_run", False):
        skip_all = "dry run: nothing installed, so no server can serve it"
    evalset = None
    if getattr(args, "eval", None):
        from .evaluate import load_eval
        try:
            evalset = load_eval(args.eval, getattr(args, "eval_images", None), getattr(args, "eval_max", 200))
            log(f"verify: eval set {evalset.source}: {evalset.kind}, {len(evalset.items)} images")
        except Exception as e:  # noqa: BLE001
            report.add(TierResult("C", "accuracy", ERROR, f"cannot load --eval {args.eval}: {e}"))
            evalset = None

    # images for tier B
    v = getattr(args, "verify", None)
    n = 8 if v is None else int(v)  # cli refuses < 1; 0 must not silently mean 8
    if getattr(args, "images", None):
        images = load_images(spread(list_images(args.images), n))
    elif evalset is not None:
        images = load_images(spread([p for p, _ in evalset.items], n))
    else:
        images = [synthetic_image()]
        log("verify: no --images / --eval: B1 runs on a synthetic image, B2 is skipped")
    if not images:
        images = [synthetic_image()]

    server = None
    if skip_all is None:
        server = serverctl.acquire(names, models_dir, url=getattr(args, "server", None),
                                   no_server=getattr(args, "no_server", False), workdir=workdir, log=log,
                                   fresh=fresh_server)
        if server is None:
            skip_all = "no usable VisionServe server (see the server: lines above)"
        else:
            report.server = server.describe()
    try:
        device = resolve_device(getattr(args, "device", "auto"))
        for b in bundles:
            installed = Path(models_dir) / b.name
            onnx_path = _installed_onnx(b, installed)
            if skip_all:
                for t in ("B1", "B2") + (("C",) if evalset is not None else ()) + \
                        (("speed",) if getattr(args, "bench", False) else ()):
                    report.add(TierResult(t, {"B1": "preprocessing", "B2": "outputs", "C": "accuracy",
                                              "speed": "speed"}[t], SKIP, skip_all, model=b.name))
                continue
            plan = _guard("B1", "reference", b.name, lambda: build_plan(b, args, onnx_path, device))
            if isinstance(plan, TierResult):
                report.add(plan)
                continue
            client = server.client
            load = _guard("B1", "load", b.name, lambda: client.load(b.name))
            if isinstance(load, TierResult):
                report.add(load)
                continue
            prompt = default_prompt(b, evalset, getattr(args, "prompt", None))
            log(f"verify: {b.name}: B1 on {len(images)} image(s) vs {plan.b1.description if plan.b1 else '-'}")
            report.add(_guard("B1", "preprocessing", b.name,
                              lambda: tier_b1(plan, client, b.name, images, th, prompt)))
            log(f"verify: {b.name}: B2 vs {plan.b2.description if plan.b2 else '-'}")
            report.add(_guard("B2", "outputs", b.name, lambda: tier_b2(plan, client, b.name, images, th, prompt)))
            if evalset is not None:
                from .evaluate import tier_c
                report.add(_guard("C", "accuracy", b.name,
                                  lambda: tier_c(plan, client, b, evalset, getattr(args, "max_map_drop", 1.0),
                                                 prompt)))
            if getattr(args, "bench", False):
                from .bench import tier_speed
                report.add(_guard("speed", "speed", b.name,
                                  lambda: tier_speed(plan, client, b, images[0], prompt,
                                                     getattr(args, "bench_warmup", 3),
                                                     getattr(args, "bench_iters", 50))))
    finally:
        if server is not None:
            server.close()
    return report


def _installed_onnx(bundle, installed_dir: Path) -> str:
    """The ONNX file the server actually loads (the installed copy), falling back to the staged one."""
    src = Path(bundle.onnx.get("model") or next(iter(bundle.onnx.values())))
    cand = installed_dir / src.name
    return str(cand if cand.is_file() else src)


def tier_a_results(records, names) -> List[TierResult]:
    """Tier A rows from the parity records collected while the family exported."""
    if not records:
        return [TierResult("A", "ONNX vs framework parity", SKIP, "the family recorded no parity run",
                           model=names[0] if names else "")]
    worst_err = max(r["max_rel_diff"] for r in records)
    whats = sorted({r["what"].strip() for r in records})
    return [TierResult("A", "ONNX vs framework parity (synthetic input)", PASS,
                       f"max|Δ|/scale {worst_err:.2e} over {len(records)} run(s), tolerance {records[0]['tol']:g}",
                       model=names[0] if len(names) == 1 else "",
                       metrics={"runs": records, "checks": whats})]
