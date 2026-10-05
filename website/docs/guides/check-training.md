# Check it behaves like training

## When you need this

Your model is installed and answers, but you want proof that it gives the answers it gave in
training. Do this after you import a model, after you edit a `manifest.yaml` by hand, or when
the served model seems worse than your training notebook. A served model can be wrong without
any error message: the most common cause is that the photo is prepared differently.

## The command

Start a server in another terminal (`visionserve serve`), then:

```console
$ visionserve check my-detector --images ./val200/images --labels ./val200/instances.json \
      --checkpoint best.pth
```

Only `--images` is required: a folder of real photos like the ones the model will see. The other
two each add a check:

- `--labels`: your labelled photos, so `check` can measure accuracy. A COCO json for a
  detector; a folder per class, or a CSV `image,label`, for a classifier.
- `--checkpoint`: the checkpoint the model was converted from, so `check` can compare the served
  answers with the original model's.

## Reading the result

A real run on 200 COCO photos with their labels:

```console
$ visionserve check my-detector --images ./val200/images --labels ./val200/instances.json \
      --checkpoint best.pth
PASS: my-detector behaves like its training pipeline on 8 photos (preprocessing within 0.4 gray levels).

Summary: my-detector (detection, rf-detr) on http://127.0.0.1:11770, 8 photo(s) from val200/images
  check                        status  what we found
  Preprocessing (B1)           PASS    The model sees the same picture as in training: average
                                       difference 0.4 gray levels (out of 255) over 8 photos — fine.
                                       Reference: the original framework's pipeline.
  Outputs vs original (B2)     PASS    The served model finds the same objects as the original
                                       model: 38 of 38 boxes match, confidence differs by 0.003 on
                                       average and boxes by 0.1 px.
  Accuracy on your labels (C)  PASS    On your 200 labelled photos the served model scores mAP 43.8,
                                       the original model 44.1 (-0.3 points: the same within noise).

Next steps
  1. Nothing to fix.
...
```

`PASS`: nothing to fix. The three rows, in plain words:

- **Preprocessing (B1)**: the picture the server gives the model, compared pixel by pixel with
  the picture the training code makes from the same photo. 0.4 gray levels (out of 255) is
  rounding. Above 8 costs accuracy.
- **Outputs vs original (B2)**: the boxes the server returns, compared with the original model's
  on the same photos. 38 of 38 match.
- **Accuracy on your labels (C)**: *mAP* (mean average precision, the standard 0 to 100 score
  of a detector: higher is better) of both models on your labelled photos. A difference under
  0.5 point is noise; more than 1 point lost is a `FAIL`.

Without `--checkpoint`, the B2 row says `SKIP` and C gives the served score alone
(`mAP 43.8 (mAP50 55.7)`), which is still worth keeping as a baseline.

### What a FAIL looks like

The same model with one wrong line in its manifest, `letterbox: true` (shrink the photo and add
bars) where RF-DETR was trained on stretched photos:

```console
$ visionserve check my-detector-lb --images ./val200/images --labels ./val200/instances.json \
      --checkpoint best.pth
FAIL: my-detector-lb does not see photos the way it was trained: the server letterboxes (shrinks the photo and adds bars) while training stretches the whole photo to 384x384.

Summary: my-detector-lb (detection, rf-detr) on http://127.0.0.1:11770, 8 photo(s) from val200/images
  check                        status  what we found
  Preprocessing (B1)           FAIL    The model sees a different picture than in training: average
                                       difference 46.6 gray levels (out of 255) over 8 photos, where
                                       more than 8 costs accuracy. Reference: the original
                                       framework's pipeline.
                                       Likely cause: the server letterboxes (shrinks the photo and
                                         adds bars) while training stretches the whole photo to
                                         384x384.
                                       Fix: in …/models/my-detector-lb/manifest.yaml set
                                         `input.letterbox: false` (unless the model really was trained
                                         letterboxed).
  Outputs vs original (B2)     WARN    The served model finds mostly the same objects as the
                                       original model: 31 of 36 boxes match, confidence differs by
                                       0.053 on average and boxes by 1.5 px.
                                       Likely cause: the preprocessing difference found in B1.
                                       Fix: fix B1 first, then re-run.
  Accuracy on your labels (C)  FAIL    On your 200 labelled photos the served model scores mAP 40.9,
                                       the original model 44.1 (-3.2 points: lower).
                                       Likely cause: the preprocessing difference found in B1.
                                       Fix: fix B1 first, then re-run.

Next steps
  1. In …/models/my-detector-lb/manifest.yaml set `input.letterbox: false` (unless the model really was trained letterboxed).
  2. Restart `visionserve serve` (it reads a manifest once), then re-run this check.
...
```

The model still answers, and most boxes still match, but it lost 3.2 points of accuracy. `check`
names the cause and the line to change. Fix the first `FAIL` row first: the later rows often
follow from it.

`--report check.html` writes the same verdict as one page, with the two pictures side by side and
the boxes of both models drawn on the photos where they disagree most. The exit status is `0` for
PASS or WARN, `1` for FAIL and `2` when it cannot run (no server, model not installed). A check you
asked for by flag that could not run (an exception, out of GPU memory, a missing package) also
exits `2`, with the verdict `ERROR` naming the check and its error: `--reference` or
`--checkpoint` ask for the preprocessing and outputs checks, `--labels` for accuracy. The checks
that did run are still printed, so a script never reads an incomplete check as passed.

## If it says WARN or FAIL

Each `Likely cause` line, and what to do. "Training" means your training code (or the original
framework's pipeline).

| Likely cause | What to do |
|---|---|
| the server letterboxes (shrinks the photo and adds bars) while training stretches the whole photo | Set `input.letterbox: false` in the manifest. |
| ... letterboxes (keeps the aspect ratio and pads) while the server stretches the photo | Set `input.letterbox: true` (or `preprocess.resize: letterbox`). |
| ... uses only the centre of the photo (a centre crop) while the server keeps all of it | Set `input.crop: center` (or `preprocess.resize: center_crop`). |
| ... reads colour channels in BGR order (OpenCV's cv2.imread) while the server feeds RGB | The server always sends RGB. Convert to RGB in training (`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`) and retrain, or fix your reference script if only it reads BGR. |
| the colours are normalised differently: ... looks like ... mean/std | Set the mean and std the message prints in the manifest (`input.normalize.mean` / `std`). |
| ... uses 0-255 pixel values while the server divides by 255 | The message gives the mean/std values that keep 0–255 pixels. |
| the server builds a [...] input but the reference builds [...] | Set the input size in the manifest to the training size. |
| the reference is mirrored left-right compared with the server | Your reference script still has a random flip on. Turn it off. |
| small resize-filter or JPEG-decoder differences | Usually nothing to fix (a WARN, not a FAIL). |
| outputs differ although the photo is prepared the same way | Check the label file order and the box format in the manifest; re-export if the file's `sha256` is not your export's. |
| the served model loses mAP although B1 found no preprocessing difference | Follow the next steps it prints: with fewer than 200 labelled photos re-run with more first (a small difference is noise there); then check the class mapping (the label file in training order, with your dataset's names) and the confidence threshold; without `--checkpoint`, re-run with it to compare the outputs photo by photo. |
| a near-zero score usually means the label names do not match the model's classes | Make the class names in the model's label file match your dataset's category names (case, spaces and `_` are ignored). |
| the manifest and the model file disagree (size, layout or files) | Fix the manifest as the error says (run [`visionserve inspect`](see-a-model.md)), then restart the server. |
| exit status 2: no server | Start one: `visionserve serve`, or pass `--server URL`. |
| `ERROR`, exit status 2: a check you asked for could not run | Fix the error it names (out of GPU memory: `--device cpu` or a free GPU), then re-run. |

After changing a manifest, restart `visionserve serve`: it reads a manifest once.

## Want the details?

??? note "What check compares with"
    For B1, `check` needs a *reference*: how training prepared photos. It takes, strongest
    first: your own script (`--reference prep.py`, a `preprocess(pil_image)` function), the
    `--checkpoint`'s own pipeline, the architecture's known recipe (RF-DETR: squash and
    ImageNet mean/std, what the `rfdetr` package does), or else what the manifest declares. In
    the last case it only checks the server's code, and the verdict says so.

    It uses up to 8 photos for B1 and B2 (`--max-images`) and up to 200 for C (`--labels-max`).
    Use 200 or more labelled photos: with 50, the *correct* model measured -1.45 mAP against its
    own reference, which is pure sampling noise.

- Every check with real runs, including three deliberate mistakes and how each was named:
  [Inspect and verify a model, section 3](inspect.md#3-let-the-converter-check-it-tiers-a-b1-b2-c).
- Doing the same comparison by hand in Python:
  [section 2](inspect.md#2-inspect-the-preprocessing) and
  [section 4](inspect.md#4-inspect-the-postprocessing).
- The resize modes, with pictures: [From pixels to tensors](../concepts/preprocessing.md).

<small>Run on 5 October 2026: a `visionserve serve` on an RTX A6000 (CUDA) at port 11770 serving
a scratch registry (shown as `…/models`), and `check` with the converter's Python package from
`clients/python`; `best.pth` is the official `rf-detr-nano.pth`; photos and labels are the first
200 images of COCO val2017.</small>
