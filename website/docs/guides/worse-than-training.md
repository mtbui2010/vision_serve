# My served model is worse than in training

## When you need this

The model loads and answers, but it misses objects, finds wrong ones, or scores lower than in
your training notebook. There is no error message. Go through the steps below in order: each one
rules out one cause, and the first ones are the most common.

## The commands, in order

| # | Question | Command | Look at |
|---|---|---|---|
| 1 | Is the model file what the manifest says, and does the input size fit? | `visionserve inspect NAME` | The first line: `PASS`. `Fits the graph: yes`. |
| 2 | What picture does the model get? | `visionserve inspect NAME --image photo.jpg` | `NAME-input.png`: same framing and colours as in training? |
| 3 | Is the photo prepared as in training? | `visionserve check NAME --images DIR` | The **Preprocessing (B1)** row. |
| 4 | Does it answer like the original model? | add `--checkpoint best.pth` | The **Outputs vs original (B2)** row. |
| 5 | How much accuracy is lost? | add `--labels val.json` | The **Accuracy on your labels (C)** row. |
| 6 | Am I measuring the same thing? | (no command) | Threshold and box format, below. |
| 7 | Which hardware ran it, at which precision? | `visionserve bench NAME` | The `device` line. |

## Reading the result

Steps 1 to 5 are covered with real outputs on their own pages:
[See what a model takes and returns](see-a-model.md) (1, 2) and
[Check it behaves like training](check-training.md) (3, 4, 5). One `check` command with all
flags does 3 to 5 at once. A real one that found a mistake:

```console
$ visionserve check my-detector-lb --images ./val200/images --labels ./val200/instances.json \
      --checkpoint best.pth
FAIL: my-detector-lb does not see photos the way it was trained: the server letterboxes (shrinks the photo and adds bars) while training stretches the whole photo to 384x384.
...
  Accuracy on your labels (C)  FAIL    On your 200 labelled photos the served model scores mAP 40.9,
                                       the original model 44.1 (-3.2 points: lower).
                                       Likely cause: the preprocessing difference found in B1.
...
```

One line in the manifest (`letterbox: true` instead of `false`) cost 3.2 points of accuracy, and
nothing else showed it: the model still found 31 of 36 boxes. That is why the photo preparation
comes first in this list.

### Step 6: are you measuring the same thing?

- **The confidence threshold.** The server drops boxes below the manifest's `conf_threshold`
  (0.5 for RF-DETR): that is what users want. A training script that computes mAP keeps almost
  every box (a threshold near 0.001), which gives a higher number. Compare like with like:
  `check` scores both models at the served threshold.
- **The box format.** Every `bbox` VisionServe returns is `[x, y, width, height]` in the pixels of
  the photo you sent. If your evaluation expects `[x1, y1, x2, y2]`, convert it first.
- **The photo.** If your client resizes, crops or re-encodes the photo before sending it,
  compare with the original photo. Recent SDKs shrink large photos before upload; see
  [Clients](../clients/index.md).
- **Text prompts** (open-vocabulary models): the server's tokens must be your tokenizer's.
  `Client.tokenize(model, text)` returns them.

### Step 7: which hardware ran it?

Every answer has a `device` field (`cpu`, `gpu:0`, `gpu:0+trt`), and `visionserve bench` prints
it. The same model gives almost the same numbers on CPU and GPU. TensorRT (`gpu:0+trt`) can be
lower: it measured 6.8 mAP lower on GroundingDINO, which is why it is off unless you ask for it.
A reduced-precision variant (`-fp16`, `-int8`, `-mixed`) also loses some accuracy; its
`manifest.yaml` header says how much it lost when it was made
([Make it smaller and faster for Jetson](jetson.md)).

## If it says WARN or FAIL

| What you see | What to do |
|---|---|
| Step 1 is a `FAIL` | Fix that first: [the table in See what a model takes and returns](see-a-model.md#if-it-says-warn-or-fail). |
| Step 2: the picture has bars, a crop, or swapped colours that training did not have | Change the manifest's resize or colour settings; step 3 tells you which line. |
| Steps 3 to 5: a `Likely cause` line | [The table in Check it behaves like training](check-training.md#if-it-says-warn-or-fail). |
| B1 and B2 pass, C still loses accuracy | The label file order, or the box format (`postprocess.box_format`) in the manifest. Compare both with your training code. |
| Everything passes, but your own numbers are still lower | Step 6: you are probably not measuring the same thing. |
| `device: cpu` where you expected a GPU | [Measure speed: If it says WARN](measure-speed.md#if-it-says-warn-or-fail). Accuracy is the same, only slower. |

## Want the details?

??? note "The same checklist by hand"
    Every step can be done without the commands: read the ONNX file in Netron, fetch the tensor
    from `/api/preprocess` and compare it with your training transform in numpy, decode the raw
    outputs yourself and compare with `/api/predict`. The deep dive shows each with code and
    real outputs, and ends with this checklist in its long form:
    [Inspect and verify a model, section 5](inspect.md#5-checklist-my-served-model-is-worse-than-in-training).

- What a heatmap of "where the model looked" tells you:
  [What the model looked at](inspect.md#what-the-model-looked-at).

<small>Run on 5 October 2026: see [Check it behaves like training](check-training.md) for the
setup of the run above.</small>
