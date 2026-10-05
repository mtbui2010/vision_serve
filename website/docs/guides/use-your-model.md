# Use a model you trained

## When you need this

You have a trained model: a checkpoint from your training run, a model on Hugging Face, or an
`.onnx` file someone exported. You want VisionServe to serve it like the built-in models, with
`visionserve run` and the HTTP API.

## The command

```console
$ visionserve convert rfdetr best.pth --name my-detector
```

`convert` turns the checkpoint into an ONNX file (the format ONNX Runtime runs), writes the
`manifest.yaml` that tells VisionServe how to use it, checks the result, and installs it under the
name you chose. Pick the line for your file:

| Your file | Command |
|---|---|
| RF-DETR training checkpoint (`.pth`, e.g. `checkpoint_best_total.pth`) | `visionserve convert rfdetr best.pth --name NAME` |
| A Hugging Face model: a folder, or a hub id such as `google/vit-base-patch16-224` | `visionserve convert hf google/vit-base-patch16-224 --name NAME` |
| TorchScript (`.pt` saved with `torch.jit.save`) | `visionserve convert torchscript model.pt --name NAME --task classification --input 224x224 --license MIT` |
| PyTorch `state_dict` plus the Python file that builds the model | `visionserve convert pytorch weights.pth --script build.py --name NAME --task detection --input 640x640 --license MIT` |
| Keras (`.keras` or `.h5`) | `visionserve convert keras model.keras --name NAME --task classification --input 224x224 --license Apache-2.0` |
| TensorFlow Lite (float `.tflite`) | `visionserve convert tflite model.tflite --name NAME --task classification --input 224x224 --license Apache-2.0` |
| TensorFlow SavedModel folder | `visionserve convert tensorflow saved_model/ --name NAME --task classification --input 224x224 --license Apache-2.0` |
| An `.onnx` file you already have | `visionserve import model.onnx --name NAME --task detection --license Apache-2.0` (see [below](#already-have-an-onnx-file-import-it)) |

The first two rows read everything from the checkpoint. The generic formats (TorchScript,
PyTorch, Keras, TFLite, TensorFlow) cannot know what the model does, so you say it:

- `--task`: `classification` (one score per class), `detection` (boxes from a DETR-style
  model, such as RF-DETR or RT-DETR), `depth` (a depth map) or `embed` (one vector per photo).
- `--input WxH`: the photo size the model was trained on, for example `224x224`.
- `--license`: the licence of the **original** model. Only Apache-2.0, MIT and BSD are accepted.
  AGPL models (Ultralytics YOLO, FastSAM, YOLO-World) are always refused, because they would put
  every user of VisionServe under the AGPL. "It is on Hugging Face" says nothing about the licence;
  read the model card.
- `--labels classes.txt`: the class names, one per line, in the model's output order. RF-DETR and
  Hugging Face checkpoints usually carry them; `convert` tells you when they do not.

`convert` and `check` run the converter, a Docker image with PyTorch and TensorFlow, so the
server itself stays a small Go binary. You need Docker on the machine where you convert.

## Reading the result

A real run on an RF-DETR Nano checkpoint. This checkpoint has no class names in it, so the run
passes `--labels`; `--images` gives the check real photos instead of a synthetic one (the
registry path is shortened to `…/models`):

```console
$ visionserve convert rfdetr best.pth --name my-detector --labels classes.txt --images ./photos
rfdetr: variant nano (from state_dict shapes (patch, dim, pe grid, decoder layers) = (16, 384, 24, 2)); trained with rfdetr unknown
rfdetr: 91 logits -> labels ['N/A', 'person', 'bicycle'] ... ['hair drier', 'toothbrush']
rfdetr: exporting nano at 384x384, opset 17 ...
...
installed my-detector (detection, Apache-2.0) -> …/models/my-detector  [3 files]
...
VisionServe convert report: my-detector (detection, rf-detr)
overall: PASS
  tier  check                                           status  result
  A     ONNX vs framework parity (synthetic + photo)    PASS    max|Δ|/scale 9.89e-04 over 3 run(s), tolerance 0.001; 100 low-score queries differ over the runs (DETR top-K near-ties), not judged one by one
  B1    preprocessing vs rfdetr predict() preprocessi~  PASS    mean|Δ| 0.38 gray levels (tensor 0.00666), max 3.1 levels, p99 1.5; 8 image(s)
  B2    outputs vs rfdetr RFDETR.predict                PASS    42/43 matched (97.7%), |Δconf| mean 0.00293 max 0.0447, box err mean 0.086 px max 4.14 px (ref 42 / served 44 dets)
full report: …/models/my-detector/convert-report.json
```

Read `overall:` first. `PASS` means the model is installed and checked. The first lines say what
`convert` read from the checkpoint: the variant (`nano`), the input size (384 × 384) and the 91
class names. The table is what it checked, each row against the original model in PyTorch:

| Row | In plain words |
|---|---|
| **A** | The ONNX file computes the same numbers as the PyTorch model, on the same input. |
| **B1** | The server prepares a photo (resize, colour scaling) like the training code: 0.38 gray levels apart on 8 photos, out of 255. Below 2 is a pass. |
| **B2** | The server finds the same objects as the original model: 42 of 43 boxes match, within 0.09 pixel on average. |
| **C** | Only with `--eval val.json` (your labelled photos): the accuracy of both, side by side. |

A running server picks the new model up the first time you ask for it, without a restart.

### Then run it

```console
$ visionserve run my-detector photo.jpg
{
  "task": "detection",
  "model": "my-detector",
  "device": "gpu:0",
  "detections": [
    {
      "bbox": [
        10.868959426879883,
        173.2867670059204,
        281.49389266967773,
        249.48060035705566
      ],
      "class": "laptop",
      "conf": 0.9175987493193849
    },
...
predict: model=my-detector task=detection device=gpu:0  client=7032.2ms server=997.6ms  (4 detections, 0 masks, 0 grasps)
```

`bbox` is `[x, y, width, height]` in the pixels of your photo. (A one-shot `run` loads the model
every time, so it is slow; a server keeps it loaded.) To check it against your labelled photos,
see [Check it behaves like training](check-training.md).

## Already have an ONNX file? Import it

`visionserve import` needs no Docker and no Python. It reads what it can from the file and says
what it had to guess:

```console
$ visionserve import exported.onnx --name my-import --task detection --license Apache-2.0 \
      --labels classes.txt
installed my-import (detection, Apache-2.0) -> …/models/my-import  [3 files]
  run it with:  visionserve run my-import <image>
WARN: imported my-import into …/models/my-import, but 3 settings were assumed, not read from the file — check them under "What import read and assumed"
...
What import read and assumed
  read:    layout NCHW: input "input" [1,3,384,384] has its 3 colour channels at axis 1
  read:    input size 384×384 (width×height) from the graph's fixed dims
  read:    outputs: boxes [1,300,4] and class scores [1,300,91] (300 queries, 91 classes)
  read:    labels: 91 names from classes.txt
  assumed: resize: squash the photo to 384×384 without keeping its aspect ratio (the usual training resize) — pass --resize letterbox if the model was trained on padded images
  assumed: normalisation: pixels/255, then ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225] (torchvision / timm default) — pass --mean and --std if the model was trained otherwise (e.g. 0.5,0.5,0.5)
  assumed: decoder: DETR-style, NMS-free (RF-DETR / RT-DETR): a sigmoid score per class and query, boxes as cx,cy,w,h normalised to 0..1 of the input, score threshold 0.5. A softmax DETR (facebook/detr, with a 'no object' class) or a YOLO export decodes WRONGLY with it — convert those with `visionserve convert`
...
```

`WARN` here is normal: an ONNX file does not say how its photos were resized or normalised
during training. Compare each `assumed:` line with your training code. If they match, you are
done. If not, run the command again with `--force` and the right `--resize`, `--mean` or `--std`.
Then [check it](check-training.md): `check` measures whether the guess was right.

## If it says WARN or FAIL

| Message | What to do |
|---|---|
| `warning: no class names in the checkpoint and no --labels; detections will be class_0..class_90` | Add `--labels classes.txt`: one name per line, in the order of the model's outputs. |
| `license "AGPL-3.0" is not allowed — only permissive licenses accepted` | Nothing to fix on your side: VisionServe cannot serve this model. Pick an Apache-2.0, MIT or BSD model. |
| `a --license is required` | Add `--license` with the original model's licence (read its model card or LICENSE file). |
| `FAIL: model "my-import" already exists at …/models/my-import (use --force to replace it)` | Use another `--name`, or add `--force` to replace it. |
| `overall: WARN` with `B2 ... top-1 agreement 87.5%` | One photo of eight changed its top class because two classes were almost tied (0.204 vs 0.196 on that photo). Usually harmless. Add `--eval` with labelled photos to measure the accuracy. |
| `overall: FAIL` on row **A** | The export is wrong: the ONNX file computes something else than the model. Send `convert-report.json` with an issue. |
| `overall: FAIL` on row **B1** | The server prepares photos differently from the training code. See [Check it behaves like training](check-training.md#if-it-says-warn-or-fail): the same table applies. |
| `overall: FAIL` on row **C** | The served model lost more than 1 point of accuracy. A FAIL uninstalls the model; add `--keep-on-fail` to keep it while you investigate. |
| import: `WARN ... settings were assumed` | Check each `assumed:` line against your training code (above). |
| import: `FAIL: import cannot build a "embed" model from the ONNX header alone` | `import` handles classification, DETR-style detection and depth. For anything else, use `convert`. |
| import: `FAIL: the first output "dets" is [1,300,4]; a classification model must output one row of class scores` | The `--task` does not match the file's outputs: this file is a detector. Fix `--task`. |

## Want the details?

??? note "What else convert can do"
    - `--eval val.json` (a COCO json, or a folder per class for a classifier) adds row **C**:
      accuracy of the original model and of the served one on your labelled photos.
    - `--precision fp16` (or `int8`, `mixed`) shrinks the model as it converts; see
      [Reduced precision](precision.md). For a Jetson,
      [`visionserve optimize`](jetson.md) picks the format for you.
    - `--dry-run` converts and checks, prints the manifest, and installs nothing.
    - The Hugging Face run on `google/vit-base-patch16-224` read its licence from the hub
      (`license=Apache-2.0`) and gave the `WARN` of the table above.

- How each check works, with real failures it caught:
  [Inspect and verify a model, section 3](inspect.md#3-let-the-converter-check-it-tiers-a-b1-b2-c).
- The plumbing (Docker image, how a model is installed):
  [Clients and the converter](../architecture/clients.md).
- Every field of `manifest.yaml`: [Manifest format](../reference/manifest.md).

<small>Run on 5 October 2026: RTX A6000 (CUDA), ONNX Runtime 1.26; `convert` with the converter's
Python package from `clients/python` (the code the Docker image runs), `import` and `run` with
the `visionserve` binary.</small>
