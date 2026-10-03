# What the models do

Every example on this page is a **real output** of VisionServe on a real photo. The pictures were
drawn from the JSON answers by
[`website/tools/figures.py`](https://github.com/mtbui2010/vision_serve/blob/main/website/tools/figures.py),
so you can regenerate them yourself.

All tasks return the same JSON shape (`api.Result` in
[`pkg/api/types.go`](https://github.com/mtbui2010/vision_serve/blob/main/pkg/api/types.go)):
only the fields that make sense for the task are filled in.

## Object detection — "what is here, and where?"

The model draws a box around every object it knows and names it. **Closed-set** detectors such
as RF-DETR know a fixed list of classes (the 80 everyday COCO classes: person, cup, chair…).
RF-DETR is a *transformer* detector: it predicts a fixed set of candidate objects directly, so,
unlike YOLO-style models, it needs no "non-maximum suppression" clean-up step.

<!-- FIGURE: detection -->

Models: `rf-detr`, `rf-detr-nano`, `rfdetr-small`, `rt-detr`. Answer: `detections[]` with
`class`, `conf`, `bbox`.

## Open-vocabulary detection — "find the things I name"

You type the names, separated by `" . "`, and the model finds them — even names that were never
a class during training. GroundingDINO reads the image and the words together and scores every
candidate box against every word.

<!-- FIGURE: openvocab -->

Models: `grounding-dino`, `owlvit`, and hybrids that combine a fast closed-set detector with
GroundingDINO for unknown words (`rfdetr-gdino`, see [Pipelines](../architecture/pipelines.md)).

## Segmentation — "the exact outline of this object"

A box is coarse; a **mask** says which pixels belong to the object. Segment-Anything models
(MobileSAM, SAM 2, EfficientSAM) cut out *whatever you point at*: give a box or a point as the
prompt.

<!-- FIGURE: segmentation -->

Without a prompt, MobileSAM tries a grid of points over the whole image and keeps the good,
non-overlapping masks — "segment everything":

<!-- FIGURE: automask -->

Answer: `masks[]` with `rle` (the mask, compressed), `bbox` and `conf`.

## Grounded-SAM — "outline everything I name"

Chain the two: GroundingDINO finds boxes for your words, then MobileSAM turns each box into a
mask. One request, text in, outlines out.

<!-- FIGURE: groundedsam -->

## Depth — "how far is each pixel?"

A depth model estimates, for every pixel, how far it is from the camera. The values are
*relative* (nearer vs farther), not metres.

<!-- FIGURE: depth -->

Models: `depth-anything-v2`, `midas`. Answer: `depth_map` (row-major, `depth_height ×
depth_width`).

## Faces and text

SCRFD finds faces; PaddleOCR finds lines of text and reads them.

<!-- FIGURE: faces -->

<!-- FIGURE: ocr -->

## Classification and embeddings — "what is this picture about?"

A classifier gives the top-5 labels for the *whole* image (1000 ImageNet classes).
CLIP and SigLIP instead turn an image — or a sentence — into an **embedding**: a vector of a few
hundred numbers. Pictures and sentences that mean the same thing get vectors that point the same
way, so you can compare a photo against any list of descriptions.

<!-- FIGURE: classification -->

<!-- FIGURE: clip -->

## Robot grasping and background

For robot arms, `grasp-rfdetr` and `grasp-gd` find objects, outline them, and compute where a
two-finger gripper could hold each one (`grasps[]`: centre, angle, opening width, quality). The
`background` model finds the supporting surface — the table or floor — so everything else can be
treated as objects.

<!-- FIGURE: grasp -->

<!-- FIGURE: background -->

## Choosing a model

| Need | Fast | More accurate / flexible |
|---|---|---|
| Everyday objects | `rf-detr-nano` | `rf-detr` |
| Any named object | `rfdetr-gdino` (hybrid) | `grounding-dino` |
| One object's outline | `mobile-sam` | `sam2` |
| Outlines from text | `grounded-sam` | — |
| Depth | `midas` | `depth-anything-v2` |

The full list, with licences, is printed by `visionserve list`.
