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

<div class="grid2" markdown>

<figure markdown="span">
  ![RF-DETR object detection](../assets/img/detect-rfdetr-372819.jpg){ loading=lazy }
  <figcaption><code>rf-detr</code> · 22 ms on gpu:0 · Photo: COCO val2017 #372819 (<a href="http://farm3.staticflickr.com/2046/2516944023_d00345997d_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![RF-DETR object detection](../assets/img/detect-rfdetr-29596.jpg){ loading=lazy }
  <figcaption><code>rf-detr</code> · 22 ms on gpu:0 · Photo: COCO val2017 #29596 (<a href="http://farm2.staticflickr.com/1174/4724268948_f93c2cb404_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

</div>

<p style="text-align:center;font-size:.75rem;opacity:.8">RF-DETR object detection</p>

Models: `rf-detr`, `rf-detr-nano`, `rfdetr-small`, `rt-detr`. Answer: `detections[]` with
`class`, `conf`, `bbox`.

## Open-vocabulary detection — "find the things I name"

You type the names, separated by `" . "`, and the model finds them — even names that were never
a class during training. GroundingDINO reads the image and the words together and scores every
candidate box against every word.

<div class="grid2" markdown>

<figure markdown="span">
  ![GroundingDINO with a text prompt — kiwi, fig and rice are not COCO classes](../assets/img/openvocab-gdino-177015.jpg){ loading=lazy }
  <figcaption><code>grounding-dino</code> · prompt='cat. laptop. couch.' · 124 ms on gpu:0 · Photo: COCO val2017 #177015 (<a href="http://farm1.staticflickr.com/131/355302776_1d1215b7c1_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![GroundingDINO with a text prompt — kiwi, fig and rice are not COCO classes](../assets/img/openvocab-gdino-389381.jpg){ loading=lazy }
  <figcaption><code>grounding-dino</code> · prompt='broccoli. carrot. kiwi fruit. fig. rice.' · 118 ms on gpu:0 · Photo: COCO val2017 #389381 (<a href="http://farm3.staticflickr.com/2544/4007091102_031486bd66_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

</div>

<p style="text-align:center;font-size:.75rem;opacity:.8">GroundingDINO with a text prompt — kiwi, fig and rice are not COCO classes</p>

Models: `grounding-dino`, `owlvit`, and hybrids that combine a fast closed-set detector with
GroundingDINO for unknown words (`rfdetr-gdino`, see [Pipelines](../architecture/pipelines.md)).

## Segmentation — "the exact outline of this object"

A box is coarse; a **mask** says which pixels belong to the object. Segment-Anything models
(MobileSAM, SAM 2, EfficientSAM) cut out *whatever you point at*: give a box or a point as the
prompt.

<div class="grid2" markdown>

<figure markdown="span">
  ![MobileSAM with a box prompt (left) and a point prompt (right)](../assets/img/segment-sam-box-177015.jpg){ loading=lazy }
  <figcaption><code>mobile-sam</code> · box='310,175,310,195' · 47 ms on gpu:0 · Photo: COCO val2017 #177015 (<a href="http://farm1.staticflickr.com/131/355302776_1d1215b7c1_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![MobileSAM with a box prompt (left) and a point prompt (right)](../assets/img/segment-sam-point-564133.jpg){ loading=lazy }
  <figcaption><code>mobile-sam</code> · point='470,195,1' · 59 ms on gpu:0 · Photo: COCO val2017 #564133 (<a href="http://farm9.staticflickr.com/8348/8197453784_a3be1b210e_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

</div>

<p style="text-align:center;font-size:.75rem;opacity:.8">MobileSAM with a box prompt (left) and a point prompt (right)</p>

Without a prompt, MobileSAM tries a grid of points over the whole image and keeps the good,
non-overlapping masks — "segment everything":

<figure markdown="span">
  ![MobileSAM with no prompt: every mask it found, in random colours](../assets/img/automask-sam-389381.jpg){ loading=lazy }
  <figcaption>MobileSAM with no prompt: every mask it found, in random colours<br/><code>mobile-sam</code> · 743 ms on gpu:0 · Photo: COCO val2017 #389381 (<a href="http://farm3.staticflickr.com/2544/4007091102_031486bd66_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

Answer: `masks[]` with `rle` (the mask, compressed), `bbox` and `conf`.

## Grounded-SAM — "outline everything I name"

Chain the two: GroundingDINO finds boxes for your words, then MobileSAM turns each box into a
mask. One request, text in, outlines out.

<figure markdown="span">
  ![Grounded-SAM: text → boxes → masks](../assets/img/grounded-sam-372819.jpg){ loading=lazy }
  <figcaption>Grounded-SAM: text → boxes → masks<br/><code>grounded-sam</code> · prompt='dog. person. bench.' · 225 ms on gpu:0 · Photo: COCO val2017 #372819 (<a href="http://farm3.staticflickr.com/2046/2516944023_d00345997d_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

## Depth — "how far is each pixel?"

A depth model estimates, for every pixel, how far it is from the camera. The values are
*relative* (nearer vs farther), not metres.

<figure markdown="span">
  ![MiDaS relative depth (bright = near)](../assets/img/depth-midas-29596.jpg){ loading=lazy }
  <figcaption>MiDaS relative depth (bright = near)<br/><code>midas</code> · 8 ms on gpu:0 · Photo: COCO val2017 #29596 (<a href="http://farm2.staticflickr.com/1174/4724268948_f93c2cb404_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

Models: `depth-anything-v2`, `midas`. Answer: `depth_map` (row-major, `depth_height ×
depth_width`).

## Faces and text

SCRFD finds faces; PaddleOCR finds lines of text and reads them.

<figure markdown="span">
  ![SCRFD face detection](../assets/img/faces-scrfd-263969.jpg){ loading=lazy }
  <figcaption>SCRFD face detection<br/><code>scrfd</code> · 10 ms on gpu:0 · Photo: COCO val2017 #263969 (<a href="http://farm8.staticflickr.com/7307/8735597371_053c077264_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![PaddleOCR: detected lines and the text read from each](../assets/img/ocr-paddle-receipt.png){ loading=lazy }
  <figcaption>PaddleOCR: detected lines and the text read from each<br/><code>paddle-ocr</code> · 516 ms on gpu:0 · Synthetic image</figcaption>
</figure>

## Classification and embeddings — "what is this picture about?"

A classifier gives the top-5 labels for the *whole* image (1000 ImageNet classes).
CLIP and SigLIP instead turn an image — or a sentence — into an **embedding**: a vector of a few
hundred numbers. Pictures and sentences that mean the same thing get vectors that point the same
way, so you can compare a photo against any list of descriptions.

<figure markdown="span">
  ![Top-5 ImageNet classes from two classifiers](../assets/img/classify-564133.jpg){ loading=lazy }
  <figcaption>Top-5 ImageNet classes from two classifiers<br/><code>efficientnet-b0 + mobilenet-v3</code> · efficientnet-b0 5 ms, mobilenet-v3 5 ms on gpu:0 · Photo: COCO val2017 #564133 (<a href="http://farm9.staticflickr.com/8348/8197453784_a3be1b210e_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![CLIP: one image compared with six sentences](../assets/img/zeroshot-clip-372819.jpg){ loading=lazy }
  <figcaption>CLIP: one image compared with six sentences<br/><code>clip + clip-text</code> · clip 15 ms, clip-text 4 ms on gpu:0 · Photo: COCO val2017 #372819 (<a href="http://farm3.staticflickr.com/2046/2516944023_d00345997d_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

## Robot grasping and background

For robot arms, `grasp-rfdetr` and `grasp-gd` find objects, outline them, and compute where a
two-finger gripper could hold each one (`grasps[]`: centre, angle, opening width, quality). The
`background` model finds the supporting surface — the table or floor — so everything else can be
treated as objects.

<figure markdown="span">
  ![grasp-rfdetr: objects, their masks, and the best grasps for a two-finger gripper](../assets/img/grasp-rfdetr-389381.jpg){ loading=lazy }
  <figcaption>grasp-rfdetr: objects, their masks, and the best grasps for a two-finger gripper<br/><code>grasp-rfdetr</code> · gripper_min=25 · 126 ms on gpu:0 · Photo: COCO val2017 #389381 (<a href="http://farm3.staticflickr.com/2544/4007091102_031486bd66_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![background (method=sam): the supporting surface, and what is left on it](../assets/img/background-sam-363840.jpg){ loading=lazy }
  <figcaption>background (method=sam): the supporting surface, and what is left on it<br/><code>background</code> · method='sam' · 189 ms on gpu:0 · Photo: COCO val2017 #363840 (<a href="http://farm3.staticflickr.com/2797/4256603007_6fbbea22de_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

## Choosing a model

| Need | Fast | More accurate / flexible |
|---|---|---|
| Everyday objects | `rf-detr-nano` | `rf-detr` |
| Any named object | `rfdetr-gdino` (hybrid) | `grounding-dino` |
| One object's outline | `mobile-sam` | `sam2` |
| Outlines from text | `grounded-sam` | — |
| Depth | `midas` | `depth-anything-v2` |

The full list, with licences, is printed by `visionserve list`.
