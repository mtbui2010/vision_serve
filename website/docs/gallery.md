# Model gallery

Real outputs of VisionServe on real photos. Each caption gives the model, the request, the
server-side time on one RTX A6000 GPU (CUDA execution provider), and the photo's source.
Regenerate them all with
[`website/tools/figures.py`](https://github.com/mtbui2010/vision_serve/blob/main/website/tools/figures.py);
the raw numbers are in [`figures.json`](assets/img/figures.json).

## Detection

<div class="grid2" markdown>

<figure markdown="span">
  ![RF-DETR object detection](assets/img/detect-rfdetr-372819.jpg){ loading=lazy }
  <figcaption><code>rf-detr</code> · 22 ms on gpu:0 · Photo: COCO val2017 #372819 (<a href="http://farm3.staticflickr.com/2046/2516944023_d00345997d_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![RF-DETR object detection](assets/img/detect-rfdetr-29596.jpg){ loading=lazy }
  <figcaption><code>rf-detr</code> · 22 ms on gpu:0 · Photo: COCO val2017 #29596 (<a href="http://farm2.staticflickr.com/1174/4724268948_f93c2cb404_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

</div>

<p style="text-align:center;font-size:.75rem;opacity:.8">RF-DETR object detection</p>

## Open-vocabulary detection

<div class="grid2" markdown>

<figure markdown="span">
  ![GroundingDINO with a text prompt — kiwi, fig and rice are not COCO classes](assets/img/openvocab-gdino-177015.jpg){ loading=lazy }
  <figcaption><code>grounding-dino</code> · prompt='cat. laptop. couch.' · 124 ms on gpu:0 · Photo: COCO val2017 #177015 (<a href="http://farm1.staticflickr.com/131/355302776_1d1215b7c1_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![GroundingDINO with a text prompt — kiwi, fig and rice are not COCO classes](assets/img/openvocab-gdino-389381.jpg){ loading=lazy }
  <figcaption><code>grounding-dino</code> · prompt='broccoli. carrot. kiwi fruit. fig. rice.' · 118 ms on gpu:0 · Photo: COCO val2017 #389381 (<a href="http://farm3.staticflickr.com/2544/4007091102_031486bd66_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

</div>

<p style="text-align:center;font-size:.75rem;opacity:.8">GroundingDINO with a text prompt — kiwi, fig and rice are not COCO classes</p>

## Segmentation

<div class="grid2" markdown>

<figure markdown="span">
  ![MobileSAM with a box prompt (left) and a point prompt (right)](assets/img/segment-sam-box-177015.jpg){ loading=lazy }
  <figcaption><code>mobile-sam</code> · box='310,175,310,195' · 47 ms on gpu:0 · Photo: COCO val2017 #177015 (<a href="http://farm1.staticflickr.com/131/355302776_1d1215b7c1_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![MobileSAM with a box prompt (left) and a point prompt (right)](assets/img/segment-sam-point-564133.jpg){ loading=lazy }
  <figcaption><code>mobile-sam</code> · point='470,195,1' · 59 ms on gpu:0 · Photo: COCO val2017 #564133 (<a href="http://farm9.staticflickr.com/8348/8197453784_a3be1b210e_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

</div>

<p style="text-align:center;font-size:.75rem;opacity:.8">MobileSAM with a box prompt (left) and a point prompt (right)</p>

<figure markdown="span">
  ![MobileSAM with no prompt: every mask it found, in random colours](assets/img/automask-sam-389381.jpg){ loading=lazy }
  <figcaption>MobileSAM with no prompt: every mask it found, in random colours<br/><code>mobile-sam</code> · 743 ms on gpu:0 · Photo: COCO val2017 #389381 (<a href="http://farm3.staticflickr.com/2544/4007091102_031486bd66_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

## Grounded-SAM

<figure markdown="span">
  ![Grounded-SAM: text → boxes → masks](assets/img/grounded-sam-372819.jpg){ loading=lazy }
  <figcaption>Grounded-SAM: text → boxes → masks<br/><code>grounded-sam</code> · prompt='dog. person. bench.' · 225 ms on gpu:0 · Photo: COCO val2017 #372819 (<a href="http://farm3.staticflickr.com/2046/2516944023_d00345997d_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

## Depth

<div class="grid2" markdown>

<figure markdown="span">
  ![MiDaS relative depth (bright = near)](assets/img/depth-midas-29596.jpg){ loading=lazy }
  <figcaption><code>midas</code> · 8 ms on gpu:0 · Photo: COCO val2017 #29596 (<a href="http://farm2.staticflickr.com/1174/4724268948_f93c2cb404_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![MiDaS relative depth (bright = near)](assets/img/depth-midas-372819.jpg){ loading=lazy }
  <figcaption><code>midas</code> · 8 ms on gpu:0 · Photo: COCO val2017 #372819 (<a href="http://farm3.staticflickr.com/2046/2516944023_d00345997d_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

</div>

<p style="text-align:center;font-size:.75rem;opacity:.8">MiDaS relative depth (bright = near)</p>

## Faces

<div class="grid2" markdown>

<figure markdown="span">
  ![SCRFD face detection](assets/img/faces-scrfd-263969.jpg){ loading=lazy }
  <figcaption><code>scrfd</code> · 10 ms on gpu:0 · Photo: COCO val2017 #263969 (<a href="http://farm8.staticflickr.com/7307/8735597371_053c077264_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![SCRFD face detection](assets/img/faces-scrfd-8021.jpg){ loading=lazy }
  <figcaption><code>scrfd</code> · 9 ms on gpu:0 · Photo: COCO val2017 #8021 (<a href="http://farm7.staticflickr.com/6117/6240308255_77782d0723_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

</div>

<p style="text-align:center;font-size:.75rem;opacity:.8">SCRFD face detection</p>

## Text (OCR)

<figure markdown="span">
  ![PaddleOCR: detected lines and the text read from each](assets/img/ocr-paddle-receipt.png){ loading=lazy }
  <figcaption>PaddleOCR: detected lines and the text read from each<br/><code>paddle-ocr</code> · 516 ms on gpu:0 · Synthetic image</figcaption>
</figure>

## Classification and CLIP

<figure markdown="span">
  ![Top-5 ImageNet classes from two classifiers](assets/img/classify-564133.jpg){ loading=lazy }
  <figcaption>Top-5 ImageNet classes from two classifiers<br/><code>efficientnet-b0 + mobilenet-v3</code> · efficientnet-b0 5 ms, mobilenet-v3 5 ms on gpu:0 · Photo: COCO val2017 #564133 (<a href="http://farm9.staticflickr.com/8348/8197453784_a3be1b210e_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![CLIP: one image compared with six sentences](assets/img/zeroshot-clip-372819.jpg){ loading=lazy }
  <figcaption>CLIP: one image compared with six sentences<br/><code>clip + clip-text</code> · clip 15 ms, clip-text 4 ms on gpu:0 · Photo: COCO val2017 #372819 (<a href="http://farm3.staticflickr.com/2046/2516944023_d00345997d_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

## Robotics: grasps and support surfaces

<figure markdown="span">
  ![grasp-rfdetr: objects, their masks, and the best grasps for a two-finger gripper](assets/img/grasp-rfdetr-389381.jpg){ loading=lazy }
  <figcaption>grasp-rfdetr: objects, their masks, and the best grasps for a two-finger gripper<br/><code>grasp-rfdetr</code> · gripper_min=25 · 126 ms on gpu:0 · Photo: COCO val2017 #389381 (<a href="http://farm3.staticflickr.com/2544/4007091102_031486bd66_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<div class="grid2" markdown>

<figure markdown="span">
  ![background (method=sam): the supporting surface, and what is left on it](assets/img/background-sam-363840.jpg){ loading=lazy }
  <figcaption><code>background</code> · method='sam' · 189 ms on gpu:0 · Photo: COCO val2017 #363840 (<a href="http://farm3.staticflickr.com/2797/4256603007_6fbbea22de_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![background (method=sam): the supporting surface, and what is left on it](assets/img/background-sam-29596.jpg){ loading=lazy }
  <figcaption><code>background</code> · method='sam' · 191 ms on gpu:0 · Photo: COCO val2017 #29596 (<a href="http://farm2.staticflickr.com/1174/4724268948_f93c2cb404_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

</div>

<p style="text-align:center;font-size:.75rem;opacity:.8">background (method=sam): the supporting surface, and what is left on it</p>

## Explaining a detection

<figure markdown="span">
  ![/api/explain: where RF-DETR looked for detection 0 (the cat) and detection 1 (the laptop)](assets/img/explain-rfdetr-177015.jpg){ loading=lazy }
  <figcaption>/api/explain: where RF-DETR looked for detection 0 (the cat) and detection 1 (the laptop)<br/><code>rfdetr-small</code> · format='numpy', detection_idx=[0, 1] · 23 ms on gpu:0 · Photo: COCO val2017 #177015 (<a href="http://farm1.staticflickr.com/131/355302776_1d1215b7c1_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

## Preprocessing

<figure markdown="span">
  ![One photo, prepared for four different models — real tensors from /api/preprocess](assets/img/preprocess-modes-177015.jpg){ loading=lazy }
  <figcaption>One photo, prepared for four different models — real tensors from /api/preprocess<br/><code>rf-detr + rf-detr-nano + clip + mobile-sam</code> · Photo: COCO val2017 #177015 (<a href="http://farm1.staticflickr.com/131/355302776_1d1215b7c1_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

<figure markdown="span">
  ![The same detection in the model's input space and in the original photo](assets/img/bbox-mapping-177015.jpg){ loading=lazy }
  <figcaption>The same detection in the model's input space and in the original photo<br/><code>rf-detr-nano</code> · 17 ms on gpu:0 · Photo: COCO val2017 #177015 (<a href="http://farm1.staticflickr.com/131/355302776_1d1215b7c1_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

---

Photos: COCO val2017, CC BY 2.0 — full list of authors' Flickr pages in
[the photo credits](assets/img/CREDITS.md). The OCR receipt is a synthetic image.
