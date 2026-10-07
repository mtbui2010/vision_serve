# From pixels to tensors

A model was trained on images prepared in one exact way. Feed it images prepared any other way
and it still answers — just worse, often silently. This page shows what "prepared" means and how
VisionServe keeps it exact.

## Step 1 — make the image the size the model expects

Most models take a fixed square input, such as 640 × 640. A photo is rarely that shape, so
there are several ways to fit it. VisionServe implements each as a **mode** of one shared
preprocessing library,
[`internal/vision/preprocess`](https://github.com/mtbui2010/visionserve/tree/main/internal/vision/preprocess).
Here is the same photo prepared for five different models — these are the real tensors, fetched
from the server's `/api/preprocess` endpoint and turned back into pictures:

<figure markdown="span">
  ![One photo, prepared for five models — real tensors from /api/preprocess](../assets/img/preprocess-modes-177015.jpg){ loading=lazy }
  <figcaption>One photo, prepared for five models — real tensors from /api/preprocess<br/><code>rf-detr + rf-detr-nano-letterbox + clip + depth-anything-v2 + mobile-sam</code> · the letterbox panel is an illustration: a scratch copy of rf-detr-nano set to letterbox (rf-detr-nano itself squashes, as RF-DETR is trained) · Photo: COCO val2017 #177015 (<a href="http://farm1.staticflickr.com/131/355302776_1d1215b7c1_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

| Mode (`resize:`) | What it does | Used by |
|---|---|---|
| `squash` | Stretch to W×H, ignoring the aspect ratio. | RF-DETR, GroundingDINO, SAM 2 |
| `letterbox` | Shrink to fit, keep the aspect ratio, pad the rest (centred). | DETR-family models that were trained that way (manifest option) |
| `center_crop` | Resize the short side, then cut out the centre; with `crop_pct` the short side is first resized a little larger (256 for a 224 crop at 0.875). | CLIP; the ImageNet classifiers (`crop_pct: 0.875`) |
| `keep_aspect` | Resize keeping the aspect ratio, sides rounded to a multiple (e.g. 14). | Depth-Anything |
| `long_side` / `long_side_pad` | Longest side to the target; `_pad` also pads the bottom/right. | MobileSAM / NanoSAM, PaddleOCR |
| `top_left_pad` | Fit and keep the aspect ratio, image at the top-left corner, pad the rest. | SCRFD (faces) |
| `none` | Use the image as it is. | — |

The mode comes from the model's [manifest](../reference/manifest.md):

```yaml
preprocess:
  resize: squash
  width: 560
  height: 560
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]
```

## Step 2 — turn pixels into the right numbers

A pixel is three integers 0–255 (red, green, blue). Most models want floats, centred and scaled
the way their training data was:

```
value = (pixel / 255 − mean) / std          for each channel
```

and arranged channel-first, `[1, 3, H, W]` ("NCHW"). The `mean`/`std` above are the ImageNet
statistics many models use.

## Step 3 — map the answer back to *your* photo

The model's boxes are in the coordinates of the prepared tensor. Preprocessing therefore also
returns a small record, `Meta`, saying how the photo was scaled and shifted:

```
input_x = original_x × ScaleX + PadX
input_y = original_y × ScaleY + PadY
```

Postprocessing applies the inverse, so every `bbox` VisionServe returns is in your photo's
pixels:

<figure markdown="span">
  ![The same detection in the model's input space and in the original photo](../assets/img/bbox-mapping-177015.jpg){ loading=lazy }
  <figcaption>The same detection in the model's input space and in the original photo<br/><code>rf-detr-nano</code> (squash: scale_x 0.6, scale_y 0.8, no padding) · 29 ms on gpu:0 · Photo: COCO val2017 #177015 (<a href="http://farm1.staticflickr.com/131/355302776_1d1215b7c1_z.jpg">Flickr</a>, CC BY 2.0)</figcaption>
</figure>

!!! code "Where in the code"
    - [`spec.go`](https://github.com/mtbui2010/visionserve/blob/main/internal/vision/preprocess/spec.go) — the `Spec` (mode, size, mean/std…) and the `Meta` record.
    - [`apply.go`](https://github.com/mtbui2010/visionserve/blob/main/internal/vision/preprocess/apply.go) — one function that runs every mode.
    - [`geometry.go`](https://github.com/mtbui2010/visionserve/blob/main/internal/vision/preprocess/geometry.go) — the size arithmetic of each mode, copied from each model's reference code.
    - [`internal/vision/geom`](https://github.com/mtbui2010/visionserve/tree/main/internal/vision/geom) — mapping boxes back (`Affine.BoxToOrig`).

!!! warning "Why so much care?"
    The most common bug in serving vision models is a preprocessing mismatch: a model trained
    on letterboxed images served with plain resizing loses accuracy with no error message. In this
    project one such mismatch cost RF-DETR about 2 mAP. That is why each model's preprocessing is
    pinned by tests against the reference implementation, and why the Python converter
    ([Clients and the converter](../architecture/clients.md)) shares the same rules, checked by
    sync tests that compare Go and Python outputs value by value. To check your own model, see
    [Inspect and verify a model](../guides/inspect.md#2-inspect-the-preprocessing).
