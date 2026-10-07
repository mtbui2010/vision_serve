# HTTP API

The server listens on `http://127.0.0.1:11435` by default. Requests are `multipart/form-data`
(upload a file) or JSON (image as base64). Answers are JSON.

!!! warning "No authentication"
    The API has no login. Keep the default loopback address, or put VisionServe behind your own
    reverse proxy / firewall when you listen on `:11435`.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/api/health` | `{"status":"ok"}` when the server is up. |
| `GET` | `/api/models` | Every model: `name`, `task`, `license`, `state` (`not_downloaded`, `available`, `loaded`), and the client-resize hint `max_useful_side` / `max_useful_short_side` ([below](#get-apimodels)). |
| `POST` | `/api/load` | `{"model":"rf-detr"}` — load now instead of on first use. |
| `POST` | `/api/unload` | `{"model":"rf-detr"}` — free its memory. |
| `POST` | `/api/predict` | Run a model on an image. **The main endpoint.** |
| `POST` | `/api/preprocess` | Return the exact input tensor a model would receive, without running it (debugging). |
| `POST` | `/api/infer_tensor` | Run a model on a tensor you prepared yourself (raw float32 body). It knows nothing of the original photo: boxes come back in the tensor's pixels. |
| `POST` | `/api/explain` | A heatmap of what the model looked at for one detection (attention map or Score-CAM). |
| `POST` | `/api/templates` | Register example images under a name, for template-prompted models. |
| `GET` | `/api/templates` | List registered templates. |
| `DELETE` | `/api/templates/{name}` | Remove a template. |

Routes are declared in
[`internal/server/server.go`](https://github.com/mtbui2010/visionserve/blob/main/internal/server/server.go).

## `GET /api/models`

```json
[
  {"name": "rf-detr", "task": "detection", "license": "Apache-2.0", "state": "loaded",
   "max_useful_side": null, "max_useful_short_side": 1120, "accepts_depth": false},
  {"name": "rf-detr-nano", "task": "detection", "license": "Apache-2.0", "state": "available",
   "max_useful_side": 768, "max_useful_short_side": null, "accepts_depth": false},
  {"name": "background", "task": "segmentation", "license": "Apache-2.0", "state": "available",
   "max_useful_side": null, "max_useful_short_side": null, "accepts_depth": true}
]
```

`max_useful_side` and `max_useful_short_side` tell a client how far it may shrink a photo before
uploading it without changing what the model sees: the model resizes every photo to its own
input anyway (RF-DETR to 560 × 560), so a 12-megapixel upload mostly carries pixels it throws
away. At most one is set:

| Field | Bounds the photo's | Models | Value |
|---|---|---|---|
| `max_useful_side` | **longer** side | those that fit the photo inside their input (`letterbox`, `top_left_pad`, `long_side`): `rf-detr-nano`, `rt-detr`, `scrfd` | 2 × the larger input side |
| `max_useful_short_side` | **shorter** side | those that fill their input on both axes (`squash`, `center_crop`): `rf-detr`, `rfdetr-small*`, `grounding-dino`, `clip`, `siglip-image`, `efficientnet-b0`, `mobilenet-v3`, `midas` | 2 × the larger input side |
| both `null` | — | everything whose output needs the full photo or that crops it itself: SAM family and every model returning masks, `paddle-ocr`, grasp models, `background`, `owlv2_base_patch16` (templates), crop-naming pipelines (`rfdetr-gdino*`, `rfdetr-textalign*`, `gdino-siglip*`), `depth-anything-v2` (`keep_aspect`), text towers | send the photo as it is |

The shorter side is the one bounded for fill modes because they resample each axis on its own: a
wide panorama shrunk by its longer side would lose rows the model then stretches back. With a
region of interest (`roi`) the bound applies to the region. A manifest overrides the hint with
[`runtime.max_useful_side`](manifest.md) (`0` = never shrink). Both keys are always present;
a server that predates them sends neither, which clients read as "no hint". The
[Python and JavaScript SDKs](../clients/python.md#client-side-resizing-on-by-default) apply the
hint by default (on a server on the same machine only when it at least halves the photo's
sides) and map every result back to the original photo's pixels.

`accepts_depth` is `true` when the model reads an uploaded depth image (the `depth` /
`depth_base64` request fields): today only `background`, and only when its manifest has a MiDaS
session (`files.depth`), because the external depth replaces MiDaS in its `depth` and `auto`
methods. Every other model, the grasp models included, ignores a depth upload. SDKs use it to
send a camera's depth frame only where it is read (the Python SDK's
[`watch(depth="auto")`](../clients/python.md#watching-a-camera-or-video)). A model that accepts
depth never gets a resize hint, because the depth image is aligned to the full photo. The key is
always present; a server that predates it sends none, which the Python SDK reads as "unknown"
(no depth upload in `auto`).

## `POST /api/predict`

=== "multipart"

    ```bash
    curl -F model=grounding-dino -F image=@photo.jpg -F "prompt=cat. laptop." \
         http://127.0.0.1:11435/api/predict
    ```

=== "JSON"

    ```bash
    # the base64 of a photo is too long for a command line: write the body to a file
    printf '{"model":"grounding-dino","prompt":"cat. laptop.","image_base64":"%s"}' \
           "$(base64 -w0 photo.jpg)" > req.json
    curl -H 'Content-Type: application/json' -d @req.json http://127.0.0.1:11435/api/predict
    ```

### Request fields

| Field | Used by | Format |
|---|---|---|
| `model` | all | model name, required |
| `image` / `image_base64` | all | JPEG, PNG, WebP, BMP, GIF or TIFF (up to 32 MiB, 40 megapixels) |
| `prompt` | open-vocabulary models | words separated by `" . "`: `"cat. red mug."` |
| `box` | SAM family | `"x,y,w,h"` in the photo's pixels; several separated by `;` (with curl, send it with `--form-string`: `-F` cuts at `;`) |
| `point` | SAM family | `"x,y[,label]"`, label `1` = object, `0` = background; several separated by `;` |
| `box_threshold`, `text_threshold` | GroundingDINO-based (`box_threshold` also OWLv2) | override the manifest score cutoffs |
| `min_size`, `max_size` | detectors, segmenters | drop objects smaller / larger than this % of the image |
| `roi` | all | `"x,y,w,h"`: run on this crop only; results come back in full-photo pixels |
| `dilate` | models that return masks | grow (`> 0`) or shrink (`< 0`) every mask by this many pixels |
| `method` | `background`, `rfdetr-textalign` | `auto`, `depth`, `sam`, `cv`, `automask` / `exact`, `folded`, `gated`, `dual` |
| `bg_max_area`, `fg_min_area`, `grid_size` | `background`, `mobile-sam` (`grid_size` also `grasp`) | automatic-mask tuning |
| `claim_threshold`, `crop_temp` | `rfdetr-textalign` with `method=dual` (`crop_temp` also `gdino-siglip`) | open-vocabulary naming tuning |
| `gripper_min`, `gripper_max` | grasp models | gripper opening range in pixels |
| `depth` / `depth_base64` (+ `depth_dtype`, `depth_width`, `depth_height`) | `background` (the models with `accepts_depth`) | an aligned depth image: raw little-endian `uint16` (read as value / 65535, `0` = no reading) or `float32` (as is; ≤ 0, NaN and ±inf = no reading), resized to the photo by nearest neighbour |
| `template_name` | template-prompted models | a name registered with `/api/templates` |
| `encoding` | depth, embeddings | `base64`: return big float arrays as base64 float32 (≈6× faster than JSON numbers) |

All fields are listed in `PredictJSONRequest` in
[`pkg/api/types.go`](https://github.com/mtbui2010/visionserve/blob/main/pkg/api/types.go).
For every field's default, valid range, the models that read it and a real example, see
[Clients › Python](../clients/python.md#every-option-at-a-glance) (the Python keyword arguments
have the same names as these fields).

### Answer

```json
{
  "task": "segmentation",
  "model": "grounded-sam",
  "device": "gpu:0",
  "detections": [ { "class": "cat", "conf": 0.71, "bbox": [210, 54, 288, 301] } ],
  "masks":      [ { "conf": 0.71, "bbox": [210, 54, 288, 301], "rle": "..." } ],
  "duration_ms": 182.4
}
```

| Field | Meaning |
|---|---|
| `detections[]` | `class`, `conf` (0–1), `bbox` = `[x, y, w, h]` in the photo's pixels |
| `masks[]` | `rle` = the mask, run-length encoded in **column-major** order (COCO style), plus `bbox`, `conf` |
| `classifications[]` | `class`, `conf` — top-K labels for the whole image |
| `embeddings` | one vector per input (`embeddings_base64` + `embeddings_shape` with `encoding=base64`) |
| `depth_map`, `depth_width`, `depth_height` | relative depth, row-major (`depth_map_base64` with `encoding=base64`) |
| `grasps[]` | `x`, `y`, `theta` (radians), `width` (pixels), `quality` (0–1), source `class`/`conf` |
| `device` | where it ran: `cpu`, `gpu:0`, `gpu:0+trt` |
| `duration_ms` | server-side time |

## Errors

Errors are `{"error": "message"}` with a status code that says whose fault it was:

| Status | Meaning |
|---|---|
| `400` | The request is wrong: missing field, bad image, a prompt the model cannot use. |
| `404` | Unknown model, or its weights are not downloaded. |
| `413` | Image or body too large. |
| `499` | The client disconnected before the work started (nothing was run). |
| `503` | Too many requests for this model right now. Retry after `Retry-After` seconds. |
| `500` | A server-side failure (for example a broken model file). |

The mapping lives in `statusOf` in
[`internal/server/errors.go`](https://github.com/mtbui2010/visionserve/blob/main/internal/server/errors.go).

## Clients

You rarely need to build requests by hand: the [Python and JavaScript clients](../clients/index.md)
handle uploads, prompts, RLE decoding and base64 arrays for you. How they are built is in
[Clients and the converter](../architecture/clients.md).
