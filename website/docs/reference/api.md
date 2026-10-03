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
| `GET` | `/api/models` | Every model: `name`, `task`, `license`, `state` (`not_downloaded`, `available`, `loaded`). |
| `POST` | `/api/load` | `{"model":"rf-detr"}` — load now instead of on first use. |
| `POST` | `/api/unload` | `{"model":"rf-detr"}` — free its memory. |
| `POST` | `/api/predict` | Run a model on an image. **The main endpoint.** |
| `POST` | `/api/preprocess` | Return the exact input tensor a model would receive, without running it (debugging). |
| `POST` | `/api/infer_tensor` | Run a model on a tensor you prepared yourself (raw float32 body). |
| `POST` | `/api/explain` | A heatmap of what the model looked at for one detection (Score-CAM). |
| `POST` | `/api/templates` | Register example images under a name, for template-prompted models. |
| `GET` | `/api/templates` | List registered templates. |
| `DELETE` | `/api/templates/{name}` | Remove a template. |

Routes are declared in
[`internal/server/server.go`](https://github.com/mtbui2010/vision_serve/blob/main/internal/server/server.go).

## `POST /api/predict`

=== "multipart"

    ```bash
    curl -F model=grounding-dino -F image=@photo.jpg -F "prompt=cat. laptop." \
         http://127.0.0.1:11435/api/predict
    ```

=== "JSON"

    ```bash
    curl -H 'Content-Type: application/json' \
         -d '{"model":"grounding-dino","image_base64":"'"$(base64 -w0 photo.jpg)"'","prompt":"cat. laptop."}' \
         http://127.0.0.1:11435/api/predict
    ```

### Request fields

| Field | Used by | Format |
|---|---|---|
| `model` | all | model name, required |
| `image` / `image_base64` | all | JPEG or PNG (up to 32 MiB, 40 megapixels) |
| `prompt` | open-vocabulary models | words separated by `" . "`: `"cat. red mug."` |
| `box` | SAM family | `"x,y,w,h"` in the photo's pixels; several separated by `;` |
| `point` | SAM family | `"x,y[,label]"`, label `1` = object, `0` = background; several separated by `;` |
| `box_threshold`, `text_threshold` | GroundingDINO-based | override the manifest thresholds |
| `min_size`, `max_size` | detectors, segmenters | drop objects smaller / larger than this % of the image |
| `roi` | all | `"x,y,w,h"`: run on this crop only; results come back in full-photo pixels |
| `method` | `background` | `auto`, `depth`, `sam`, `cv`, `automask` |
| `gripper_min`, `gripper_max` | grasp models | gripper opening range in pixels |
| `depth` / `depth_base64` | grasp / background | an aligned depth image |
| `template_name` | template-prompted models | a name registered with `/api/templates` |
| `encoding` | depth, embeddings | `base64`: return big float arrays as base64 float32 (≈6× faster than JSON numbers) |

All fields are listed in `PredictJSONRequest` in
[`pkg/api/types.go`](https://github.com/mtbui2010/vision_serve/blob/main/pkg/api/types.go).

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
[`internal/server/errors.go`](https://github.com/mtbui2010/vision_serve/blob/main/internal/server/errors.go).

## Clients

You rarely need to build requests by hand: the [Python and JavaScript clients](../architecture/clients.md)
handle uploads, prompts, RLE decoding and base64 arrays for you.
