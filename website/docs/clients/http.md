# Plain HTTP

Any language that can send an HTTP request can use VisionServe; the SDKs only save you some
typing. This page shows the request shapes with `curl`. The endpoints, answer fields and status
codes are in the [HTTP API reference](../reference/api.md); the options themselves (names,
defaults, which models read them) are the [Python option table](python.md#every-option-at-a-glance),
because every Python keyword is a form field of the same name.

Examples were run with curl 7.81 against the test server described on the
[Python page](python.md) (port 11680 there; 11435 below, the default).

## Two ways to send a photo

**Multipart form** (what the SDKs send): the photo is a file part named `image`, every option
is a text field.

```bash
curl -F model=rf-detr -F image=@dogs.jpg -F min_size=1 -F roi=200,100,260,240 \
     http://127.0.0.1:11435/api/predict
```

```text
{"task":"detection","model":"rf-detr","device":"gpu:0","detections":[{"bbox":[226.03216871619225,138.73076558113098,41.86750799417496,89.99318361282349],"class":"dog","conf":0.9382620444893054},...
```

**JSON**: the photo is base64 text in `image_base64`, the options are keys with the same names.
Numbers can be JSON numbers. A photo's base64 is too long for a command line, so write the body
to a file:

```bash
printf '{"model":"grounding-dino","prompt":"cat. laptop.","box_threshold":0.35,"image_base64":"%s"}' \
       "$(base64 -w0 cat.jpg)" > req.json
curl -H 'Content-Type: application/json' -d @req.json http://127.0.0.1:11435/api/predict
```

```text
{"task":"open_vocab","model":"grounding-dino","device":"gpu:0","detections":[{"bbox":[311.63137435913086,182.54337787628174,303.4507369995117,181.91691398620605],"class":"cat","conf":0.9011985014790193},{"bbox":[7.504005432128906,172.89509296417236,310.39520263671875,246.29897117614746],"class":"laptop","conf":0.7858986071147976}],"duration_ms":134.548}
```

Prefer multipart for large photos: a JSON body is read whole (at most 32 MiB including the base64,
which is a third larger than the file), and the server can refuse a multipart upload with 503
before reading the photo when the model is busy.

## Field formats

| Kind | Format | Example |
|---|---|---|
| boxes | `x,y,w,h`, several joined by `;` | `310,175,310,195;5,170,300,245` |
| points | `x,y` or `x,y,label`, several joined by `;` | `470,195,1;470,300,0` |
| `roi` | one `x,y,w,h` (pixels, or fractions when `w`, `h` ≤ 1) | `0.25,0.25,0.5,0.5` |
| numbers | decimal text | `0.35`, `1` |
| integers (`grid_size`, `dilate`) | whole numbers only | `8`, `-3` |

!!! warning "A number the server cannot read is ignored"
    In a multipart form, a field that does not parse (`min_size=abc`, `grid_size=2.5`) is
    treated as absent: the model's default is used and there is no error. The SDKs check these
    before sending; when you build the form yourself, check them yourself.

!!! warning "curl `-F` and `;`"
    `curl -F` reads `;` as the start of an option such as `;type=`, so
    `-F "box=310,175,310,195;5,170,300,245"` silently sends only the first box. Use
    `--form-string` for values that contain `;`:

    ```bash
    curl -F model=mobile-sam -F image=@cat.jpg --form-string "box=310,175,310,195;5,170,300,245" \
         http://127.0.0.1:11435/api/predict
    ```

    That returned two masks (boxes `[320, 186, 290, 177]` and `[5, 175, 217, 242]`); with `-F`
    it returned one.

## Large arrays: `encoding=base64`

Add `encoding=base64` (a form field, a JSON key, or `?encoding=base64` in the URL) and the
server sends `depth_map` and `embeddings` as base64 float32 instead of JSON numbers:
`depth_map_base64` (row-major `depth_height × depth_width`), `embeddings_base64` with
`embeddings_shape = [N, D]`. In numpy:
`np.frombuffer(base64.b64decode(s), "<f4").reshape(shape)`.

```bash
curl -F model=midas -F image=@dogs.jpg "http://127.0.0.1:11435/api/predict?encoding=base64"
```

```text
{"task":"depth","model":"midas","device":"gpu:0","depth_width":256,"depth_height":256,"duration_ms":6.209,"depth_map_base64":"AAAAACm9ZTpcF4M6nkc8OtPNODqyG1Q6vU...
```

## A depth image

For `background`, send the raw little-endian array as a file part `depth` (JSON:
`depth_base64`) with `depth_dtype` (`uint16` or `float32`), `depth_width` and `depth_height`.
The byte length must be exactly `width × height × 2` (uint16) or `× 4` (float32), otherwise the
answer is a 400.

```bash
# depth.u16: 640 x 428 uint16 values, row-major, little-endian
curl -F model=background -F image=@living-room.jpg -F method=depth \
     -F depth=@depth.u16 -F depth_dtype=uint16 -F depth_width=640 -F depth_height=428 \
     http://127.0.0.1:11435/api/predict
```

```text
{"task":"segmentation","model":"background","device":"gpu:0","masks":[{"rle":"0 273920","bbox":[0,0,640,428],"conf":1}],...
```

## Errors

A non-2xx answer has the body `{"error": "message"}`. Retry only `503` (after the
`Retry-After` header, in seconds) and connection failures; see the
[status table](python.md#errors-and-retries).

```bash
curl -i -F model=nope -F image=@dogs.jpg http://127.0.0.1:11435/api/predict
```

```text
HTTP/1.1 404 Not Found
...
{"error":"lifecycle: model not found: \"nope\" is not in the registry"}
```

## Decoding masks yourself

A mask's `rle` is a list of run lengths over the photo's `H × W` pixels read **column by
column** (top to bottom, then the next column to the right), starting with a run of background:
`"0 273920"` above is "0 background pixels, then 273 920 mask pixels", the whole 640 × 428 photo.
The decoders in the SDKs are a few lines:
[`Mask.to_ndarray`](https://github.com/mtbui2010/vision_serve/blob/main/clients/python/visionserve/types.py#L198-L254)
(Python) and
[`Mask.toMask`](https://github.com/mtbui2010/vision_serve/blob/main/clients/js/src/types.ts#L86-L122)
(TypeScript). The format is explained in
[Shared vision library](../architecture/vision.md).
