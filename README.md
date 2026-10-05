# VisionServe

**Ollama for computer vision.** One small Go binary that serves vision models on your own
machine over HTTP: detection, segmentation, open-vocabulary detection, depth, faces, OCR,
embeddings, grasping. No account, no cloud, no telemetry. Apache-2.0.

📖 **Documentation: <https://mtbui2010.github.io/vision_serve/>**: concepts, how it works,
model gallery, HTTP API, and a Go tutorial built on this code base.

| I want to… | Read |
|---|---|
| Install it (Docker or from source, GPU setup) | [Getting started](https://mtbui2010.github.io/vision_serve/getting-started/) |
| Choose a model and run it | [Models and performance](https://mtbui2010.github.io/vision_serve/reference/models/) · [What the models do](https://mtbui2010.github.io/vision_serve/concepts/tasks/) · [Run your first request](https://mtbui2010.github.io/vision_serve/getting-started/#4-make-a-request) |
| Learn by doing (notebooks) | [Hands-on notebooks](hands-on/) · [Hands-on on the website](https://mtbui2010.github.io/vision_serve/hands-on/) |
| Use a model you trained (convert a checkpoint, import an ONNX file) | [Use a model you trained](https://mtbui2010.github.io/vision_serve/guides/use-your-model/) · [Getting started, step 6](https://mtbui2010.github.io/vision_serve/getting-started/#6-use-a-model-you-trained) |
| Inspect a model: what it takes and returns, does it behave like training | [See what a model takes and returns](https://mtbui2010.github.io/vision_serve/guides/see-a-model/) · [Check it behaves like training](https://mtbui2010.github.io/vision_serve/guides/check-training/) · [Worse than in training?](https://mtbui2010.github.io/vision_serve/guides/worse-than-training/) |
| Make it fast on Jetson (Orin, Thor) | [Make it smaller and faster for Jetson](https://mtbui2010.github.io/vision_serve/guides/jetson/) · [Measure speed](https://mtbui2010.github.io/vision_serve/guides/measure-speed/) |
| Call it from Python / JavaScript, with every parameter explained | [Clients](https://mtbui2010.github.io/vision_serve/clients/) · [Python](https://mtbui2010.github.io/vision_serve/clients/python/) |
| Call the HTTP API directly | [HTTP API](https://mtbui2010.github.io/vision_serve/reference/api/) |
| Understand or change the code | [How it works](https://mtbui2010.github.io/vision_serve/architecture/) · [Go for this project](https://mtbui2010.github.io/vision_serve/go/) |

![Grounded-SAM: "dog. person. bench."](website/docs/assets/img/grounded-sam-372819.jpg)

## Quick start

```bash
# Docker (GPU; use :latest-cpu on machines without an NVIDIA GPU)
docker run -d --gpus all -p 11435:11435 -v ~/.visionserve_models:/root/.models \
  --name visionserve mtbui2010/visionserve:latest
docker exec visionserve visionserve pull rf-detr

# Ask what is in a photo
curl -F model=rf-detr -F image=@photo.jpg http://127.0.0.1:11435/api/predict
```

```json
{ "task": "detection", "model": "rf-detr",
  "detections": [ { "class": "cat", "conf": 0.94, "bbox": [210, 54, 288, 301] } ] }
```

`bbox` is `[x, y, width, height]` in the pixels of the photo you sent.

**From source:** Go ≥ 1.22 and the ONNX Runtime shared library.

```bash
make build                                   # -> bin/visionserve
bin/visionserve pull grounded-sam
bin/visionserve serve                        # http://127.0.0.1:11435
```

**Clients:** `pip install visionserve` (Python), `npm install visionserve` (JS/TS).

See [Getting started](https://mtbui2010.github.io/vision_serve/getting-started/) for GPU setup,
prompts, Python and JavaScript examples.

## Models

| Task | Models |
|---|---|
| Detection | `rf-detr`, `rf-detr-nano`, `rfdetr-small` |
| Open-vocabulary detection | `grounding-dino`, `owlvit`, `rfdetr-gdino` (hybrid), `rfdetr-textalign-*` |
| Segmentation | `mobile-sam`, `sam2`, `efficient-sam`, `grounded-sam` (text → masks) |
| Depth | `depth-anything-v2`, `midas` |
| Faces / OCR | `scrfd`, `paddle-ocr` |
| Classification / embeddings | `efficientnet-b0`, `mobilenet-v3`, `clip`, `siglip-*` |
| Robotics | `grasp-rfdetr`, `grasp-gd`, `background` |

`visionserve list` shows what is installed and what can be pulled. Sources, licences, a selection
guide and GPU latency: [Models and performance](https://mtbui2010.github.io/vision_serve/reference/models/).
Real outputs: [model gallery](https://mtbui2010.github.io/vision_serve/gallery/).

## Principles

- **Permissive licences only.** Every model declares its licence; the registry refuses anything
  that is not Apache-2.0, MIT or BSD (no AGPL, e.g. Ultralytics YOLO).
- **All inference through ONNX Runtime.** No Python at runtime. GPU by default (CUDA → CPU),
  TensorRT opt-in with `--tensorrt`.
- **One answer format for every model**, with boxes and masks in original-image coordinates.

## Contributing

Read [CLAUDE.md](CLAUDE.md) (project rules) and
[docs/engineering-rules.md](docs/engineering-rules.md) (rules learned from real bugs). Adding a
model: [docs/contributing-models.md](docs/contributing-models.md) and the
[walkthrough](https://mtbui2010.github.io/vision_serve/go/08-add-a-model/).
Reference material that used to live here: [docs/architecture.md](docs/architecture.md),
[docs/manifest-spec.md](docs/manifest-spec.md), [deploy/README.md](deploy/README.md) (Docker),
[clients/python](clients/python/README.md), [clients/js](clients/js/README.md).

## Citation

```bibtex
@misc{visionserve2026,
  title  = {VisionServe: A Lean, License-Safe Inference Server for Computer Vision},
  author = {Bui, Trung Minh},
  year   = {2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE). Example photo: COCO val2017 #372819
([Flickr](http://farm3.staticflickr.com/2046/2516944023_d00345997d_z.jpg), CC BY 2.0).
