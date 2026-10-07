# Hands-on notebooks

Learn VisionServe by doing: nine Jupyter notebooks in the
[`hands-on/`](https://github.com/mtbui2010/vision_serve/tree/main/hands-on) folder of the
repository. You run real models on real photos, change one thing at a time, and look at the
result. No computer-vision knowledge is needed: every new word is explained the first time.

Each notebook starts with an overview (what you will learn, the time, what you need, the steps).
Each section has a short explanation, the code, **What you see** (how to read the output) and a
folded **In detail** part that links to the pages of this site. The notebooks are saved with
their outputs, so you can read them on GitHub before you run them.

## The notebooks

Start with 00, 01 and 02; then pick the ones you need.

| # | Notebook | What you learn | Time | GPU? | Needs Docker? |
|---|---|---|---|---|---|
| 00 | [Overview and quick start](https://github.com/mtbui2010/vision_serve/blob/main/hands-on/00-overview-and-quickstart.ipynb) | What VisionServe is; check the server; download a model; first request; read and draw the JSON; the same with `curl` | 20 min | no | no |
| 01 | [Tour of tasks](https://github.com/mtbui2010/vision_serve/blob/main/hands-on/01-tour-of-tasks.ipynb) | One example per task: detection, open-vocabulary detection, segmentation, Grounded-SAM, depth, faces, text (OCR), classification, CLIP | 40 min | no (faster with one) | no |
| 02 | [Parameters](https://github.com/mtbui2010/vision_serve/blob/main/hands-on/02-parameters.ipynb) | What the common options change, with before/after pictures: thresholds, prompt words, size filters, `roi`, `dilate`, point labels, client resize; a slider | 40 min | no (faster with one) | no |
| 03 | [What a model takes and gives back](https://github.com/mtbui2010/vision_serve/blob/main/hands-on/03-inspect-a-model.ipynb) | `visionserve inspect`: the model card, input shapes, the picture the model really gets, a broken manifest | 25 min | no | no |
| 04 | [Your own checkpoint](https://github.com/mtbui2010/vision_serve/blob/main/hands-on/04-your-own-checkpoint.ipynb) | Serve a model you trained: `visionserve convert` and `import`, the convert report, class names, licences | 20 min + 2–4 min of commands | no (faster with one) | for the converter (or `visionserve[convert]`) |
| 05 | [Check against training](https://github.com/mtbui2010/vision_serve/blob/main/hands-on/05-check-against-training.ipynb) | `visionserve check`: does the server prepare photos like your training code? PASS/FAIL, the HTML report, accuracy on labels | 30 min | no | for the converter (or `visionserve[convert]`) |
| 06 | [Speed and client resize](https://github.com/mtbui2010/vision_serve/blob/main/hands-on/06-speed-and-client-resize.ipynb) | `visionserve bench`: latency, throughput, p50/p95, CPU vs GPU; what client resize saves | 20 min | optional | no |
| 07 | [Smaller and faster for a Jetson](https://github.com/mtbui2010/vision_serve/blob/main/hands-on/07-jetson-sensitivity-and-optimize.ipynb) | `visionserve sensitivity` and `optimize`: FP16 / INT8 / mixed versions, size, accuracy and speed | 30 min + 8–20 min of commands (GPU, depends on load) | recommended | for the converter (or `visionserve[convert]`) |
| 08 | [Robotics: depth, grasps, background](https://github.com/mtbui2010/vision_serve/blob/main/hands-on/08-robotics.ipynb) | Relative depth and the nearest object, grasps for a two-finger gripper, the support surface, and the limits | 40 min | no (faster with one) | no |

## Setup in four steps

The full instructions are in the folder's
[README](https://github.com/mtbui2010/vision_serve/blob/main/hands-on/README.md).

1. **Python packages.** In a new conda or venv environment, from the repository folder:
   `pip install -r hands-on/requirements.txt` (the VisionServe client, JupyterLab, matplotlib,
   ipywidgets, pandas, Pillow, numpy). To use the client from your checkout instead (for example
   a version newer than the one on PyPI), run `pip install -e clients/python`.
2. **A running server** at `http://127.0.0.1:11435`: Docker
   (`docker run -d --gpus all -p 11435:11435 -v ~/.visionserve_models:/root/.models --name visionserve mtbui2010/visionserve:latest`,
   or the `:latest-cpu` image without `--gpus all`), or from source (`make build`, then
   `bin/visionserve serve`). See [Getting started](getting-started.md). Another address: set
   `VS_HOST`.
3. **The `visionserve` program** for notebooks 03 to 07 and for downloading models. With Docker:
   `export VISIONSERVE_CLI="docker exec visionserve visionserve"`. From source:
   `export VISIONSERVE_BIN=$PWD/bin/visionserve`. Notebooks 04, 05 and 07 also need the
   converter: Docker, or `pip install "visionserve[convert]"`.
4. **Jupyter:** `jupyter lab hands-on/`, then open `00-overview-and-quickstart.ipynb`.

The notebooks share a small helper file,
[`handson.py`](https://github.com/mtbui2010/vision_serve/blob/main/hands-on/handson.py):
`connect()`, `ensure_model()`, `photo()`, `show()`, `run_cli()` and a few more. The photos are
COCO val2017 images (CC BY 2.0), credited in
[`images/CREDITS.md`](https://github.com/mtbui2010/vision_serve/blob/main/hands-on/images/CREDITS.md).

Problems or ideas: [open an issue](https://github.com/mtbui2010/vision_serve/issues) with the
notebook name, the section and the full error message.
