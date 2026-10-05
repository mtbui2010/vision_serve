# VisionServe hands-on: learn by doing

Nine Jupyter notebooks that teach VisionServe step by step. You run real models on real
photos, change one thing at a time, and look at the result. You do **not** need to know computer
vision: each notebook explains every new word the first time it is used.

Each notebook has the same shape:

- an **overview**: what you will learn, the time it takes, what you need, the steps;
- **sections**: a short explanation, the code, **What you see** (how to read the output), and a
  folded **In detail** part with more background and links to the
  [documentation](https://mtbui2010.github.io/vision_serve/);
- a **recap** and the **next** notebook.

The notebooks are saved **with their outputs**, so you can read them on GitHub first, and run
them later.

## The notebooks

Start with 00, 01 and 02. After that, pick the ones you need.

| # | Notebook | What you learn | Time | GPU? | `visionserve` program? | Needs Docker? |
|---|---|---|---|---|---|---|
| 00 | [Overview and quick start](00-overview-and-quickstart.ipynb) | What VisionServe is; check the server; download a model; first request; read and draw the JSON; the same with `curl` | 20 min | no | only to download models | no |
| 01 | [Tour of tasks](01-tour-of-tasks.ipynb) | One example per task: detection, open-vocabulary detection, segmentation, Grounded-SAM, depth, faces, text (OCR), classification, CLIP | 40 min | no (faster with one) | only to download models | no |
| 02 | [Parameters](02-parameters.ipynb) | What the common options change, with before/after pictures: thresholds, prompt words, size filters, `roi`, `dilate`, point labels, client resize; a slider | 40 min | no (faster with one) | only to download models | no |
| 03 | [What a model takes and gives back](03-inspect-a-model.ipynb) | `visionserve inspect`: the model card, input shapes, the picture the model really gets, a broken manifest | 25 min | no | yes | no |
| 04 | [Your own checkpoint](04-your-own-checkpoint.ipynb) | Serve a model you trained: `visionserve convert` and `import`, the convert report, class names, licences | 20 min + 2–4 min of commands | no (faster with one) | yes | for the converter, or use `visionserve[convert]` (see setup) |
| 05 | [Check against training](05-check-against-training.ipynb) | `visionserve check`: does the server prepare photos like your training code? PASS/FAIL, the HTML report, accuracy on labels | 30 min | no | yes | for the converter, or use `visionserve[convert]` (see setup) |
| 06 | [Speed and client resize](06-speed-and-client-resize.ipynb) | `visionserve bench`: latency, throughput, p50/p95, CPU vs GPU; what client resize saves | 20 min | optional | yes | no |
| 07 | [Smaller and faster for a Jetson](07-jetson-sensitivity-and-optimize.ipynb) | `visionserve sensitivity` and `optimize`: FP16 / INT8 / mixed versions, size, accuracy and speed | 30 min + 8–20 min of commands (GPU, depends on load) | recommended | yes | for the converter, or use `visionserve[convert]` (see setup) |
| 08 | [Robotics: depth, grasps, background](08-robotics.ipynb) | Relative depth and the nearest object, grasps for a two-finger gripper, the support surface, and the limits | 40 min | no (faster with one) | only to download models | no |

"`visionserve` program" is the command-line program (the Go binary), see setup step 3.
"Only to download models" means: if a model is already on the server, the notebook does not need
the program.

## Setup

You need four things: Python with a few packages, a running server, (for notebooks 03 to 07)
the `visionserve` program, and Jupyter.

### 1. Python and the packages

Use Python 3.9 or newer, in a separate environment (so these packages do not change other
projects). Pick conda **or** venv. Run the commands in the repository folder (`vision_serve`).

```bash
# conda
conda create -n vs-handson python=3.11 -y
conda activate vs-handson

# or venv
python -m venv .venv-handson
source .venv-handson/bin/activate          # Windows: .venv-handson\Scripts\activate

pip install -r hands-on/requirements.txt
```

`requirements.txt` installs the VisionServe Python client (the SDK) from PyPI. If that fails, or
you want the client from this repository, install it from the source:
`pip install -e clients/python`. (If the SDK is not installed at all, `handson.py` uses the copy
in `clients/python` by itself.)

### 2. Start the server

The notebooks talk to a VisionServe server at `http://127.0.0.1:11435`. Pick **one** way to run it.

**Docker** (easiest). With an NVIDIA GPU (needs the NVIDIA container toolkit):

```bash
docker run -d --gpus all -p 11435:11435 -v ~/.visionserve_models:/root/.models \
  --name visionserve mtbui2010/visionserve:latest
```

Without a GPU:

```bash
docker run -d -p 11435:11435 -v ~/.visionserve_models:/root/.models \
  --name visionserve mtbui2010/visionserve:latest-cpu
```

Models are stored in `~/.visionserve_models` on your computer, so they stay after a restart.

**From source** (you need Go 1.22 or newer and the ONNX Runtime library; see
[Getting started](https://mtbui2010.github.io/vision_serve/getting-started/)):

```bash
make build                 # makes bin/visionserve
bin/visionserve serve      # or: make serve (also finds a GPU build of ONNX Runtime)
```

Check it: open <http://127.0.0.1:11435/api/health> in a browser. It shows `{"status":"ok"}`.
If the server runs on another computer or port, set `VS_HOST` before you start Jupyter, for
example `export VS_HOST=http://192.168.0.10:11435`.

### 3. The `visionserve` program (for notebooks 03 to 07, and to download models)

Notebooks call the program with `run_cli(...)` from `handson.py`. It looks for the program in
this order:

1. `VISIONSERVE_CLI`: a full command. Use this when the server runs in **Docker**:

   ```bash
   export VISIONSERVE_CLI="docker exec visionserve visionserve"
   ```

   Then `run_cli("pull", "rf-detr")` runs `docker exec visionserve visionserve pull rf-detr`, so
   the model goes into the container's models folder, the one the server uses.
2. `VISIONSERVE_BIN`: the path of the program, for example
   `export VISIONSERVE_BIN=$PWD/bin/visionserve`.
3. `bin/visionserve` in this repository (after `make build`).
4. `visionserve` on your `PATH`.

Careful: `pip install visionserve` also installs a **Python** command called `visionserve`. It
only talks to a running server; it cannot pull, inspect or check models. `handson.py` skips it.

In Docker, the program sees only the files **inside the container**. When a command must read
files from your disk (notebooks 03 to 07), mount that folder when you start the container
(`-v /my/folder:/data`), or use the program built from source for those notebooks.

The program and the server must use the **same models folder**: Docker does this for you; from
source, both use `~/.visionserve/models` (or `VISIONSERVE_MODELS` if you set it).

Notebooks 04, 05 and 07 also need the **converter**, the Python part that reads PyTorch files.
The program starts its Docker image for you (`mtbui2010/visionserve-convert`), or you install
it into the notebook's Python: `pip install "visionserve[convert]"`. Each of these notebooks says
how.

### 4. Start Jupyter

```bash
jupyter lab hands-on/
```

Open `00-overview-and-quickstart.ipynb` and run the cells from top to bottom
(**Shift + Enter** runs one cell). The first cell that talks to the server prints
*Connected to VisionServe at ...*. If it prints an error instead, it tells you how to start
the server.

### Which environment does each notebook need?

- **00, 01, 02, 08**: steps 1, 2 and 4. The models they use are downloaded the first time (for
  this, the program from step 3 is needed once; or download them yourself with
  `visionserve pull NAME`). All of 01 needs about 2 GB of models.
- **03, 06**: steps 1 to 4.
- **04, 05, 07**: steps 1 to 4, and the converter (Docker or `visionserve[convert]`).
- A GPU is never required. It makes 01, 02, 06, 07 and 08 faster.

## What is in this folder

| Path | What it is |
|---|---|
| `00-...ipynb` to `08-...ipynb` | the notebooks |
| `handson.py` | small helpers that every notebook uses: `connect`, `ensure_model`, `photo`, `show`, `show_side_by_side`, `run_cli`, `table`, ... Read it: each function has a short explanation. |
| `images/` | the 8 example photos (COCO val2017, CC BY 2.0; authors in [`images/CREDITS.md`](images/CREDITS.md)) |
| `requirements.txt` | the Python packages |
| `tests/` | tests of `handson.py`: `python -m pytest hands-on/tests -q` |

## About the saved outputs

Every notebook was run before it was saved. The outputs were made with an NVIDIA RTX A6000 GPU
(ONNX Runtime with CUDA) and a server built from this repository; the server ran on another port,
and the outputs show the default address. On your computer the times will be different, and
some scores can change a little (in the second or third digit).

## Problems?

- First check the server: <http://127.0.0.1:11435/api/health>, and `visionserve version`.
- Look at the "In detail" part of the section: it often explains the cause.
- If this does not help, open an issue: <https://github.com/mtbui2010/vision_serve/issues>. Please write
  the notebook name and section, the full error message, how you run the server (Docker or from
  source, CPU or GPU), and the output of `visionserve version`.
