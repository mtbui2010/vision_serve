# Guides

Each guide answers one question with one command. You do not need to know computer vision to
follow them. They all have the same five parts:

1. **When you need this**: two or three lines, so you know you are on the right page.
2. **The command**: one command to copy.
3. **Reading the result**: a real output, trimmed. The first line is always the verdict:
   `PASS` (all good), `WARN` (works, but look at the reason) or `FAIL` (fix it first).
4. **If it says WARN or FAIL**: a table from the message to what you do about it.
5. **Want the details?**: links to the long pages, for when you want to know how it works.

| Your question | Guide | The command |
|---|---|---|
| I trained a model. How do I serve it? | [Use a model you trained](use-your-model.md) | `visionserve convert` or `visionserve import` |
| What does this model take as input, and what does it return? | [See what a model takes and returns](see-a-model.md) | `visionserve inspect` |
| Does the served model behave like it did in training? | [Check it behaves like training](check-training.md) | `visionserve check` |
| How do I make it smaller and faster for a Jetson? | [Make it smaller and faster for Jetson](jetson.md) | `visionserve optimize` |
| How fast is it on this machine? | [Measure speed](measure-speed.md) | `visionserve bench` |
| It works, but worse than in training. Where do I look? | [My served model is worse than in training](worse-than-training.md) | the commands above, in order |

The **Deep dives** under these guides are the long versions: every check explained with real
measurements, how to do the same by hand, and the numbers behind the defaults.

!!! note "Where the outputs come from"
    Every output in these guides is a real run from 5 October 2026 on the development PC (an
    NVIDIA RTX A6000, ONNX Runtime 1.26), trimmed where it says `...`. The model is the official
    RF-DETR Nano checkpoint (`rf-detr-nano.pth`, Apache-2.0), installed under the name
    `my-detector`, so it plays the part of "a model you trained". The photos are from COCO
    val2017 (CC BY 2.0). The commands that need the converter (`convert`, `check`,
    `sensitivity`, `optimize`) ran with the converter's Python package from `clients/python`,
    which is the code the Docker image runs; the image published at that date was older than
    these commands.
