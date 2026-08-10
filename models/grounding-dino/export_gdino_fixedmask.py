#!/usr/bin/env python3
"""export_gdino_fixedmask.py -- DEV-ONLY re-export of GroundingDINO with a CORRECT,
dynamic text self-attention mask.

NOT part of the Go runtime (see CLAUDE.md: no Python at inference time). Run it once, with
the `label` conda env, to regenerate models/grounding-dino/model-fixedmask.onnx:

    python3 export_gdino_fixedmask.py models/grounding-dino/model-fixedmask.onnx

Model source: IDEA-Research/grounding-dino-tiny (Apache-2.0).

The community export (onnx-community/grounding-dino-tiny-ONNX) baked the trip count of the
transformers-4.48 Python loop in generate_masks_with_special_tokens_and_transfer_map, so only
the FIRST "."-separated phrase ever gets its attention block. Here we monkeypatch that
function with a fully vectorized, exportable equivalent (no isin, no cummax/cummin, no
torch.eye) and export the whole model.

Same 5 input names / 2 output names as the old graph so the Go side binds unchanged.
"""
import os, shutil, sys, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import onnx
import torch
from onnx import TensorProto, numpy_helper

import transformers.models.grounding_dino.modeling_grounding_dino as gd
from transformers import GroundingDinoForObjectDetection

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gdino_fixed.onnx"
SIZE = 800  # the old graph fixes pixel_values at [1,3,800,800], pixel_mask [1,800,800]


def exportable_masks(input_ids):
    """Drop-in for generate_masks_with_special_tokens_and_transfer_map.

    isin      -> Equal/Or chain over SPECIAL_TOKENS = [CLS], [SEP], ".", "?"
    cummax    -> LxL compare + ReduceMax   (L <= max_text_len = 256, negligible)
    cummin    -> LxL compare + ReduceMin
    torch.eye -> (i == j)   (EyeLike(bool) has no ORT kernel)
    """
    n = input_ids.shape[1]
    sm = (input_ids == 101) | (input_ids == 102) | (input_ids == 1012) | (input_ids == 1029)
    idx = torch.arange(n, device=input_ids.device).unsqueeze(0)
    j = idx.unsqueeze(1)
    i = idx.unsqueeze(2)
    smj = sm.unsqueeze(1)
    prev = torch.where(smj & (j <= i), j, torch.full_like(j, -1)).max(dim=2)[0]
    nxt = torch.where(smj & (j >= i), j, torch.full_like(j, n)).min(dim=2)[0]
    valid = (nxt != 0) & (nxt != n - 1) & (nxt != n)
    am = (nxt.unsqueeze(2) == nxt.unsqueeze(1)) & valid.unsqueeze(1)
    am = (i == j) | am
    pos = torch.clamp(idx.expand_as(nxt) - prev - 1, min=0)
    return am, torch.where(valid, pos, torch.zeros_like(pos))


DOUBLE, FLOAT = TensorProto.DOUBLE, TensorProto.FLOAT


def demote_tensor(t):
    if t.data_type != DOUBLE:
        return False
    arr = numpy_helper.to_array(t).astype(np.float32)
    new = numpy_helper.from_array(arr, t.name)
    t.CopyFrom(new)
    return True


def demote_double(src, dst):
    m = onnx.load(src)
    g = m.graph
    n_cast = n_init = n_attr = n_vi = 0

    for node in g.node:
        for a in node.attribute:
            # Cast(to=DOUBLE) / any op with a `dtype` attribute (ConstantOfShape, EyeLike...)
            if a.name in ("to", "dtype") and a.i == DOUBLE:
                a.i = FLOAT
                n_cast += 1
            elif a.type == onnx.AttributeProto.TENSOR and demote_tensor(a.t):
                n_attr += 1
            elif a.type == onnx.AttributeProto.TENSORS:
                for t in a.tensors:
                    if demote_tensor(t):
                        n_attr += 1

    for init in g.initializer:
        if demote_tensor(init):
            n_init += 1

    for coll in (g.value_info, g.input, g.output):
        for v in coll:
            if v.type.tensor_type.elem_type == DOUBLE:
                v.type.tensor_type.elem_type = FLOAT
                n_vi += 1

    print(f"demoted: {n_cast} cast/dtype attrs, {n_init} initializers, "
          f"{n_attr} attr tensors, {n_vi} value_infos")
    onnx.save(m, dst)


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, pixel_mask, input_ids, attention_mask, token_type_ids):
        out = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return out.logits, out.pred_boxes


def main():
    gd.generate_masks_with_special_tokens_and_transfer_map = exportable_masks
    print("monkeypatched generate_masks_with_special_tokens_and_transfer_map")

    model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
    # The fused multi-scale-deformable-attention CUDA kernel is not traceable; HF ships a
    # pure-PyTorch fallback behind this flag.
    model.config.disable_custom_kernels = True
    if hasattr(model.config, "backbone_config"):
        model.config.backbone_config.disable_custom_kernels = True
    model.eval()

    wrapper = Wrapper(model).eval()

    # Trace with a TWO-class prompt on purpose: the old export's bug was that a 1-class
    # example baked the loop; the replacement must generalize regardless.
    #   "cat. remote." -> [CLS] cat . remote . [SEP]
    input_ids = torch.tensor([[101, 4937, 1012, 6556, 1012, 102]], dtype=torch.long)
    args = (
        torch.randn(1, 3, SIZE, SIZE, dtype=torch.float32),
        torch.ones(1, SIZE, SIZE, dtype=torch.long),
        input_ids,
        torch.ones_like(input_ids),
        torch.zeros_like(input_ids),
    )

    with torch.no_grad():
        ref = wrapper(*args)
    print("torch forward ok:", [tuple(t.shape) for t in ref])

    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            args,
            OUT,
            input_names=["pixel_values", "pixel_mask", "input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits", "pred_boxes"],
            dynamic_axes={
                "input_ids": {1: "sequence_length"},
                "attention_mask": {1: "sequence_length"},
                "token_type_ids": {1: "sequence_length"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"exported in {time.time()-t0:.1f}s ({os.path.getsize(OUT)/1e6:.1f} MB)")

    # Step 3: the legacy exporter promoted the deformable-attention sampling grid to float64;
    # ORT has no GridSample kernel for a double grid, so demote every double to float32.
    tmp = OUT + ".demote.tmp"
    demote_double(OUT, tmp)
    shutil.move(tmp, OUT)
    print(f"done -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
