#!/usr/bin/env python3
"""Export the CLIP ViT-B/32 TEXT tower (openai/clip-vit-base-patch32, MIT) to ONNX.

This is DEV TOOLING, not runtime: VisionServe itself never runs Python (CLAUDE.md).
It produces models/clip-text/model.onnx, the text half of the image tower already
shipped as models/clip/model.onnx.

    python scripts/export_onnx_clip_text.py models/clip-text

Notes on the two easy ways to get this wrong:
  * Use CLIPTextModelWithProjection, NOT CLIPTextModel — the `text_projection` matrix is
    what maps the pooled hidden state into the SHARED image/text space. Without it the
    vectors are 512-d but in a different space, and every cosine is noise.
  * Pooling happens at the <|endoftext|> position, which HF handles internally. Because
    the transformer is causally masked and padding uses <|endoftext|> itself, tokens
    after the first EOS cannot influence the pooled vector — so the graph needs NO
    attention_mask input.
"""
import os
import sys

import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

MODEL_ID = "openai/clip-vit-base-patch32"
CONTEXT_LENGTH = 77


class TextTower(torch.nn.Module):
    """input_ids [batch, 77] -> text_embeds [batch, 512]."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids):
        return self.inner(input_ids=input_ids).text_embeds


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_clip_text_tmp.onnx")
    final = os.path.join(out_dir, "model.onnx")

    tok = CLIPTokenizerFast.from_pretrained(MODEL_ID)
    net = TextTower(CLIPTextModelWithProjection.from_pretrained(MODEL_ID)).eval()

    ids = tok(["a photo of a remote", "a photo of a coffee cup"],
              padding="max_length", max_length=CONTEXT_LENGTH, truncation=True,
              return_tensors="pt")["input_ids"]
    with torch.no_grad():
        ref = net(ids)
    print("torch reference:", tuple(ref.shape))

    # dynamo=True: the legacy TorchScript exporter chokes on some ops in these graphs.
    torch.onnx.export(
        net, (ids,), tmp,
        input_names=["input_ids"], output_names=["text_embeds"],
        dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}},
        opset_version=18, dynamo=True,
    )

    # Re-save as ONE self-contained file (the dynamo exporter writes external data),
    # so manifest.yaml's sha256 can pin a single artifact.
    import onnx
    onnx.save_model(onnx.load(tmp), final, save_as_external_data=False)
    onnx.checker.check_model(final)
    os.remove(tmp)
    for stale in (tmp + ".data", os.path.join(out_dir, "_clip_text_tmp.onnx.data")):
        if os.path.exists(stale):
            os.remove(stale)

    import onnxruntime as ort
    sess = ort.InferenceSession(final, providers=["CPUExecutionProvider"])
    print("inputs :", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
    got = sess.run(None, {"input_ids": ids.numpy()})[0]
    print("max |onnx - torch| =", float(torch.tensor(got).sub(ref).abs().max()))
    for n in (1, 3):
        shape = sess.run(None, {"input_ids": ids[:1].repeat(n, 1).numpy()})[0].shape
        print(f"batch {n} -> {shape}")

    # The pure-Go tokenizer reads these from the model directory at load time.
    from huggingface_hub import hf_hub_download
    import shutil
    for name in ("vocab.json", "merges.txt"):
        shutil.copy(hf_hub_download(MODEL_ID, name), os.path.join(out_dir, name))
        print("copied", name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/clip-text")
