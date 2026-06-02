---
license: apache-2.0
library_name: transformers.js
---

https://huggingface.co/IDEA-Research/grounding-dino-tiny with ONNX weights to be compatible with Transformers.js.

## Usage (Transformers.js)

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

**Example:** Zero-shot object detection with `onnx-community/grounding-dino-tiny-ONNX` using the `pipeline` API.
```js
import { pipeline } from "@huggingface/transformers";

const detector = await pipeline("zero-shot-object-detection", "onnx-community/grounding-dino-tiny-ONNX");

const url = "http://images.cocodataset.org/val2017/000000039769.jpg";
const candidate_labels = ["a cat."];
const output = await detector(url, candidate_labels, {
  threshold: 0.3,
});
```


<details>

<summary>See example output</summary>

```
[
  { score: 0.45316222310066223, label: "a cat", box: { xmin: 343, ymin: 23, xmax: 637, ymax: 372 } },
  { score: 0.36190420389175415, label: "a cat", box: { xmin: 12, ymin: 52, xmax: 317, ymax: 472 } },
]
```

</details>


**Example:** Zero-shot object detection with `onnx-community/grounding-dino-tiny-ONNX` using the `AutoModel` API.
```js
import { AutoModelForZeroShotObjectDetection, AutoProcessor, load_image } from "@huggingface/transformers";

// Load model and processor
const model_id = "onnx-community/grounding-dino-tiny-ONNX";
const processor = await AutoProcessor.from_pretrained(model_id);
const model = await AutoModelForZeroShotObjectDetection.from_pretrained(model_id, { dtype: "fp32" });

// Prepare image and text inputs
const image = await load_image("http://images.cocodataset.org/val2017/000000039769.jpg");
const text = "a cat."; // NB: text query needs to be lowercased + end with a dot

// Preprocess image and text
const inputs = await processor(image, text);

// Run model
const outputs = await model(inputs);

// Post-process outputs
const results = processor.post_process_grounded_object_detection(
  outputs,
  inputs.input_ids,
  {
    box_threshold: 0.3,
    text_threshold: 0.3,
    target_sizes: [image.size.reverse()],
  },
);
console.log(results);
```

<details>

<summary>See example output</summary>

```
[
  {
    scores: [ 0.45316222310066223, 0.36190420389175415 ],
    boxes: [
      [ 343.7238121032715, 23.02229404449463, 637.0737648010254, 372.6510000228882 ],
      [ 12.311229705810547, 52.27128982543945, 317.4389839172363, 472.60459899902344 ]
    ],
    labels: [ 'a cat', 'a cat' ]
  }
]
```

</details>

---