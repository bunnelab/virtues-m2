# VirTues-M2
This repository contains inference examples to run VirTues-M2 for spatial proteomics and histopathology images.
Full training code will be released soon.

## Installation
To install all dependencies needed to run VirTues-M2, execute:
```
conda create --name virtuesm2 python=3.11
conda activate virtuesm2
pip install -r requirements.txt
```

The model weights for VirTues-M2 are available on [HuggingFace](https://huggingface.co/bunnelab/virtues-m2) and the modelcan be instantiated as follows:

```python
from virtuesm2.models.virtuesm2_hf import VirTuesM2_HF
model = VirTuesM2_HF.from_pretrained("bunnelab/virtues-m2", marker_embedding_dir="example_data/esm_embeddings", force_download=True)
```

## Demos
We provide two inference notebooks in the `notebooks` folder two show how to embed multi- and uni-modal images with VirTues-M2  and potential downstream tasks that leverage the frozen VirTues-M2 encoding abilities.
In notebook `1_inference_PCAs.ipynb`, simple PCAs are shown, whereas `2_cell_segmentation.ipynb` shows panoptic cell segmentation based on VirTues-M2 for joint cell phenotyping and cell instance segmentation in a single forward pass.

## License and Terms of Use

Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
Laboratory of Artificial Intelligence in Molecular Medicine, 2025

This model and associated code are released under the MIT Licence. See `LICENSE.md` for details. 