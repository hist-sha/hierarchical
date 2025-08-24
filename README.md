# Hierarchical WSI Patch Pipeline

## Overview
This repository provides a focused pipeline for working with whole-slide images (WSI):

* **Patch extraction** at multiple magnifications using OpenSlide and DeepZoom.
* **Custom dataset** and safe collation logic for PyTorch.
* **Feature extraction and clustering** with a pre-trained ResNet34 backbone and K-Means.
* **Exploratory training and visualization** via notebook examples.

The project is targeted at histopathology research, but the utilities are generic enough to be applied to other domains requiring tiled image analysis.

---

## Code Structure
- **`extract_patches.py`**  
  Parallel patch extraction from WSIs with configurable magnification levels, tile size, format, and filtering threshold. Uses multiple worker processes for throughput.

- **`dataset.py`**  
  Defines a `CustomDataset` for PyTorch that stores features, labels, and coordinates. Includes a robust `collate_fn` that gracefully handles `None` entries and logs errors or warnings.

- **`eval.py`**  
  Applies feature extraction via a pretrained ResNet34 model, clusters with K-Means, and provides visualization utilities for inspection of clustering results.

- **`main.ipynb`**  
  Demonstrates a full workflow: feature extraction, clustering, and training of an MLP classifier. Intended as an interactive exploration rather than production code.

---

## Installation

### 1. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows
```

### 2. Install dependencies
This project uses a modern Python packaging workflow. Dependencies are declared in `pyproject.toml`. Install with your preferred tool, e.g.:

```bash
pip install .
# or, if using uv
uv sync
```

---

## Usage

### Patch Extraction
Run `extract_patches.py` to generate tiles from WSIs:

```bash
python extract_patches.py -i /path/to/wsi -o ./patches
```

**Key arguments:**
- `-i, --input` — Path to input WSI file or directory.
- `-o, --output` — Output directory for patches.
- `-l, --levels` — List of magnification levels (default `[0,1,2]`).
- `-t, --tile_size` — Patch size in pixels (default 224).
- `-f, --format` — Output image format (default `jpeg`).
- `-th, --threshold` — Edge-detection threshold for filtering (default 5).
- `-w, --workers` — Number of parallel worker processes (default 20).

### Feature Extraction & Clustering
Use `eval.py` to compute patch embeddings with ResNet34 and cluster them via K-Means:

```bash
python eval.py --features ./patches --clusters 5
```

### Model Training
In `main.ipynb` you will find an example pipeline:
1. Load extracted patch features.
2. Cluster features to generate pseudo-labels.
3. Train a lightweight MLP classifier.
4. Evaluate using metrics such as accuracy and F1-score.

### Visualization
The notebook also demonstrates visualization of clustered patches and classification outputs, color-coded for intuitive inspection of tissue regions.

---

## Applications
- **Cancer detection** — classification of tissue patches for diagnostic pipelines.
- **Tumor localization** — identifying candidate tumor regions across WSIs.
- **Research workflows** — scalable exploration of large-scale histology datasets.

While designed for histopathology, the patch extraction and clustering utilities can generalize to any tiled image domain (remote sensing, materials science, etc.).
