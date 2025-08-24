# Project Overview

This repository provides a comprehensive pipeline for extracting patches from whole-slide images (WSI) and performing clustering and classification tasks. It is specifically designed for analyzing histopathological images in cancer diagnosis. The code extracts image patches at multiple magnifications (2.5x, 10x, 20x), extracts features using a pre-trained `ResNet34` model, performs clustering via `K-Means`, and trains a Multi-Layer Perceptron (MLP) classifier without labels.

## Code Structure
- `extract_patches.py`: Handles patch extraction from WSIs using DeepZoomGenerator. Utilizes multiple worker processes to speed up extraction.
- `dataset.py`: Defines a custom dataset class for managing patch data, including features, labels, and coordinates. Includes a custom collate function for batch creation.
- `eval.py`: Extracts features with a pre-trained `ResNet34` model, applies `K-Means` clustering, visualization and evaluates  performance.
- `main.ipynb`: The Jupyter notebook, as an example use case, combines feature extraction, clustering, and model training with an MLP classifier.
- `requirements.txt`: Lists the Python dependencies required for the project.

## Installation

To set up the environment, use `pip` to install dependencies.

### 1. Create a Virtual Environment
First, create a virtual environment (if you haven't already):

```
python -m venv venv
```

Activate the virtual environment:
* Windows:
```
venv\Scripts\activate
```
* Linux/macOS:
```
source venv/bin/activate
```

### 2. Install Dependencies
Install the required dependencies using the requirements.txt file:
```
pip install -r requirements.txt
```

## Usage
### 1. Patch Extraction
Use extract_patches.py to extract patches from whole-slide images (WSI). The extracted patches will be saved in the specified directory, filtered by an edge threshold.
```
python extract_patches.py -i /path/to/WSI/files -o /output/directory
```
Arguments:

* `-i` or `--input`: Path to the input WSI file or directory.
* `-o` or `--output`: Output directory for saved patches.
* `-l` or `--levels`: List of magnification levels (e.g., [0, 1, 2]).
* `-t` or `--tile_size`: Size of the extracted patches (default: 224).
* `-f` or `--format`: Image format for saved patches (default: jpeg).
* `-th` or `--threshold`: Edge detection threshold for filtering patches (default: 5).
* `-w` or `--workers`: Number of worker processes (default: 20).

### 2. Feature Extraction and Clustering
Use `eval.py` to extract features using a pre-trained `ResNet34` model for patches at 2.5x, 10x, and 20x magnifications, followed by clustering the features using `K-Means` on the extracted features.

### 3. Model Training
Use the `main.ipynb` script to train a Multi-Layer Perceptron (MLP) model using the extracted features and labels. It also evaluates the model’s performance.

Steps:
1. Load extracted features.
2. Train the MLP model.
3. Evaluate using `accuracy` and `F1-score`.
   
You can run the notebook in a Jupyter environment or execute it as a script.

### 4. Visualizing Results
After training, visualize predicted labels for patches with the following function. It displays patches and their predicted labels, color-coded for each class.

## Where to Use This Code
This repository is intended for medical image analysis, especially for histopathological images. Possible applications include:
+ Cancer Detection: Classifying patches from histopathological slides for cancer diagnosis.
+ Tumor Identification: Identifying regions of interest in tissue samples.
+ Medical Research: Analyzing large datasets of histopathological images.
The workflow is scalable and can be adapted to other image analysis tasks requiring patch-based processing and feature extraction.
