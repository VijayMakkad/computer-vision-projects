# Computer Vision Projects

Collection of self-contained Jupyter notebooks for small-to-medium computer vision experiments and demos by VijayMakkad.

## Overview

This repository gathers several notebooks that demonstrate common computer vision tasks, including medical image classification, traffic sign classification, license plate detection + OCR, malaria cell detection, and road marking detection. Each notebook is intended to be runnable interactively and documents data preparation, model training or inference, and example results.

## Repository structure

- `brain_tumor_detection.ipynb` — Brain tumor detection / classification notebook (CNN-based workflow; includes preprocessing and evaluation).
- `german_traffic_signs_classification_97.ipynb` — German Traffic Sign recognition experiment (notebook reports ~97% accuracy; contains dataset processing and model training).
- `License_Plate_OCR_YOLOv8_Final.ipynb` — License plate detection using YOLOv8 and OCR post-processing (detection -> crop -> OCR pipeline).
- `Malaria_Detection.ipynb` — Malaria cell classification/detection notebook (microscopy image preprocessing + model training/evaluation).
- `road marking detection/` — Folder containing road marking detection notebook:
  - `road_mark_detection.ipynb` — Road marking detection and visualization pipeline.

> Note: Datasets are not included in this repository. See each notebook's top cells for dataset download instructions or expected local path variables.

## Quick start (recommended)

1. Install Python 3.8+ (3.9/3.10 recommended). Use a virtual environment.

Zsh (macOS) example using venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install common dependencies used across notebooks. Each notebook may require additional packages (for example, PyTorch or TensorFlow, or YOLOv8). The list below covers many common imports used in CV notebooks:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python scikit-image jupyterlab notebook pillow tqdm
# If using PyTorch-based notebooks (GPU support):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# If using TensorFlow notebooks:
# pip install tensorflow
# For OCR tasks you may need: easyocr or pytesseract
pip install easyocr pytesseract
# Ultralytics YOLOv8 (if Notebook uses it):
pip install ultralytics
```

3. Start JupyterLab or Notebook and open the desired notebook:

```bash
jupyter lab
# or
jupyter notebook
```

4. Run cells sequentially. If data paths are required, edit the top cells where `DATA_DIR`, `DATA_PATH` or similar variables are defined.

## Notebook-specific notes

- Brain tumor detection: Watch for dataset formatting (slices or 3D volumes). Many examples expect 2D slices; check the preprocessing cell.
- Traffic sign classification: The notebook name suggests ~97% accuracy — model architecture and data augmentation steps are included. Datasets like GTSRB must be downloaded separately.
- License plate OCR: Uses a detection model (YOLOv8) to localize plates, then an OCR engine to read text. For best OCR results, ensure high-quality crops and consider Tesseract language/data files or easyOCR model downloads.
- Malaria detection: Typical datasets are microscopy cell images — confirm class mapping (parasitized/uninfected) before training.
- Road marking detection: The notebook contains visualization and filtering steps; camera calibration or perspective transforms may be optional depending on use-case.

## Datasets & reproducibility

- Datasets are typically large and not stored in this repo. Each notebook contains instructions or cells to download or point to expected dataset directories.
- For reproducibility, pin package versions in a `requirements.txt` (not yet provided) or share a Conda environment YAML. Consider saving model weights and random seeds in notebooks.

## Suggested improvements / next steps

- Add a top-level `requirements.txt` or `environment.yml` that pins package versions used by the notebooks.
- Add a `LICENSE` file to clarify reuse/redistribution rights.
- Add small README snippets inside `road marking detection/` describing any dataset or extra assets in that folder.
- Provide a small sample dataset or script to download publicly-available sample data for each notebook for quick demos.

## Contributing

Contributions are welcome. Please open issues or pull requests with changes. When adding notebooks, include dataset notes and a short README or cell at the top describing expected inputs.

## Contact

Repository owner: VijayMakkad

If you want me to also generate a `requirements.txt`, add brief metadata to each notebook (author/date), or create example dataset download scripts, tell me which item to prioritize and I'll do it next.
