# YOLO-1D: A Unified Neural Architecture for ECG Delineation and Defectogram Analysis

Official repository for the paper **"Neural Network YOLO-1D for Useful Signals Detection on Eddy Current Rail Defectograms"** published in *Trudy ISP RAN/Proc. ISP RAS* (RSCI).

This repository provides the core neural network architecture, domain-specific pre- and post-processing pipelines, pre-trained weights, and evaluation scripts for two distinct signal/spatial processing domains:
1. **Biomedical Signal Processing (ECG):** Automated delineation on the Lobachevsky University Electrocardiography Database (**LUDB**).
2. **Non-Destructive Testing (Defectograms):** Automated constructive elements detection on our newly released dataset [**ECRv1**](https://huggingface.co/datasets/Nasochec/ECRv1) containing records of eddy current signals obtained from defectoscope during non-destructive testing of railway tracks.

---

## Released Artifacts & Licenses

All components of this research are openly available for academic and commercial use:

| Component | Task / Domain | Link | License | Rospatent Reg. No. |
| :--- | :--- | :--- | :--- | :--- |
| **Source Code** | Core backbone + ECG & Defectogram pipelines | This repository | [Apache 2.0](LICENSE) | *Pending* <!-- No. 20266XXXXXX, No. 20266YYYYYY --> |
| **Weights (Task 1)** | Pre-trained checkpoint for ECG (LUDB) | [HF Model Hub](https://huggingface.co/[your-username]/[ecg-weights-repo]) | [Apache 2.0](LICENSE) | *Included in SW Reg.* |
| **Weights (Task 2)** | Pre-trained checkpoint for Defectograms | [HF Model Hub](https://huggingface.co/[your-username]/[defect-weights-repo]) | [Apache 2.0](LICENSE) | *Included in SW Reg.* |
| **Defectogram Dataset** | New large-scale annotated dataset for Task 2 | [HF Datasets](https://huggingface.co/datasets/[your-username]/[dataset-repo]) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | *Pending* <!-- No. 202662ZZZZZ --> |

---

## Overview of the Architecture & Pipelines

While both tasks share a unified core feature-extraction backbone (`src/core/`), they rely on dedicated domain-specific input/output heads and processing pipelines:

* **Task 1 — ECG Pipeline (`src/ecg/`):** 
  * *Preprocessing:* WFDB record parsing, baseline wander removal, bandpass filtering, and multi-lead normalization.
  * *Architecture:* Core backbone with a 1D temporal segmentation/classification head.
  * *Postprocessing:* Wave boundary decoding (P, QRS, T complexes) and physiological constraint filtering.
* **Task 2 — Defectogram Pipeline (`src/defectograms/`):** 
  * *Preprocessing:* Multi-channel sensor calibration, background noise suppression, and sliding-window spatial slicing.
  * *Architecture:* Modified backbone adapted for multi-channel defectogram representations.
  * *Postprocessing:* Defect region merging, confidence thresholding, and bounding/segmentation mask reconstruction.

---

## Installation

Clone the repository and install the required dependencies (Python 3.9+ and PyTorch 2.0+ recommended):

```bash
git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
cd [your-repo-name]
pip install -r requirements.txt
```

---

## Quick Start (Out-of-the-Box Inference)

We provide minimal offline sample files in `examples/sample_data/` and interactive Jupyter notebooks so you can test the full pipeline (preprocessing -> model inference -> postprocessing) in under a minute. Pre-trained weights are downloaded automatically from Hugging Face on the first run.

### 1. Run ECG Pipeline (LUDB Sample)

**Via Command Line:**
```bash
python examples/run_ecg_infer.py \
    --config configs/ecg_ludb.yaml \
    --input examples/sample_data/ludb_sample/record_1 \
    --output results/ecg_prediction.png
```

**Via Python / Jupyter Notebook:**
See [`examples/demo_ecg_ludb.ipynb`](examples/demo_ecg_ludb.ipynb) for step-by-step visualization of filtered leads and predicted wave boundaries.

### 2. Run Defectogram Pipeline (New Dataset Sample)

**Via Command Line:**
```bash
python examples/run_defect_infer.py \
    --config configs/defectograms.yaml \
    --input examples/sample_data/defect_sample/sample_01.npy \
    --output results/defect_prediction.png
```

**Via Streaming from Hugging Face (No full dataset download required):**
```python
from datasets import load_dataset
from src.defectograms.pipeline_defect import DefectogramPipeline

# Stream a single test sample directly from Hugging Face
dataset = load_dataset("[your-username]/[dataset-repo]", split="test", streaming=True)
sample = next(iter(dataset))

# Initialize pipeline (automatically downloads weights from HF Hub)
pipeline = DefectogramPipeline.from_pretrained("[your-username]/[defect-weights-repo]")
prediction = pipeline.run(sample["raw_signal"])
pipeline.visualize(sample["raw_signal"], prediction, save_path="defect_result.png")
```
For an interactive walkthrough, open [`examples/demo_defectograms.ipynb`](examples/demo_defectograms.ipynb).

---

## Full Datasets & Training

To reproduce the training results from our paper, download the full datasets using our helper script:

```bash
# Downloads LUDB via PhysioNet (wfdb) and our Defectogram dataset via Hugging Face
python scripts/download_datasets.py --target-dir ./data
```

**Training from scratch:**
```bash
# Train Task 1 (ECG on LUDB)
python scripts/train_ecg.py --config configs/ecg_ludb.yaml --data-dir ./data/ludb

# Train Task 2 (Defectograms)
python scripts/train_defectograms.py --config configs/defectograms.yaml --data-dir ./data/defectograms
```

---

## Repository Structure

```text
├── CITATION.cff                 # Citation metadata for GitHub
├── LICENSE                      # Apache 2.0 License
├── NOTICE                       # Institutional copyright and IP registration notices
├── configs/                     # YAML configuration files for both tasks
├── src/
│   ├── core/                    # Shared neural network backbone and layers
│   ├── ecg/                     # Task 1: ECG model head, WFDB preprocessing & postprocessing
│   └── defectograms/            # Task 2: Modified defectogram model & processing pipeline
├── examples/                    # Jupyter notebooks, CLI inference scripts, and mini samples
└── scripts/                     # Dataset downloaders and full training scripts
```

---

## Third-Party Data Attribution

* **LUDB Dataset:** The ECG experiments and sample data in `examples/sample_data/ludb_sample/` utilize the [Lobachevsky University Electrocardiography Database (LUDB)](https://physionet.org/content/ludb/1.0.1/) by Kalyakulina et al., distributed via PhysioNet under the **Open Data Commons Attribution License (ODC-BY) v1.0**.

---

## License and Intellectual Property

This project was developed at **[University Name in English]** (*[Department or Laboratory Name]*).

* **Source Code & Pre-trained Weights:** Licensed under the [Apache License 2.0](LICENSE) (see [NOTICE](NOTICE) for attribution details).
* **Defectogram Dataset:** Licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

Copyright (c) 2026 [University Name in English].  
Developed by: [Your Name], [Co-author Name].

<!-- РАСКОММЕНТИРОВАТЬ ПОСЛЕ ПОЛУЧЕНИЯ СВИДЕТЕЛЬСТВ РОСПАТЕНТА:
### State Registration of Intellectual Property (Rospatent / FIPS)
* **Computer Program (ECG Analysis Pipeline):** Certificate of State Registration No. 20266XXXXXX (dated DD.MM.2026).
* **Computer Program (Defectogram Analysis Pipeline):** Certificate of State Registration No. 20266YYYYYY (dated DD.MM.2026).
* **Database (Defectogram Dataset):** Certificate of State Registration No. 202662ZZZZZ (dated DD.MM.2026).
-->

---

## Citation

If you use our neural network architecture, processing pipelines, pre-trained weights, or dataset in your research, please cite our paper:

```bibtex
@article{surname2026modelname,
  title   = {[English Title of Your Paper]},
  author  = {[Surname, Name and Coauthor, Name]},
  journal = {[English or Transliterated Name of the RSCI Journal]},
  year    = {2026},
  volume  = {[XX]},
  number  = {[X]},
  pages   = {[XX--XX]},
  doi     = {[10.XXXX/XXXXXX]},
  note    = {RSCI. EDN: [XXXXXX]}
}
```

**Для цитирования в русскоязычных изданиях (РИНЦ / ГОСТ):**
> *Фамилия И. О., Соавтор И. О.* Название вашей статьи на русском языке // Название журнала. — 2026. — Т. XX, № X. — С. XX–XX. — DOI: 10.XXXX/XXXXXX. — EDN: XXXXXX.