# TASK SPECIFICATION: Cross-Domain Generalization Experiment for MyNet-1d on ECG Delineation

## 1. Project Overview & Objective
We are preparing an academic paper presenting **MyNet-1d**, a novel 1-D deep learning architecture originally designed for multi-channel eddy current defectoscopy (eddy current NDT). The model handles long multi-channel sequences (~50,000 samples) and performs precise temporal boundary detection (left and right bounds) alongside signal classification.

**Goal:** To prove the method's generalizability and robustness on a different electromagnetic domain within a tight deadline (< 1 month), we will adapt and evaluate MyNet-1d on a standard ECG delineation benchmark (LUDB - Lobachevsky University Database). 
**Execution:** You (Claude Code) are fully responsible for the end-to-end experiment pipeline: data downloading, preprocessing, benchmark formatting, model adaptation, augmentation ablation study, training, and evaluation.

---

## 2. Environment & Repository Structure
- **Framework:** PyTorch
- **Target Dataset:** LUDB (Lobachevsky University Database) — 12-channel ECG records with expert boundary annotations for P, QRS, and T waves.
- **Project Structure to Setup:**
  ```text
  ├── data/                  # Raw and processed LUDB dataset
  ├── src/yolo1d/
  │   ├── yolo_1d.py         # Training, validation and testing loops, logging and etc
  │   ├── nets/              # neural network code and loss function
  │   └── utils/             # Dataset loaders, preprocessing, and augmentations 
  │        ├── settings/     # 
  │        ├── transforms/   # Data augmentations 
  │        ├── util.py       # metric computation and postprocessing steps like NMS
  │        ├── dataset.py    # Dataset class for preprocessing data before training
  │        └── constants.py  # Global constants to define model size
  ├── experiments/           # Logging, checkpoints, and ablation configs
  └── run_experiment.py      # Main entry point
- Repository have several branches. You can only use branch `EKG`