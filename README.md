# ACCESS_v2: Graph Neural Network-Based EC Number Prediction Analysis Software version 2
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 🧬 Project Overview
ACCESS_v2 is a cutting-edge computational tool in the field of biomanufacturing. It innovatively integrates 3D protein structural topology, residue-level Rosetta energy, and side-chain features to construct a hybrid graph neural network architecture based on multimodal feature fusion. By leveraging hierarchical contrastive learning to build a protein "semantic space," it enables accurate prediction of protein functions. Moreover, it employs a topology-aware gradient attention mechanism to precisely identify key functional residues. This creates a comprehensive closed-loop intelligent optimization system that covers activity existence, function prediction, and rational design, thus overcoming the limitations of traditional tools.
![image](https://github.com/user-attachments/assets/1ee86e4d-225c-476b-a6c4-066a006717dd)

## ✨ Core Features
- **Activity Existence Determination**: Accurately identifies protein activity status, addressing the inability of existing tools to distinguish inactive proteins.
- **Hierarchical EC Number Prediction**: Implements a novel dual-modal model for tiered EC number prediction.
- **Structure-Function Joint Analysis**: Combines 3D structural features with energy calculation data for comprehensive analysis.
- **Contrastive Learning Strategy**: Employs advanced contrastive learning to enhance functional recognition of low-sequence-homology proteins.
- **Key Residue Identification**: Uses a topology-aware gradient attention mechanism to precisely locate functionally critical residues.
- **Custom Prediction Support**: Supports model retraining with user-specific datasets for specialized applications.

## 🛠️ Installation Guide
### Requirements
- NVIDIA GPU (24GB VRAM, 16-core CPU recommended)
- CUDA 12.4+
- Python 3.8+

### Quick Installation
```bash
# Create virtual environment
conda create -n ACCESS python=3.8
conda activate ACCESS
# Install core dependencies
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
pip install argparse biopython torchdrug torch-scatter fair-esm scipy natsort
# Install GVP library
git clone https://github.com/drorlab/gvp-pytorch.git
cd gvp-pytorch
patch -p0 < ../patch_file.patch
pip install .
```

## 🚀 Usage Guide
### 1. Complete Workflow
#### Data Preparation Module
```bash
python ./final_code/data/data_prepare.py \
  --pdb_dir /path/to/pdb_files \
  --include_dir /path/to/include_files \
  --embedding_dir /path/to/embedding_output \
  --processed_dir /path/to/processed_output \
  --label_file labels.csv
```
Optional Parameters: \
--split_mode: Data splitting strategy. Options are train_split, all_train, all_valid, all_test. Use 'train_split' for an 8:1:1 split during training. Use 'all_test' to utilize the full dataset during testing. (default: all_test) \
--num_workers: Parallel workers (default: 64) \
--seed: Random seed (default: 42)

#### Training Module
```bash
python ./final_code/training/train.py \
  --root_dir ./final_code/gvp_protein_embedding/processed \
  --resultFolder ./final_code/result
```
Optional Parameters: \
--batch_size: Training batch size (default: 8) \
--num_workers: Data loader workers (default: 2)

#### Inference Module
```bash
python ./final_code/inference/inference.py \
  --model ./final_code/saved_models/best_model.pt \
  --root_dir ./final_code/gvp_protein_embedding/processed \
  --resultFolder ./complete_code/result
```
Optional Parameters:
--print_true_label: Print true labels in output \
--print_embedding: Print embeddings in output \
--batch_size: Inference batch size (default: 8) \
--num_workers: Data loader workers (default: 2)

## 📝 Format Specifications
Use -- separators between major sections
Required parameters shown in command templates
Optional parameters listed with defaults
Maintain aligned parameters in code blocks
Keep descriptions concise and technical

## ⚠️ Important Notes:
Path Configuration:
Always replace /path/to/... with actual file paths
Use absolute paths for system operations
System Requirements:
Verify directory permissions match execution privileges
Adjust batch size according to GPU VRAM:
24GB VRAM: Max batch size 8
Best Practices:
Pre-process all input files before execution
Validate output files after each module execution
