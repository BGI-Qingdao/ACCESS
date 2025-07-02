# Training Examples

## 1. Data Preprocessing
```bash
python ./access/data/data_prepare.py \
  --pdb_dir ./rosetta/pdb_rosetta \          # Input directory for original PDB files
  --include_dir ./PDB_file \                 # Directory for specific PDB files to include
  --embedding_dir ./access/gvp_protein_embedding \  # Output directory for GVP embeddings
  --processed_dir ./access/gvp_protein_embedding/processed \  # Directory for preprocessed data
  --label_file train_labels_all_0001.csv \  # Training label file
  --split_mode train_split                   # Data splitting mode (training set)
```

## 2. Model Training
```bash
python ./access/training/train.py \
  --root_dir ./access/gvp_protein_embedding/processed \  # Path to preprocessed data
  --resultFolder ./access/result             # Output directory for training results
```

## 3. Output Results
```bash
./access/result/models/
├── best_model.pt     # Best-performing model
└── checkpoint.pt     # Training checkpoint (for incremental training)
```
