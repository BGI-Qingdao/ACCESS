# Prediction Examples

## 1. Data Preprocessing
```bash
python ./access/data/data_prepare.py \
  --pdb_dir ./rosetta/pdb_rosetta \          # Input directory for original PDB files
  --include_dir ./PDB_file \                 # Directory for specific PDB files to include
  --embedding_dir ./access/gvp_protein_embedding \  # Output directory for GVP embeddings
  --processed_dir ./access/gvp_protein_embedding/processed \  # Directory for preprocessed data
  --label_file train_labels_all_0001.csv  # Prediction file
```

## 2. Model Prediction
```bash
python ./access/inference/inference.py \
  --model ./access/saved_models/best_model.pt \
  --root_dir ./access/gvp_protein_embedding/processed \
  --resultFolder ./access/result \
  --print_embedding
```

## 3. Output Results

| protein_name                   | pred_label    | key_residues                                                                 |
|-------------------------------|---------------|-----------------------------------------------------------------------------|
| AF-P00780-F1-model_v4_0001    | 3.4.21.62/0.3088 | 170, 324, 229, 146, 171, 200, 227, 230, 138, 326, 137, 328, 167, 173, 198, 196, 330, 164, 256, 142, 141, 169, 168, 140, 139, 323, 329, 172, 325, 304, 279, 166, 284, 305, 165, 231, 163, 261 |
| AF-A0A0K8P6T7-F1-model_v4_0001 | 3.1.1.101/0.3083 | 85, 162, 158, 254, 163, 86, 202, 164, 159, 160, 118, 237, 161, 82, 166, 186, 184, 214, 165, 156, 183, 182, 181, 87, 157, 240, 123, 88, 120 |
| AF-P11797-F1-model_v4_0001    | 3.2.1.14/0.3240  | 290, 182, 95, 211, 380, 144, 403, 185, 305, 402, 292, 294, 214, 142, 407, 288, 187, 124, 188, 140, 93, 10, 269, 186, 404, 96, 291, 141, 212, 270, 53, 217, 228, 289, 268, 374, 184, 143, 215, 295, 408, 210, 213, 183, 97, 227, 379, 389, 98, 94 |

