# Prediction Examples

## 1. Data Preprocessing
```bash
python ./access/data/data_prepare.py \
  --pdb_dir ./rosetta/pdb_rosetta \          # Input directory for original PDB files
  --include_dir ./PDB_file \                 # Directory for specific PDB files to include
  --embedding_dir ./access/gvp_protein_embedding \  # Output directory for GVP embeddings
  --processed_dir ./access/gvp_protein_embedding/processed \  # Directory for preprocessed data
  --label_file benchmark_train_labels_all_0001.csv  # Prediction file
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
| AF-P00780-F1-model_v4_0001    | 3.4.21.62/0.3088 | 170, 324, 229, 146, 171, 200, 230, 227, 138, 326, 137, 167, 328, 173, 198, 196, 169, 164, 256, 330, 168, 141, 142, 140, 139, 172, 325, 323, 329, 304, 166, 279, 284, 231, 163, 305, 314, 313 |
| AF-A0A0K8P6T7-F1-model_v4_0001 | 3.1.1.101/0.3083 | 85, 162, 158, 254, 163, 86, 164, 202, 159, 160, 118, 161, 237, 82, 166, 184, 186, 165, 156, 214, 183, 182, 87, 181, 157, 84, 240, 123, 185 |
| AF-P11797-F1-model_v4_0001    | 3.2.1.14/0.3240  | 290, 95, 182, 211, 380, 144, 403, 185, 305, 402, 292, 142, 214, 294, 407, 288, 187, 124, 140, 188, 93, 186, 404, 269, 141, 291, 96, 10, 212, 289, 270, 53, 217, 228, 268, 143, 184, 374, 215, 295, 210, 213, 183, 408, 97, 379, 94, 227, 98, 293 |

