import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import pandas as pd
import sklearn
import torch
import torch_geometric
import torch_cluster
from functools import partial
from multiprocessing import Pool, Manager
from tqdm import tqdm
from collections import defaultdict
import csv
import numpy as np
import gzip
from Bio.PDB import PDBParser
from gvp.data import ProteinGraphDataset
import threading
import re
import torch.nn.functional as F
import io
import logging
from math import atan2
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
torch.set_num_threads(1)



HEADER_PATTERN = re.compile(r'#BEGIN_POSE_ENERGIES_TABLE')
TAIL_PATTERN = re.compile(r'#END_POSE_ENERGIES_TABLE')
GLOBAL_VALUE_PATTERN = re.compile(r'^\w+\s+([-+]?\d+\.?\d*)$')

def search_pdb_files(directory, include_file=None, to_file=None):
    """Collect PDB files with optional CSV-based filtering"""
    
    input_ = []
    processed_pdb = set()
    
    # Load inclusion list if provided
    include_names = set()
    if include_file:
        include_names = set(pd.read_csv(include_file)['protein_name'])
    
    # Scan directory for PDB files
    for file in os.listdir(directory):
        if not (file.endswith('.pdb.gz') or file.endswith('.pdb')):
            continue
            
        # Extract base name
        pdb = file[:-7] if file.endswith('.pdb.gz') else file[:-4]
        
        # Check processing status
        if pdb in processed_pdb:
            continue
            
        # Apply inclusion filter
        if not include_file or pdb in include_names:
            protein_path = os.path.join(directory, file)
            output_path = f"{to_file}/{pdb}.pt"
            input_.append((pdb, protein_path, output_path))
            processed_pdb.add(pdb)
            
    return input_

three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
#ensure_ca_exist=true only  CA
def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            # if res.resname not in three_to_one:
            #     if verbose:
            #         print(res, "has non-standard resname")
            #     continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list

def parse_energy_data(lines):
    # Stage 1: Single pass to locate key positions
    start, end = -1, len(lines)
    for i, line in enumerate(lines):
        if HEADER_PATTERN.match(line):
            start = i + 1
        elif TAIL_PATTERN.match(line) and start != -1:
            end = i
            break  # Stop scanning once both start and end are found
    filtered_lines = [
        line for line in lines[start+3:end]
        if not line.strip().startswith('pdb_UNK')
        and line.strip()[0:3] in three_to_one
    ]
    # Stage 2: Direct memory-mapped parsing
    energy_matrix = np.genfromtxt(
        filtered_lines,      # Directly pass the sliced lines
        delimiter=' ',
        dtype=np.float32,
        autostrip=True
    )[:, 1:]  # Remove the first column
    
    # Stage 3: Precompute target range + Vectorized operations
    global_features = []
    # Directly locate the target line range (avoiding line-by-line checks)
    target_lines = lines[end+2:end+12]  # Extract up to 10 lines
    for line in target_lines:
        if not line.strip(): 
            continue
        # Use split instead of regex for simplicity
        last_val = line.strip().split()[-1] 
        try:
            global_features.append(float(last_val))
        except ValueError:
            pass
        if len(global_features) >= 10:
            break
    
    # Combine features
    pose = list(map(float, lines[start + 2].split()[1:]))
    pose_with_global = global_features + pose
    
    return energy_matrix, pose_with_global

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

class CustomProteinGraphDataset(ProteinGraphDataset):
    """
    Enhanced protein graph dataset with custom residue encoding
    
    Inherits from base ProteinGraphDataset and adds:
    - Custom amino acid residue encoding/decoding
    - Enhanced geometric feature extraction
    - Integrated label handling for downstream tasks
    """
    def __init__(self, data_list, 
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cpu"):
        
        super().__init__(
            data_list=data_list,
            num_positional_embeddings=num_positional_embeddings,
            top_k=top_k,
            num_rbf=num_rbf,
            device=device
        )
        
        self.letter_to_num = {
            'C': 23, 'D': 13, 'S': 8, 'Q': 16, 'K': 15, 'I': 12,
            'P': 14, 'T': 11, 'F': 18, 'A': 5, 'G': 6, 'H': 21,
            'E': 9, 'L': 4, 'R': 10, 'W': 22, 'V': 7, 
            'N': 17, 'Y': 19, 'M': 20
        }
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        
    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design

        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.linalg.cross(u_2, u_1))
        n_1 = _normalize(torch.linalg.cross(u_1, u_0))
        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
        
    def _featurize_as_graph(self, protein):
        name = protein['name']
        features = protein['features']
        labels = protein['labels']
        with torch.no_grad():
            coords = torch.as_tensor(protein['coords'], 
                                     device=self.device, dtype=torch.float32)   
            seq = torch.as_tensor([self.letter_to_num[a] for a in protein['seq']],
                                  device=self.device, dtype=torch.long)
            res_features = torch.as_tensor(protein['res_features'], 
                                device=self.device,
                                dtype=torch.float32)
            energy_features = torch.as_tensor(protein['res_energy'], 
                                device=self.device,
                                dtype=torch.float32)
            protein_center = torch.mean(coords, dim=0, keepdim=True)
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)

            # edge_index = torch_cluster.radius_graph(
            #     X_ca, 
            #     r=10.0,                
            #     )
            
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            # Calculate the Euclidean distance between nodes.
            edge_dis = torch.norm(E_vectors, p=2, dim=-1)
            dihedrals = self._dihedrals(coords)                     
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)
            
            node_s = torch.cat([dihedrals, res_features, energy_features], dim=-1)  #
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = _normalize(E_vectors).unsqueeze(-2)
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))
            
        data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name,features = features,
                                         protein_center=protein_center, node_s=node_s, node_v=node_v,
                                         edge_s=edge_s, edge_v=edge_v,edge_dis=edge_dis,
                                         edge_index=edge_index,labels = labels, mask=mask)
        return data  
    
CHI_ATOMS = {
    # Chi1: First sidechain dihedral angle
    "CHI1": {
        "SER": ["N", "CA", "CB", "OG"],
        "THR": ["N", "CA", "CB", "OG1"],
        "CYS": ["N", "CA", "CB", "SG"],
        "VAL": ["N", "CA", "CB", "CG1"],
        "ILE": ["N", "CA", "CB", "CG1"],
        "ASP": ["N", "CA", "CB", "CG"],
        "ASN": ["N", "CA", "CB", "CG"],
        "HIS": ["N", "CA", "CB", "CG"],
        "TRP": ["N", "CA", "CB", "CG"],
        "PHE": ["N", "CA", "CB", "CG"],
        "PRO": ["N", "CA", "CB", "CG"],
        "TYR": ["N", "CA", "CB", "CG"],
        "GLU": ["N", "CA", "CB", "CG"],
        "GLN": ["N", "CA", "CB", "CG"],
        "MET": ["N", "CA", "CB", "CG"],
        "LEU": ["N", "CA", "CB", "CG"],
        "LYS": ["N", "CA", "CB", "CG"],
        "ARG": ["N", "CA", "CB", "CG"]
    },
    
    # Chi2: Second sidechain dihedral angle
    "CHI2": {
        "ASP": ["CA", "CB", "CG", "OD1"],
        "ASN": ["CA", "CB", "CG", "OD1"],
        "HIS": ["CA", "CB", "CG", "ND1"],
        "ILE": ["CA", "CB", "CG1","CD1"],
        "TRP": ["CA", "CB", "CG", "CD1"],
        "PHE": ["CA", "CB", "CG", "CD1"],
        "TYR": ["CA", "CB", "CG", "CD1"],
        "PRO": ["CA", "CB", "CG", "CD"],
        "GLU": ["CA", "CB", "CG", "CD"],
        "GLN": ["CA", "CB", "CG", "CD"],
        "MET": ["CA", "CB", "CG", "SD"],
        "LEU": ["CA", "CB", "CG", "CD1"],
        "LYS": ["CA", "CB", "CG", "CD"],
        "ARG": ["CA", "CB", "CG", "CD"]
    },
    
    # Chi3: Third sidechain dihedral angle
    "CHI3": {
        "GLU": ["CB", "CG", "CD", "OE1"],
        "GLN": ["CB", "CG", "CD", "OE1"],
        "MET": ["CB", "CG", "SD", "CE"],
        "LYS": ["CB", "CG", "CD", "CE"],
        "ARG": ["CB", "CG", "CD", "NE"]
    },
    
    # Chi4: Fourth sidechain dihedral angle
    "CHI4": {
        "LYS": ["CG", "CD", "CE", "NZ"],
        "ARG": ["CG", "CD", "NE", "CZ"]
    }
}

def compute_dihedral(p0, p1, p2, p3):
    """Calculate the dihedral angle (in radians) of four points"""
    # Vector calculation
    v0 = p1 - p0
    v1 = p2 - p1
    v2 = p3 - p2
    
    # Normal vector calculation
    n1 = np.cross(v0, v1)
    n2 = np.cross(v1, v2)
    
    # Normalization
    n1 /= np.linalg.norm(n1) + 1e-8
    n2 /= np.linalg.norm(n2) + 1e-8
    
    # Calculate the angle
    angle = atan2(
        np.dot(np.cross(n1, v1/(np.linalg.norm(v1)+1e-8)), n2),
        np.dot(n1, n2)
    )
    return [np.cos(angle),np.sin(angle)]

def compute_chi_features(res, chi_level):
    """Calculate sidechain dihedral angle features"""
    atom_names = CHI_ATOMS.get(chi_level, {}).get(res.resname, None)
    if not atom_names:
        return [0.0, 0.0]
    
    atoms = []
    valid = True
    for name in atom_names:
        if name in res:
            atoms.append(res[name].coord)
        else:
            valid = False
            break
    
    if valid and len(atoms) == 4:
        return compute_dihedral(*map(np.array, atoms))
    else:
        return [0.0, 0.0]

def is_hydrogen(atom):
    """Accurately determine if an atom is hydrogen based on element type"""
    return atom.element == "H"
def geometric_sidechain_length(res):
    ca = res['CA'].coord
    max_distance = 0.0
    for atom in res.get_atoms():
        atom_name = atom.get_name()
        if atom_name not in ['N', 'CA', 'C', 'O'] and not is_hydrogen(atom):
            distance = np.linalg.norm(atom.coord - ca)
            max_distance = max(max_distance, distance)
    return max_distance

def get_sidechain_direction(res):
    # CA → CB vector
    if 'CB' in res:
        ca = res['CA'].coord
        cb = res['CB'].coord
        direction = cb - ca
        direction /= np.linalg.norm(direction) + 1e-8
        return direction.tolist()
    else:  # Glycine has no sidechain
        return [0.0, 0.0, 0.0]
    

def energy_normalization(rosetta_energy):
    # Calculate mean
    mean = np.mean(rosetta_energy,axis=0, keepdims=True) 

    # Calculate variance
    var = np.var(rosetta_energy,axis=0, keepdims=True)  

    # Calculate standard deviation
    std = np.sqrt(var + 1e-8)  # Numerical stability

    # Normalize
    normalized = (rosetta_energy - mean) / std

    return normalized


def get_protein_feature(res_list,pose,res_energy,labels):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    res_energy = energy_normalization(res_energy)
    structure = {
        'name': "placeholder",
        'seq': "".join([three_to_one.get(res.resname, 'X') for res in res_list]),
        'features': pose,
        'res_energy': res_energy,
        'labels': labels,
        'res_features': [],
        'coords': []
    }
    sc_lengths = []
    all_features = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            res_coords.append(list(atom.coord))
        structure['coords'].append(res_coords)
        # Sidechain dihedral angle features (CHI1-CHI4)
        chi_features = []
        direction = []
        sc_length = []
        for chi_level in ["CHI1", "CHI2", "CHI3", "CHI4"]:
            chi_features.extend(compute_chi_features(res, chi_level))
        # Pad to 8 dimensions
        chi_features += [0.0] * (8 - len(chi_features))
        # Sidechain direction
        direction = get_sidechain_direction(res)
        # Sidechain length
        sc_length = geometric_sidechain_length(res)
        sc_lengths.append(sc_length)

        # Concatenate all features
        res_features = chi_features[:8] + direction
        all_features.append(res_features)
        
    features_to_normalize = np.array([sc_lengths]).T
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_to_normalize)
    # Concatenate all features
    all_features = np.array(all_features)
    normalized_features = np.array(normalized_features)
    structure['res_features'] = np.hstack([all_features, normalized_features])
    
    torch.set_num_threads(1)        # this reduce the overhead, and speed up the process for me.
    dataset = CustomProteinGraphDataset([structure])

    protein = dataset[0]
    # print(protein)
    x = (protein.x, protein.seq, protein.features, protein.protein_center, protein.node_s, protein.node_v, protein.edge_index, protein.edge_s, protein.edge_v,protein.edge_dis,protein.labels)
    return x

def batch_run(x, shared_dict):
    """
    Process protein structure files in batches to generate and save feature embeddings.
    
    Args:
        x (tuple): Contains three elements:
            - pdb (str): Protein identifier
            - proteinFile (str): Path to input protein structure file (.pdb or .pdb.gz)
            - toFile (str): Output path for processed features
        shared_dict (dict): Shared dictionary containing additional protein metadata:
            - Contains feature data under various keys
            - 'labels' key stores classification labels
            - 'protein_name' key stores identifier
    
    Returns:
        None: Output is written directly to disk
    """
    protein_dict = {}
    pdb, proteinFile, toFile = x
    # File existence check with validation
    # if os.path.exists(toFile):
    #     try:  # Verify if existing file is valid
    #         loaded_data = torch.load(toFile, weights_only=False)
    #         if loaded_data and isinstance(loaded_data, dict):
    #             return  # Valid file exists, skip processing
    #     except Exception as e:
    #         print(f"Error loading existing file {toFile}: {e}. Re-processing...")
    try:
        # Unified content reading from file
        if proteinFile.endswith('.pdb.gz'):
            with gzip.open(proteinFile, 'rb') as f:
                content = f.read().decode('utf-8')
        else:
            with open(proteinFile, 'r') as f:
                content = f.read()
        
        # Generate a file-like object for structure parsing
        file_obj = io.StringIO(content)
        parser = PDBParser(QUIET=True)
        s = parser.get_structure(pdb, file_obj)
        
        # Extract energy table data
        lines = content.split('\n')  # Split the file content into lines for further processing
        energy_data, pose = parse_energy_data(lines)
        
        # Process residue features
        res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
        
        # Validate presence in shared metadata
        if pdb not in shared_dict:
            return
        #[value for key, value in shared_dict[pdb].items() if key not in ['protein_name', 'labels']]
        if 'labels' in shared_dict[pdb] and shared_dict[pdb]['labels']:
            labels = shared_dict[pdb]['labels']
        else:
            # If the key 'labels' does not exist or is empty, assign it an empty list
            labels = ""
        protein_dict[pdb] = get_protein_feature(res_list,pose,energy_data,labels)
        torch.save(protein_dict, toFile)
    except Exception as e:
        print(f"Error: {e} in {pdb}")
        
def disable_logging():
    logging.getLogger().disabled = True

def process_and_save(pdb_input, protein_embedding_folder, dataset_type, batch_size=10000, num_workers=32):
    """
    Process protein data in parallel batches and save embeddings with metadata tracking.
    
    :param pdb_input: List of protein data entries (name, path, labels)
    :param protein_embedding_folder: Output directory for embeddings and metadata
    :param dataset_type: Type identifier for the dataset (e.g., 'train', 'test')
    :param batch_size: Number of proteins processed per batch
    :param num_workers: Number of parallel threads for data loading
    """
    protein_dict = {}
    dict_lock = threading.Lock()
    pbar = tqdm(total=len(pdb_input), desc=f"{dataset_type} Progress")
    data_lengths_file = Path(protein_embedding_folder) / 'data_length.csv'
    processed_first_batch = False
    # Clean existing data for this dataset type before processing
    def clean_existing_data():
        """Remove existing entries and files for current dataset type"""
        # Clean CSV entries
        if data_lengths_file.exists():
            try:
                # Read and filter out entries for current dataset type
                with open(data_lengths_file, 'r', encoding='utf-8') as f:
                    lines = [line for line in csv.reader(f) 
                            if not line[0].split('/')[-1].startswith(f'protein_embedding_{dataset_type}')]
                
                # Write filtered content back
                with open(data_lengths_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(lines)
            except Exception as e:
                print(f"Error cleaning CSV: {e}")
        # Clean existing PT files
        for pt_file in Path(protein_embedding_folder).glob(f'protein_embedding_{dataset_type}_*.pt'):
            try:
                pt_file.unlink()
                print(f"Removed old file: {pt_file}")
            except Exception as e:
                print(f"Error deleting {pt_file}: {e}")
    # Process data in batches
    for batch_idx, start in enumerate(range(0, len(pdb_input), batch_size)):
        # Clean data before first batch processing
        if not processed_first_batch:
            clean_existing_data()
            processed_first_batch = True
        end = start + batch_size
        batch_pdb_input = pdb_input[start:end]
        with ThreadPoolExecutor(max_workers=num_workers, initializer=disable_logging) as executor:
            futures = [executor.submit(partial(load_protein, 
                                             protein_embedding_folder=protein_embedding_folder), 
                                     pdb_inp[0]) for pdb_inp in batch_pdb_input]
            for future in as_completed(futures):
                try:
                    protein_data = future.result()
                    if protein_data:
                        with dict_lock:
                            protein_dict.update(protein_data)
                        pbar.update(1)
                except Exception as e:
                    print(f"Error processing task: {e}")
        if protein_dict:
            file_name = f'{protein_embedding_folder}/protein_embedding_{dataset_type}_{batch_idx+1}.pt'
            
            # Create CSV file if not exists
            write_header = not data_lengths_file.exists()
            with open(data_lengths_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(['file_path', 'length'])
                writer.writerow([file_name, len(protein_dict)])
            
            torch.save(protein_dict, file_name)
            protein_dict.clear()
    pbar.close()
        
def load_protein(pdb, protein_embedding_folder):
    """Load pre-computed protein embeddings from disk."""
    embedding_dir = protein_embedding_folder.parent
    try:
        return torch.load(f"{embedding_dir}/{pdb}.pt", 
                map_location=torch.device('cpu'),
                weights_only=False)
    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading {pdb}: {e}")
        return None


def filter_pdb_items(pdb_input, names_set):
    """Filter protein entries based on allowed names."""
    return [(item[0], item[1], item[2]) 
           for item in pdb_input 
           if item[0] in names_set]

def filter_pdb_items_parallel(pdb_input, names_set, num_processes=32):
    """Parallel implementation of protein filtering using multiprocessing."""
    chunk_size = len(pdb_input) // num_processes + 1
    
    with Pool(processes=num_processes) as pool:
        # Split data into chunks for parallel processing
        chunks = [pdb_input[i:i+chunk_size] 
                for i in range(0, len(pdb_input), chunk_size)]
        
        # Process chunks in parallel
        filtered_chunks = pool.map(partial(filter_pdb_items, 
                                         names_set=names_set), 
                                 chunks)
        
        # Combine results from all chunks
        return [item for sublist in filtered_chunks 
              for item in sublist]

def calculate_label_counts(data):
    """
    Calculate the sample count for each label in a multi-label dataset.
    
    :param data: DataFrame containing two columns: 'protein_name' and 'labels'
    :return: Dictionary with label counts {label: count}
    """
    label_counts = defaultdict(int)
    for labels in data['labels']:
        # Split labels by semicolon, treat as single label if no semicolon
        for label in labels.split(';'):
            label_counts[label.strip()] += 1
    return label_counts

def undersampling_multilabel(data: pd.DataFrame, valid_ratio=0.2, seed=0):
    """
    Perform undersampling on multi-label data while preserving original label format.
    
    :param data: DataFrame containing two columns: 'protein_name' and 'labels'
    :param valid_ratio: Proportion of data to allocate for validation set
    :param seed: Random seed for reproducibility
    :return: Tuple of (training_set DataFrame, validation_set DataFrame)
    """
    # Calculate initial label distribution
    label_counts = calculate_label_counts(data)

    # Determine sample allocation per label
    valid_label_counts = {label: int(count * valid_ratio) 
                         for label, count in label_counts.items()}
    train_label_counts = {label: count - valid_label_counts[label] 
                         for label, count in label_counts.items()}

    # Initialize containers for dataset splits
    train_set = []
    valid_set = []

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Shuffle dataset while maintaining row relationships
    shuffled_data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Expand multi-label entries into individual rows
    exploded_labels = shuffled_data.assign(
        labels=shuffled_data['labels'].str.split(';')
    ).explode('labels')
    exploded_labels['labels'] = exploded_labels['labels'].str.strip()
    
    # Create label-to-sample index mapping
    label_to_samples = exploded_labels.groupby('labels').groups

    # Allocate samples per label
    for label, count in train_label_counts.items():
        # Get indices of samples containing current label
        label_indices = label_to_samples[label]
        
        # Assign to training set
        train_indices = label_indices[:count]
        train_set.append(shuffled_data.loc[train_indices])
        
        # Remaining samples to validation set
        remaining_indices = label_indices[count:]
        valid_count = valid_label_counts[label]
        valid_indices = remaining_indices[:valid_count]
        valid_set.append(shuffled_data.loc[valid_indices])

    # Aggregate and deduplicate results
    train_set = pd.concat(train_set).drop_duplicates(
        subset='protein_name'
    ).reset_index(drop=True)
    valid_set = pd.concat(valid_set).drop_duplicates(
        subset='protein_name'
    ).reset_index(drop=True)
    
    # Final shuffle of datasets
    train_set = train_set.sample(frac=1, random_state=seed).reset_index(drop=True)
    valid_set = valid_set.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return train_set, valid_set

def process_dataset(args):
    # Create output directories
    Path(args.processed_dir).mkdir(parents=True, exist_ok=True)
    os.system(f"mkdir -p {args.processed_dir}")

    # Process labels
    label_path = Path(args.include_dir) / args.label_file
    processed_data = pd.read_csv(label_path).drop_duplicates('protein_name')
    
    # ================== Data Splitting ==================
    split_config = {'train': None, 'valid': None, 'test': None}
    
    if args.split_mode == 'train_split':
        # Standard stratified 8:1:1 split
        train_set, remaining = undersampling_multilabel(
            processed_data, valid_ratio=0.2, seed=args.seed
        )
        valid_set, test_set = undersampling_multilabel(
            remaining, valid_ratio=0.5, seed=args.seed
        )
        split_config.update({
            'train': train_set,
            'valid': valid_set,
            'test': test_set
        })
    else:
        # Full dataset mode
        target_split = args.split_mode.split('_')[-1]
        split_config[target_split] = processed_data
    # ================== Parallel Processing ==================
    with Manager() as manager:
        # Create shared dictionary for multiprocessing
        combined_data = pd.concat(
            [df for df in split_config.values() if df is not None]
        ).reset_index(drop=True).drop_duplicates('protein_name').set_index('protein_name', inplace=False)
        shared_dict = manager.dict(combined_data.to_dict('index'))
        # Process PDB files
        pdb_input = search_pdb_files(
            args.pdb_dir,
            include_file=label_path,
            to_file=Path(args.embedding_dir)
        )
        # Batch processing with progress tracking
        partial_batch_run = partial(batch_run, shared_dict=shared_dict)
        with Pool(args.num_workers) as pool:
            list(tqdm(pool.imap(partial_batch_run, pdb_input),total=len(pdb_input),desc='Processing PDB files'))
            
    # ================== Dataset Export ==================
    for split_name, split_data in split_config.items():
        if split_data is None or split_data.empty:
            continue
        protein_ids = set(split_data['protein_name'])
        filtered_files = filter_pdb_items_parallel(
            pdb_input, protein_ids, args.num_workers
        )
        process_and_save(
            filtered_files,
            protein_embedding_folder=Path(args.processed_dir),
            dataset_type=split_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein graph dataset processing')
    
    # Input/output parameters
    parser.add_argument('--pdb_dir', type=str, default='./rosetta/pdb_rosetta',
                        help='Directory containing PDB files')
    parser.add_argument('--include_dir', type=str, default='./PDB_file',
                        help='Directory containing filter/label files')
    parser.add_argument('--embedding_dir', type=str, default='./access/gvp_protein_embedding',
                        help='Base directory for embedding outputs')
    parser.add_argument('--processed_dir', type=str, default='./access/gvp_protein_embedding/processed',  
                        help='Processed dataset output directory')
    
    # File parameters
    parser.add_argument('--label_file', type=str, default='train_labels_all_0001.csv',
                        help='Name of label CSV file in include_dir')
    parser.add_argument('--split_mode', type=str, 
                      choices=['train_split', 'all_train', 'all_valid', 'all_test'],
                      default='all_test',
                      help='''Data splitting strategy:
                          train_split - Standard 8:1:1 split
                          all_* - Use full dataset for specified split''')

    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Number of samples per batch')
    parser.add_argument('--num_workers', type=int, default=64,
                        help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()
    
    process_dataset(args)