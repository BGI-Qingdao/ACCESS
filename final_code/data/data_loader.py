import csv
import os
import re
import numpy as np
import torch
from bisect import bisect_left
from natsort import natsorted, ns
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from torch.utils.data import RandomSampler


class FileWiseSubsetRandomSampler(RandomSampler):
    """
    A custom sampler that performs random sampling within each file without crossing file boundaries.
    """
    def __init__(self, dataset, replacement=False, num_samples=None):
        """
        Initialize the FileWiseSubsetRandomSampler.
        
        Args:
            dataset: Dataset object containing file lengths.
            replacement (bool): Whether to sample with replacement.
            num_samples (int, optional): Number of samples to draw. Defaults to total size of the dataset.
        """
        self.dataset = dataset
        self.replacement = replacement
        self.file_sizes = list(dataset.file_lengths.values())
        self.total_size = sum(self.file_sizes)  # Total number of samples
        if num_samples is None:
            num_samples = self.total_size
        self._num_samples = num_samples

    @property
    def num_samples(self):
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("num_samples should be a positive integer")
        self._num_samples = value

    def __iter__(self):
        file_idx = 0
        samples_drawn = 0
        current_offset = 0  # Current file's starting offset

        while samples_drawn < self.num_samples:
            file_size = self.file_sizes[file_idx]
            # Calculate the number of samples to draw from the current file
            samples_to_draw = min(self.num_samples - samples_drawn, file_size)

            if self.replacement:
                # Sample with replacement
                indices = torch.randint(high=file_size, size=(samples_to_draw,), dtype=torch.int64)
            else:
                # Sample without replacement
                indices = torch.randperm(file_size)[:samples_to_draw]

            # Adjust indices by adding the current file's offset
            indices += current_offset
            yield from indices.tolist()

            samples_drawn += samples_to_draw
            file_idx += 1
            current_offset += file_size  # Update offset

            # If all files are exhausted and more samples are needed, restart from the first file
            if file_idx >= len(self.file_sizes) and samples_drawn < self.num_samples:
                file_idx = 0
                current_offset = 0  # Reset offset

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        else:
            return self.total_size

def get_data_files(self, root_dir, type):
    files = [f for f in os.listdir(root_dir) if f.startswith(f'protein_embedding_{type}_')]
    sorted_files = natsorted(files, alg=ns.IGNORECASE)
    return [os.path.join(root_dir, f) for f in sorted_files]

torch.set_num_threads(1)


def get_keepNode(n_node, use_whole_protein):
    if use_whole_protein:
        keepNode = np.ones(n_node, dtype=bool)
    return keepNode
def get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, protein_edge_v,protein_edge_dis, keepNode):
    # protein
    new_node_index = np.cumsum(keepNode) - 1
    keepEdge = keepNode[protein_edge_index].min(axis=0)
    new_edge_inex = new_node_index[protein_edge_index]
    input_edge_idx = torch.tensor(new_edge_inex[:, keepEdge], dtype=torch.long)
    input_protein_edge_s = protein_edge_s[keepEdge]
    input_protein_edge_v = protein_edge_v[keepEdge]
    input_protein_edge_dis = protein_edge_dis[keepEdge]
    return input_edge_idx, input_protein_edge_s, input_protein_edge_v, input_protein_edge_dis

def construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                                  protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,protein_edge_dis,use_whole_protein=False):
    n_node = protein_node_xyz.shape[0]
    keepNode = get_keepNode( n_node, use_whole_protein)

    if keepNode.sum() < 5:
        # if only include less than 5 residues, simply add first 100 residues.
        keepNode[:100] = True
    input_node_xyz = protein_node_xyz[keepNode]
    input_edge_idx, input_protein_edge_s, input_protein_edge_v, input_protein_edge_dis = get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, protein_edge_v,protein_edge_dis, keepNode)
    # construct graph data.
    data = HeteroData()
    # additional information. keep records.
    data.node_xyz = input_node_xyz


    data.seq = protein_seq[keepNode]
    data['protein'].node_s = protein_node_s[keepNode] # [num_protein_nodes, num_protein_feautre]
    data['protein'].node_v = protein_node_v[keepNode]
    data['protein', 'p2p', 'protein'].edge_index = input_edge_idx
    data['protein', 'p2p', 'protein'].edge_s = input_protein_edge_s
    data['protein', 'p2p', 'protein'].edge_v = input_protein_edge_v
    data['protein', 'p2p', 'protein'].edge_dis = input_protein_edge_dis

    return data, input_node_xyz, keepNode

class ACCSEE_DataSet(Dataset):
    """
    A custom dataset class for loading and processing protein data for Graph Neural Networks (GNNs).
    This dataset assumes that protein data is stored in multiple files, each containing embeddings for a subset of proteins.
    The dataset also loads metadata from a CSV file that contains file paths and their corresponding lengths.
    """
    
    def __init__(self, root, csv_file_path,protein_dict=None, proteinMode=0, 
                add_noise_to_com=None, pocket_radius=20, contactCutoff=8.0, predDis=True, shake_nodes=None,
                 transform=None, pre_transform=None, pre_filter=None,type = 'train'):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data_files = self.get_data_files(root,type)#得到了文件目录
        print(self.data_files)
        self.add_noise_to_com = add_noise_to_com
        self.proteinMode = proteinMode
        self.pocket_radius = pocket_radius
        self.contactCutoff = contactCutoff
        self.predDis = predDis
        self.shake_nodes = shake_nodes
        self.file_lengths = self._pre_load_file_lengths(csv_file_path, type)
        self.protein_dict = protein_dict
        self.protein_dict_file = None
        self.index_to_protein_name = []
        self.cumulative_sizes = []
        cumulative_size = 0
        for size in self.file_lengths.values():
            cumulative_size += size
            self.cumulative_sizes.append(cumulative_size)

    def get_data_files(self, root_dir,type):
        files = [f for f in os.listdir(root_dir) if f.startswith(f'protein_embedding_{type}_')]
        sorted_files = natsorted(files, alg=ns.IGNORECASE)
        return [os.path.join(root_dir, f) for f in sorted_files]
    
    def _pre_load_file_lengths(self, csv_file_path, type):
        """Preload file paths and their lengths from a CSV file.
        
        Args:
            csv_file_path (str): Path to the CSV file containing file metadata
            type (str): Data type filter ('train', 'valid', or 'test')
            
        Returns:
            dict: Dictionary mapping file paths to their lengths
        """
        file_lengths = {}
        try:
            # Read CSV with tab delimiter (adjust if using different separator)
            with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    file_path = row['file_path']
                    length = int(row['length'])
                    
                    # Filter files using regex pattern: protein_embedding_{type}_*.pt
                    match = re.search(
                        r'protein_embedding_(train|valid|test)_\d+\.pt', 
                        file_path
                    )
                    if match and match.group(1) == type:
                        file_lengths[file_path] = length
                        
        except Exception as e:
            print(f"Error loading file lengths from {csv_file_path}: {e}")
            
        return file_lengths

    def _get_file_index(self, idx):
        """Map global dataset index to corresponding file and local index.
        
        Args:
            idx (int): Global dataset index (0 <= idx < total_samples)
            
        Returns:
            tuple: (file_path, local_index_within_file)
            
        Raises:
            IndexError: If input index is out of valid range
        """
        # Validate index range
        if idx < 0 or idx >= self.cumulative_sizes[-1]:
            raise IndexError("Index out of range")
            
        # Binary search to find containing file
        # Note: bisect_left returns first entry >= target, hence idx+1
        file_idx = bisect_left(self.cumulative_sizes, idx + 1)
        
        # Calculate local index within the file
        local_idx = idx if file_idx == 0 else idx - self.cumulative_sizes[file_idx-1]
        
        # Get corresponding file path
        file_path = self.data_files[file_idx]
        
        return file_path, local_idx
    
    def _load_protein_dict(self, file_path):
        # Load protein dictionary from file
        return torch.load(file_path, map_location='cpu', weights_only=False)
    
    def len(self):
        # Calculate total length by summing lengths of all files
        # return sum(len(torch.load(file, map_location='cpu', weights_only=False)) for file in self.data_files)
        if not self.cumulative_sizes:
            return 0
        return self.cumulative_sizes[-1]
    
    def get(self, idx):
        # Determine which file this index belongs to
        file_idx, idx = self._get_file_index(idx)
        if self.protein_dict_file != file_idx:

            self.protein_dict_file = file_idx
            self.protein_dict = self._load_protein_dict(file_idx)
            self.index_to_protein_name=list(self.protein_dict.keys())
        protein_name = self.index_to_protein_name[idx]
        protein_node_xyz, protein_seq,protein_features, protein_center,protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,protein_edge_dis,protein_labels = self.protein_dict[protein_name]
        data, input_node_list, keepNode = construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                    protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,protein_edge_dis,use_whole_protein=True)
        data.y = protein_labels
        data.protein_name = protein_name

        data.features = torch.tensor(protein_features, dtype=torch.float)
        return data   

if __name__ == '__main__':
    root_dir = "./complete_code/gvp_protein_embedding_54w_complemented/processed"
    csv_file_path = root_dir + '/data_length.csv'
    dataset = ACCSEE_DataSet(root_dir,csv_file_path,type='train') 
    train_sampler = FileWiseSubsetRandomSampler(dataset, replacement=False)
    train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
    # Fetch the first sample after random sampling
    for batch in train_loader:
        first_sample = batch  
        break  
    print("Total number of samples in the dataset :",dataset.len())
    print("The first sample in the original dataset order:", dataset[0])
    print("The first sample after random sampling:", first_sample)  