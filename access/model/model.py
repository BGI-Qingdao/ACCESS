import math
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from itertools import combinations
from torch_geometric.nn import GCNConv, GATConv, global_max_pool
from torch_geometric.utils import to_dense_batch
from torch.nn.init import xavier_uniform_
from gvp import GVP, GVPConvLayer, LayerNorm
import esm

    
def flatten_dense_to_sparse(x_dense, mask):
    """
    Converts a dense tensor to sparse format using a boolean mask.
    
    Args:
        x_dense: Dense tensor of shape [batch_size, max_seq_len, feature_dim]
        mask: Boolean mask tensor of shape [batch_size, max_seq_len], where True indicates valid elements
    
    Returns:
        Sparse tensor of shape [num_valid_elements, feature_dim] containing only valid entries
    
    Example:
        >>> dense = torch.randn(2, 3, 4)
        >>> mask = torch.BoolTensor([[True, False, True], [False, True, False]])
        >>> sparse = flatten_dense_to_sparse(dense, mask)
        >>> sparse.shape
        torch.Size([3, 4])
    """
    # Expand mask to match feature dimensions for element-wise selection
    mask_expanded = mask.unsqueeze(-1).expand_as(x_dense)
    # Select valid elements and preserve feature dimensionality
    valid_elements = x_dense[mask_expanded].view(-1, x_dense.size(-1))
    return valid_elements

class GVP_embedding(nn.Module):
    '''
    Modified based on https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]
    
    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.3):

        super(GVP_embedding, self).__init__()
        if seq_in:
            self.esm_model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.fc = nn.Linear(320, 128)
            self.layer_norm = nn.LayerNorm(128)
            node_in_dim = (node_in_dim[0] + 128, node_in_dim[1])
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate) 
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, h_V, edge_index, h_E, seq, batch):      
        """
        Forward pass of the model, incorporating ESM embeddings for sequence information.

        Args:
            h_V (tuple): Tuple of node embeddings (s, V).
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            h_E (tuple): Tuple of edge embeddings (s, V).
            seq (torch.Tensor, optional): Sequence information to be embedded and appended to node features. Defaults to None.
            batch (torch.Tensor): Batch indices for the nodes.

        Returns:
            tuple: Transformed node embeddings and CLS embeddings.
        """
        if seq is not None:
            self.esm_model.eval()
            with torch.no_grad():
                # Convert sequence to dense format and get mask
                seq_dense, seq_mask = to_dense_batch(seq, batch, fill_value=self.alphabet.padding_idx)  
                # Add CLS token to the beginning of each sequence
                cls_token_id = self.alphabet.cls_idx
                seq_dense = torch.cat([torch.full((seq_dense.size(0), 1), cls_token_id, dtype=seq_dense.dtype, device=seq_dense.device), seq_dense], dim=1)             
                # Get embeddings from ESM model
                results = self.esm_model(seq_dense, repr_layers=[6])
                token_representations = results['representations'][6]
                
            # Process embeddings
            seq_embeddings = self.fc(token_representations)
            seq_embeddings = self.layer_norm(seq_embeddings)
            cls_embeddings = seq_embeddings[:, 0, :]
            seq_embeddings = seq_embeddings[:, 1:, :] * seq_mask.unsqueeze(-1).float()
            # Map embeddings back to sparse format
            seq_embeddings_sparse = flatten_dense_to_sparse(seq_embeddings, seq_mask)
            # Combine ESM embeddings with node features           
            h_V = (torch.cat([h_V[0],seq_embeddings_sparse], dim=-1), h_V[1])#[:, :6],rosetta_normalized

        # self.last_h_V = h_V[0].clone()
        def print_grad(grad):
            self.last_h_V_grad = grad.clone()
        if h_V[0].requires_grad:
            h_V[0].register_hook(print_grad)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)
        return out,cls_embeddings
    
class GCNEmbedding(nn.Module):
    """
    Graph Convolutional Network for node embeddings with label-specific transformations.
    """
    def __init__(self, num_labels, embedding_dim):
        """
        Initialize the GCNEmbedding module.
        
        Args:
            num_labels (int): Number of labels.
            embedding_dim (int): Embedding dimension.
        """
        super(GCNEmbedding, self).__init__()
        self.embed_dim = embedding_dim
        self.layernorm = torch.nn.LayerNorm(embedding_dim)
        # Graph convolutional layers
        self.conv1 = GCNConv(embedding_dim, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)
        # Label-specific transformation parameters
        self.W1 = nn.Parameter(torch.Tensor(num_labels, embedding_dim, embedding_dim))
        self.b1 = nn.Parameter(torch.Tensor(num_labels, embedding_dim))
        self.W2 = nn.Parameter(torch.Tensor(num_labels, embedding_dim, embedding_dim))
        self.b2 = nn.Parameter(torch.Tensor(num_labels, embedding_dim))
        
        self._init_parameters()

    def _init_parameters(self):
        """Initialize model parameters."""
        xavier_uniform_(self.W1.view(-1, self.embed_dim))
        xavier_uniform_(self.W2.view(-1, self.embed_dim))
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)
        
    def forward(self,x, edge_index, edge_weight):
        """
        Forward pass through the GCN layers with label-specific transformations.
        
        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            edge_weight (torch.Tensor): Edge weights.
            
        Returns:
            torch.Tensor: Transformed node embeddings.
        """
        # Graph convolution layers
        x = self.conv1(x, edge_index,edge_weight=edge_weight)
        x = torch.selu(x)
        x = self.conv2(x, edge_index,edge_weight=edge_weight)
        x = torch.selu(x)
        # Expand dimensions for label-specific transformations
        x_expanded = x.unsqueeze(1)
        
        # First label-specific transformation
        transformed = torch.einsum('nid,ndj->nij', x_expanded, self.W1).squeeze(1)+ self.b1
        transformed = self.layernorm(transformed)
        # Second label-specific transformation
        transformed = torch.einsum('nid,ndj->nij', transformed.unsqueeze(1), self.W2).squeeze(1) + self.b2
        
        return transformed   
            
class ACCESS(torch.nn.Module):
    """
    ACCESS Model: A hierarchical graph neural network for protein function prediction.
    """    
    def __init__(self,ec_file,emb_dim=64,num_node_features=128,protein_feature_count=30, num_heads=8, hidden_channels=128, embedding_channels=128, dropout=0.2):
        """
        Initialize the ACCESS model.

        Args:
            ec_file (str): Path to the EC labels file.
            emb_dim (int, optional): Embedding dimension. Defaults to 64.
            num_node_features (int, optional): Number of node features. Defaults to 128.
            protein_feature_count (int, optional): Number of protein features. Defaults to 30.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            hidden_channels (int, optional): Number of hidden channels. Defaults to 128.
            embedding_channels (int, optional): Number of embedding channels. Defaults to 128.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm2 = torch.nn.LayerNorm(protein_feature_count)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.SELU = nn.SELU()
        self.LeakyReLU = nn.LeakyReLU()
        self.emb_dim = emb_dim
        self.ec_file = ec_file
        self.cooc_counts = defaultdict(int)
        self._init_ec_embedding()
        self.ec_cache = {} # Cache for parsed EC numbers
        
        self.conv_protein = GVP_embedding((38, 3), (embedding_channels, 16), 
                                            (32, 1), (32, 1), seq_in=True,drop_rate = dropout)
        self.gat_layers = nn.ModuleList([
                GATConv(num_node_features, hidden_channels // num_heads, heads=num_heads, dropout=dropout,edge_dim=32)
            for _ in range(1)])
        self.fc2 = nn.Linear(embedding_channels+embedding_channels+ protein_feature_count, 128) 
        self.fc3 = nn.Linear(128, 64)  
        self.label_optimizer = GCNEmbedding(len(self.node_dict), emb_dim)

                
    def _validate_ec(self, ec):
        """Validate the format of an EC number."""
        
        parts = ec.split('.')
        # Check if the EC number has between 1 and 4 parts
        if not (1 <= len(parts) <= 4):
            return False
        
        # Validate each part of the EC number
        for i, p in enumerate(parts):
            # First part must be a digit
            if i == 0 and not p.isdigit():
                return False
            # Other parts can be digits, letters, or '-'
            if i > 0 and p != '-' and not (p.isdigit() or p[0].isalpha()):
                return False
        
        return True

    def _parse_ec(self):
        """Parse the EC file and build a hierarchical structure."""
        all_ec = set()
        ec_counts = defaultdict(int)
        parent_child_counts = defaultdict(lambda: defaultdict(int))
        full_hierarchy = defaultdict(set)

        def normalize_ec(parts, depth):
            """Generate a four-layer standardized EC node, filling missing parts with '-'."""
            padded = (parts[:depth] + ['-']*(4))[:4] 
            return '.'.join(padded)
        
        # Read the EC file
        df = pd.read_csv(self.ec_file)

        for index, row in df.iterrows():
            ec_group = row['labels'].split(';')
            valid_ec = [ec.strip() for ec in ec_group if self._validate_ec(ec)]

            # Record co-occurrence relationships
            for ec1, ec2 in combinations(valid_ec, 2):
                self.cooc_counts[(ec1, ec2)] += 1
                self.cooc_counts[(ec2, ec1)] += 1

            # Build hierarchical structure
            for ec in valid_ec:

                ec_counts[ec] += 1  # Increment EC occurrence count
                parts = ec.split('.')
                k = 0
                for i in reversed(range(len(parts))):
                    if parts[i] != '-':
                        k = i + 1  # Effective layer is the index of the last non-'-' part + 1
                        break
                #  Generate standardized nodes for each layer
                for depth in range(1, k+1):
                    current_node = normalize_ec(parts, depth)    
                    all_ec.add(current_node)                    
                    full_hierarchy[depth].add(current_node)

                    # Link to parent node
                    if depth > 1:
                        parent_node = normalize_ec(parts, depth-1)
                        # all_ec.add(parent_node)
                        parent_child_counts[parent_node][current_node] += 1
                        # full_hierarchy[depth-1].add(parent_node)

        # Update parent node occurrence counts
        for parent in parent_child_counts:
            if parent in ec_counts:
                ec_counts[parent] += sum(parent_child_counts[parent].values())
            else:
                ec_counts[parent] = sum(parent_child_counts[parent].values())

        return full_hierarchy, all_ec, ec_counts, parent_child_counts
    
    def _build_adjacency(self):
        """
        Build the adjacency matrix for the EC hierarchy.
        """
        adj_matrix = torch.eye(self.num_ec_nodes, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Build adjacency matrix using statistical information
        for parent, children in self.parent_child_counts.items():
            total = sum(children.values())
            parent_idx = self.node_dict[parent]
            
            for child, count in children.items():
                child_idx = self.node_dict[child]
                # Parent to child weight based on distribution
                adj_matrix[parent_idx, child_idx] += count / total  
                # Child to parent fixed weight
                adj_matrix[child_idx, parent_idx] += 1.0  
        
        # Add co-occurrence edges
        for (ec1, ec2), count in self.cooc_counts.items():
            i = self.node_dict[ec1]
            j = self.node_dict[ec2]
            # Scale using sigmoid function
            exp_weight = min(1.0, 1 / (1 + math.exp(-count / 100)))  
            adj_matrix[i, j] += exp_weight
            adj_matrix[j, i] += exp_weight
        
        # Normalization
        row_sum = adj_matrix.sum(1).clamp(min=1)
        degree_mat = torch.diag(1.0 / torch.sqrt(row_sum))
        self.adj_matrix = degree_mat @ adj_matrix @ degree_mat
        
    def _get_ec_embeddings(self,device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        # Initialize embeddings
        emb = self.ec_embedding(torch.tensor(list(self.node_dict.values()),device = device))
        return emb.detach() 
          
    def _init_ec_embedding(self,margin_config={'parent':0.02, 'sibling':0.1, 'unrelated':0.2},device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """Initialize EC hierarchy graph embeddings."""
        # Parse EC file to build graph structure
        self.ec_hierarchy, self.all_ec, self.ec_counts, self.parent_child_counts = self._parse_ec()
        self.node_dict = {node:i for i, node in enumerate(sorted(self.all_ec))}
        self.reverse_dict = {v:k for k,v in self.node_dict.items()}
        self.num_ec_nodes = len(self.node_dict)
        self.margin = margin_config
        # Build adjacency matrix and register as buffer
        self._build_adjacency()
        # Trainable initial embeddings
        self.ec_embedding = nn.Embedding(self.num_ec_nodes, self.emb_dim)
        nn.init.xavier_uniform_(self.ec_embedding.weight)
        self.ec_embedding.to(device)
        self.ec_embeddings = self.ec_embedding(torch.tensor(list(self.node_dict.values()),device=device))

    def get_embedding(self, ec_number, device='cpu'):
        """Retrieve the embedding vector for a given EC number."""
        # Validate EC number format
        if not self._validate_ec(ec_number):
            raise ValueError(f"Invalid EC number: {ec_number}")
        
        # Check if EC number exists in the node dictionary
        if ec_number not in self.node_dict:
            return torch.zeros(self.emb_dim, device=device)
        
        ec_index = self.node_dict[ec_number]
        
        # Return the corresponding embedding
        return self.ec_embeddings[ec_index].to(device)
             
    def get_embeddings(self, ec_list, device='cpu'):
        """Retrieve embedding vectors for a list of EC numbers."""
        batch_embeddings = []
        
        for ec_string in ec_list:
            ec_numbers = ec_string.split(';')
            embedding_list = []
            for ec in ec_numbers:
                ec = ec.strip()
                if ec and ec in self.node_dict:
                    # If EC exists, get its embedding
                    embedding_list.append(self.ec_embeddings[self.node_dict[ec]].to(device))
                else:
                    # If EC does not exist, use a random known embedding
                    random_ec = random.randint(0, self.ec_embeddings.size(0) - 1)
                    random_embedding = self.ec_embeddings[random_ec].to(device)
                    embedding_list.append(random_embedding)
            batch_embeddings.append(embedding_list)
        return batch_embeddings
    
    def _parse_level_ec(self, ec):
        """Parse the hierarchical structure of an EC number."""
        if ec in self.ec_cache:
            return self.ec_cache[ec]

        parts = [part for part in ec.split('.') if part != '-']
        parents = []
        current = []
        for p in parts:
            current.append(p)
            parent = '.'.join(current + ['-'] * (4 - len(current)))
            parents.append(parent)
        parents = parents[:-1]  # Exclude itself
        
        self.ec_cache[ec] = {
            'parts': parts,
            'parents': parents,
            'depth': len(parts)
        }
        return self.ec_cache[ec]
    
    def _get_parents(self, ec):
            """Dynamically generate all parent nodes for a given EC label."""
            parsed = self._parse_level_ec(ec)
            return [p for p in parsed['parents'] if p in self.node_dict]

    def _get_siblings(self, ec):
        """Generate sibling nodes for a given EC label."""
        parsed = self._parse_level_ec(ec)
        current_depth = parsed['depth']
        
        # Handle special case for root node (0.-.-.-)
        if current_depth == 1:
            if ec == '0.-.-.-':
                return []  # Return empty list for root node
            else:
                # Generate siblings for non-root level 1 nodes
                brothers = [f'{i}.-.-.-' for i in range(1, 8) if f'{i}.-.-.-' != ec]
                return brothers
        else:
            # Identify valid parents that exist in the node dictionary
            valid_parents = [p for p in parsed['parents'] if p in self.node_dict]
            if not valid_parents:
                return []
            
            # Use the nearest parent as the reference
            parent = valid_parents[-1]
            parent_depth = self._parse_level_ec(parent)['depth']
            
            # Define the target depth for siblings
            target_depth = parent_depth + 1
            sibling_prefix = '.'.join(parent.split('.')[:parent_depth]) + '.'
            
            # Efficiently filter sibling nodes
            siblings = []
            for candidate in self.all_ec:
                #  Skip the node itself
                if candidate == ec:
                    continue

                candidate_depth = self._parse_level_ec(candidate)['depth']
                if candidate_depth != target_depth:
                    continue  
                
                if candidate.startswith(sibling_prefix):
          
                    if (ec, candidate) not in self.cooc_counts:
                        siblings.append(candidate)            
            
            return siblings    
    def _sample_hard_negatives(self, pred, true_ecs, projected_emb, top_k=5):
        """Sample hard negatives with co-occurrence filtering."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Calculate Euclidean distances to all ECs
        all_dists = torch.norm(projected_emb - pred, p=2, dim=1)  # (num_classes,)
        
        # Identify positive indices and minimum positive distance
        if true_ecs in self.node_dict:
            pos_indices = self.node_dict[true_ecs]
        else:
            return  (
                [],             
                torch.tensor([], dtype=torch.float32, device=device),  
                torch.tensor([], dtype=torch.float32, device=device)   
            )
        min_pos_dist = all_dists[pos_indices].min() if pos_indices else torch.tensor(0.0)
        
        # Initialize negative mask
        neg_mask = torch.ones_like(all_dists, dtype=torch.bool)
        
        # Exclude positives
        neg_mask[pos_indices] = False
        
        # Exclude co-occurring ECs
        cooc_ecs = set()

        for (ec1, ec2) in self.cooc_counts.keys():
            if ec1 in true_ecs:# and count > cooc_threshold:
                cooc_ecs.add(ec2)
        
        cooc_indices = [self.node_dict[ec] for ec in cooc_ecs]
        neg_mask[cooc_indices] = False
        
        # Hierarchical margin calculation
        candidate_indices = torch.where(neg_mask)[0]
        parents = self._get_parents(true_ecs)
        siblings = self._get_siblings(true_ecs)
        margin_rules = torch.full_like(all_dists, self.margin['unrelated'], device=device)
        for idx in candidate_indices:
            ec_neg = self.reverse_dict[idx.item()]
            if ec_neg in parents:
                margin_rules[idx] = self.margin['parent']
            elif ec_neg in siblings:
                margin_rules[idx] = self.margin['sibling']
        # Apply margin filtering        
        valid_mask = all_dists[candidate_indices] < (min_pos_dist + margin_rules[candidate_indices])
        final_candidates = candidate_indices[valid_mask]
        

        # Select top-k hard negatives
        if len(final_candidates) > 0:
            _, selected = torch.topk(all_dists[final_candidates], 
                                k=min(top_k, len(final_candidates)), 
                                largest=False)
            negative_labels = [self.reverse_dict[idx.item()] for idx in final_candidates[selected]]
            return negative_labels,all_dists[final_candidates[selected]], margin_rules[final_candidates[selected]]
        else:
            return  (
                [],             
                torch.tensor([], dtype=torch.float32, device=device),  
                torch.tensor([], dtype=torch.float32, device=device)  
            )
    def _precompute_hard_negatives(self, preds, ec_batch, projected_embs):
        """Precompute hard negatives for a batch of predictions."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Calculate maximum ECs per batch
        max_ec_per_batch = max(len(ec.strip().split(';')) for ec in ec_batch)
        max_negs_per_batch = 0
        batch_data = []
        
        for idx in range(preds.size(0)):
            if not ec_batch[idx].strip():
                ec_data = [None] * max_ec_per_batch
            else:
                ec_list = ec_batch[idx].strip().split(';')[:max_ec_per_batch]  
                ec_data = [None] * max_ec_per_batch 
            
                for ec_idx, ec in enumerate(ec_list):
                    _, Dn, margins = self._sample_hard_negatives(preds[idx], ec, projected_embs)
                    valid_neg_num = Dn.size(0)
                    
                    if valid_neg_num > 0:
                        ec_data[ec_idx] = (Dn, margins)
                        max_negs_per_batch = max(max_negs_per_batch, valid_neg_num)
            
            batch_data.append(ec_data)
        
        # Ensure at least one negative sample
        max_negs_per_batch = max(max_negs_per_batch, 1)
        
        # Initialize tensors
        Dn_batch = torch.full(
            (len(preds), max_ec_per_batch, max_negs_per_batch),
            fill_value=-1e9,
            device=device
        )
        margin_batch = torch.full_like(Dn_batch, self.margin['unrelated'], device=device)
        valid_mask = torch.zeros_like(Dn_batch, dtype=torch.bool, device=device)
        
        # Populate tensors
        for idx in range(len(preds)):
            ec_data = batch_data[idx]
            for ec_idx in range(max_ec_per_batch):
                data = ec_data[ec_idx]
                if data is None:
                    continue  
                
                Dn, margins = data
                valid_neg_num = Dn.size(0)
                Dn_batch[idx, ec_idx, :valid_neg_num] = Dn
                margin_batch[idx, ec_idx, :valid_neg_num] = margins
                valid_mask[idx, ec_idx, :valid_neg_num] = True
        
        return {
            'Dn': Dn_batch,
            'margins': margin_batch,
            'mask': valid_mask,
            'negs_exist': valid_mask.any(dim=-1).any(dim=-1)
        }        
           
                   
    def forward(self, data):
        """
        Forward pass of the model, processing protein graph data and generating predictions.

        Args:
            data (torch_geometric.data.Data): Input graph data containing protein nodes and edges.

        Returns:
            dict: Dictionary containing model outputs, including predicted features and hard negatives.
        """
        nodes = (data['protein']['node_s'], data['protein']['node_v'])
        edges = (data[("protein", "p2p", "protein")]["edge_s"], data[("protein", "p2p", "protein")]["edge_v"])
        protein_batch = data['protein'].batch
        edge_index =  data[("protein", "p2p", "protein")].edge_index
        # edge_dis = data[("protein", "p2p", "protein")].edge_dis
        unique = torch.unique(protein_batch)
        protein_feature_count = 30
        features_batch = torch.repeat_interleave(unique, repeats=protein_feature_count)
        features = data.features
        protein_out,cls_embeddings = self.conv_protein(nodes, edge_index, edges, data.seq,protein_batch)
        protein_out = self.layernorm(protein_out)
        for gat_layer in self.gat_layers:
            residual = protein_out
            protein_out = gat_layer(protein_out, edge_index, edges[0])
            protein_out = protein_out + residual
            protein_out = self.SELU(self.layernorm(protein_out))
        protein_out = global_max_pool(protein_out, protein_batch)
        
        p_features_batched, p_features_mask = to_dense_batch(features, features_batch)
        p_features_batched = self.layernorm2(p_features_batched)
        combined_features = torch.cat((protein_out, cls_embeddings, p_features_batched), dim=1)
        combined_features = self.fc2(combined_features)
        
        pred_features = self.layernorm(combined_features)
        pred_features = self.fc3(pred_features)
        pred_features = F.normalize(pred_features, p=2, dim=1)

        edge_indices = torch.nonzero(self.adj_matrix, as_tuple=False).t()
        edge_weights = self.adj_matrix[edge_indices[0], edge_indices[1]]
        self.ec_embeddings = self.label_optimizer(self.ec_embeddings.detach(),edge_index=edge_indices, edge_weight=edge_weights)
        self.ec_embeddings = F.normalize(self.ec_embeddings, p=2, dim=1) 

        hard_negatives = self._precompute_hard_negatives(pred_features,data.y,self.ec_embeddings)
        
        return {
            'pred': pred_features,
            'projected_embs': self.ec_embeddings,
            'hard_negatives': hard_negatives
        }

    
def get_model(device,ec_file,dropout=None):
    model = ACCESS(ec_file = ec_file,dropout = dropout).to(device)
    return model