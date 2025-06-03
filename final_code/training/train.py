import argparse
from collections import defaultdict
import csv
import numpy as np
import os
import pandas as pd
import random
import torch
import time
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from scipy.stats import iqr
from sklearn.metrics import (f1_score,accuracy_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import MultiLabelBinarizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.data_loader import ACCSEE_DataSet,FileWiseSubsetRandomSampler
from model.model import *
import logging

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def maximum_separation(distances,first_grad):
    """
    Automatic threshold separation algorithm based on gradient features.
    
    Args:
        distances (list): Sorted list of distances (ascending order).
        first_grad (bool, optional): Whether to use the first gradient for optimization. Defaults to True.
        
    Returns:
        tuple: (Separation point index, threshold parameters used)
    """
    opt = 0 if first_grad else -1
    dist_arr = np.array(distances, dtype=np.float32)
    
    # Calculate gradient features
    gradients = dist_arr[1:] - dist_arr[:-1]
    denominator = np.where(dist_arr[:-1] < 1e-6, 1e-6, dist_arr[:-1])
    rel_gradients = gradients / denominator

    # Automatically calculate thresholds using robust statistics
    alpha = np.median(rel_gradients) + 1.5 * iqr(rel_gradients)  # Relative gradient threshold
    beta = np.median(gradients) + 1.5 * iqr(gradients)          # Absolute gradient threshold
    
    # Detect candidate points
    candidate_mask = (rel_gradients >= alpha) & (gradients >= beta)
    candidate_indices = np.where(candidate_mask)[0]

    # Multi-level decision strategy
    if len(candidate_indices) > 0:
        best_idx = candidate_indices[opt]
    else:
        # Fallback strategy: find the largest gradient jump
        best_idx = np.argmax(gradients)
        
    if best_idx >= 5:
        best_idx = 0
    return best_idx  

def filter_ec_numbers(ec_list):
    """
    Filter EC numbers by removing parent nodes if their deeper child nodes exist in the list.
    
    Args:
        ec_list (list): List of EC numbers to filter.
        
    Returns:
        list: Filtered list of EC numbers with redundant parent nodes removed.
    """
    zero_ec = '0.-.-.-'
    if zero_ec in ec_list:
        # Find the index of the first occurrence of '0.-.-.-'
        zero_index = ec_list.index(zero_ec)
        
        # If '0.-.-.-' is the first element, remove all subsequent elements
        if zero_index == 0:
            ec_list = [zero_ec]
        else:
            # If '0.-.-.-' is not first, remove all elements after the first occurrence
            ec_list = ec_list[:zero_index]
            
    # Convert the EC list to a set for efficient lookups and deletions
    ec_set = set(ec_list)
    
    # Sort EC numbers by effective depth (deepest first)
    sorted_ec_list = sorted(ec_set, key=lambda x: len([p for p in x.split('.') if p != '-']), reverse=True)
    
    # Track EC numbers to remove (parent nodes)
    to_remove = set()
    
    # Process each EC number starting from the deepest effective layer
    for ec in sorted_ec_list:
        parts = ec.split('.')
        for i in range(len([p for p in parts if p != '-']) - 1, 0, -1):
            # construct the parent node
            parent = '.'.join(parts[:i] + ['-'] * (4 - i))
            if parent in ec_set:
                to_remove.add(parent)  # Mark parent for removal
    # Remove all marked parent nodes from the original set
    filtered_ec_set = ec_set - to_remove
    # Return the filtered list in the original order
    filtered_ec_list = [ec for ec in ec_list if ec in filtered_ec_set]

    return filtered_ec_list

# def write_max_sep_choices(df, csv_name, first_grad=True):#use_max_grad=False,
#     out_file = open(csv_name + '_maxsep.csv', 'a', newline='')
#     csvwriter = csv.writer(out_file, delimiter=',')
#     for col in df.columns:
#         ec = []
#         smallest_10_dist_df = df[col].nsmallest(10)
#         dist_lst = list(smallest_10_dist_df)
#         max_sep_i = maximum_separation(dist_lst, first_grad)
#         candidate_ecs = smallest_10_dist_df.index[:max_sep_i+1].tolist()
#         filtered_ecs = filter_ec_numbers(candidate_ecs)
#         ec_row = [col]
#         for ec in filtered_ecs:
#             dist = smallest_10_dist_df[ec]
#             dist_str = "{:.4f}".format(dist)
            
#             ec_row.append('EC:' + str(ec) + '/' + dist_str)
#         csvwriter.writerow(ec_row)
#     return

def write_max_sep_choices(df, first_grad=True):
    """返回包含预测EC和原始距离数据的元组"""
    result_ec = []
    all_top_candidates = {}
    
    for col in df.columns:
        # 获取前10最近邻数据
        smallest_10 = df[col].nsmallest(10)
        all_top_candidates[col] = smallest_10.copy()
        
        # 核心算法保持不变
        distances = smallest_10.tolist()
        max_sep_i = maximum_separation(distances, first_grad)
        candidate_ecs = smallest_10.index[:max_sep_i+1].tolist()
        filtered_ecs = filter_ec_numbers(candidate_ecs)
        
        result_ec.append(';'.join(filtered_ecs))
    
    return result_ec, all_top_candidates

def save_prediction_results(ids, result_ec, pred_path, top10_dist, batch_top_nodes=None):
    """
    通用预测结果保存函数
    :param ids: 蛋白质ID列表
    :param result_ec: 预测EC号列表
    :param pred_path: 预测文件路径
    :param top10_dist: EC号距离字典
    :param batch_top_nodes: 关键残基列表（可选）
    """
    # 创建最终结果列表
    final_results = []
    
    # 根据是否有关键残基数据选择遍历方式
    iter_data = zip(ids, result_ec, batch_top_nodes) if batch_top_nodes else zip(ids, result_ec)
    
    for item in iter_data:
        name = item[0]
        ec = item[1]
        top_nodes = item[2] if batch_top_nodes else None
        
        # 处理EC距离信息
        ec_list = ec.split(';')
        ec_dist_list = []
        has_non_zero_ec = any(e != '0.-.-.-' for e in ec_list)
        
        for e in ec_list:
            dist = top10_dist[name][e]
            ec_dist_list.append(f"{e}/{dist:.4f}")
        
        ec_dist_str = ';'.join(ec_dist_list)
        
        # 处理关键残基数据
        if batch_top_nodes:
            # 如果所有EC号均为0则清空关键残基
            top_nodes = [] if not has_non_zero_ec else top_nodes
            top_nodes_str = ', '.join(map(str, top_nodes)) if top_nodes else ''
        
        # 组装结果元组
        if batch_top_nodes:
            final_results.append((name, ec_dist_str, top_nodes_str))
        else:
            final_results.append((name, ec_dist_str))

    file_exists = os.path.isfile(pred_path)
    
    with open(pred_path, 'a', newline='') as predfile:
        writer = csv.writer(predfile, delimiter='\t')
        
        # 动态生成表头
        headers = ['protein_name', 'pred_label']
        if batch_top_nodes:
            headers.append('key_residues')
        
        # 写入表头（仅当文件不存在时）
        if not file_exists or os.path.getsize(pred_path) == 0:
            writer.writerow(headers)
        
        # 写入数据行
        for row in final_results:
            writer.writerow(row)

def get_pred_labels(out_filename, pred_type="_pred"):
    file_name = out_filename+pred_type
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter='\t')
    next(csvreader, None)
    pred_label = []
    for row in csvreader:
        preds_ec_lst = []
        multi_preds_with_dist = row[1]
        preds_with_dist = multi_preds_with_dist.split(';')
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from 3.5.2.6/10.8359
            ec_i = pred_ec_dist.split("/")[0]
            preds_ec_lst.append(ec_i)
        pred_label.append(preds_ec_lst)
    return pred_label


def get_pred_probs(out_filename, pred_type="_pred"):
    file_name = out_filename+pred_type
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter='\t')
    next(csvreader, None)
    pred_probs = []
    for row in csvreader:
        preds_ec_lst = []
        multi_preds_with_dist = row[1]
        preds_with_dist = multi_preds_with_dist.split(';')
        probs = torch.zeros(len(preds_with_dist))
        count = 0
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = float(pred_ec_dist.split("/")[1])
            probs[count] = ec_i
            #preds_ec_lst.append(probs)
            count += 1
        # sigmoid of the negative distances 
        probs = (1 - torch.exp(-1/probs)) / (1 + torch.exp(-1/probs))
        probs = probs/torch.sum(probs)
        pred_probs.append(probs)
    return pred_probs


def get_true_labels(file_name, pred_type="_true"):
    out_filename = file_name+pred_type
    all_labels = set()
    true_label = []
    with open(out_filename+ '.csv', mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        next(csv_reader, None)  
        for row in csv_reader:
            protein_name = row[0].strip()  
            true_labels_str = row[1].strip()  
            true_labels = true_labels_str.split(';')  
            all_labels.update(true_labels)  
            true_label.append(true_labels)  
    
    return true_label, all_labels    

def get_ec_pos_dict(mlb, true_label, pred_label):
    ec_list = []
    pos_list = []
    for i in range(len(true_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([true_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([true_label[i]]))[1])
    for i in range(len(pred_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([pred_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([pred_label[i]]))[1])
    label_pos_dict = {}
    for i in range(len(ec_list)):
        ec, pos = ec_list[i], pos_list[i]
        label_pos_dict[ec] = pos
        
    return label_pos_dict

def get_eval_metrics(pred_label, pred_probs, true_label, all_label):
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    n_test = len(pred_label)
    pred_m = np.zeros((n_test, len(mlb.classes_)))
    true_m = np.zeros((n_test, len(mlb.classes_)))
    # for including probability
    pred_m_auc = np.zeros((n_test, len(mlb.classes_)))
    label_pos_dict = get_ec_pos_dict(mlb, true_label, pred_label)
    for i in range(n_test):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
         # fill in probabilities for prediction
        labels, probs = pred_label[i], pred_probs[i]
        for label, prob in zip(labels, probs):
            if label in all_label:
                pos = label_pos_dict[label]
                pred_m_auc[i, pos] = prob
    prec = precision_score(true_m, pred_m, average='weighted', zero_division=0)
    rec = recall_score(true_m, pred_m, average='weighted')
    f1 = f1_score(true_m, pred_m, average='weighted')
    roc = roc_auc_score(true_m, pred_m_auc, average='weighted')
    acc = accuracy_score(true_m, pred_m)
    return prec, rec, f1, roc, acc

def dist_map_helper(keys1, lookup1, keys2, lookup2):
    dist = {}
    for i, key1 in tqdm(enumerate(keys1)):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist

def ec_similarity(pred_ec, true_ec):
    """
    Calculate the hierarchical similarity between two EC labels.
    Supports wildcard handling and dynamic level processing.
    
    Args:
        pred_ec (str): Predicted EC label (e.g., "1.2.3.4").
        true_ec (str): True EC label (e.g., "1.2.3.4").
    
    Returns:
        float: Similarity score between 0 and 1.
    """
    pred_parts = pred_ec.split('.')
    true_parts = true_ec.split('.')
    
    max_level = max(len(pred_parts), len(true_parts))

    
    match_level = 0
    for i in range(max_level):
        # Get the comparison values for the current level (treat out-of-range as wildcard)
        t = true_parts[i] if i < len(true_parts) else '-'
        p = pred_parts[i] if i < len(pred_parts) else '-'
        
        if t == '-':
            # True label allows any value at this level
            match_level += 1
        elif p == t:
            # Strict match continues
            match_level += 1
        else:
            # Mismatch and true label is not a wildcard, terminate
            break
    
    return match_level / max_level

def hierarchical_metrics(true_labels, pred_labels):
    """
    Calculate weighted hierarchical metrics based on label frequencies.
    
    Args:
        true_labels (list[list[str]]): List of true EC labels for each sample.
        pred_labels (list[list[str]]): List of predicted EC labels for each sample.
    
    Returns:
        dict: Dictionary containing weighted precision, recall, and F1 score.
    """
    # 统计真实标签和预测标签的出现次数
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    
    # Track counts of true and predicted labels
    label_metrics = defaultdict(lambda: {'tp_recall': 0.0,'tp_precision':0.0, 'true_total': 0, 'pred_total': 0})
    
    # Track metrics for each label
    for sample_true, sample_pred in zip(true_labels, pred_labels):
        # Update true label counts
        for ec in sample_true:
            true_counts[ec] += 1
            label_metrics[ec]['true_total'] += 1
        
        # Update predicted label counts
        for ec in sample_pred:
            pred_counts[ec] += 1
            label_metrics[ec]['pred_total'] += 1
        
        # Calculate recall-oriented matches
        for true_ec in sample_true:
            max_sim = max([ec_similarity(p, true_ec) for p in sample_pred], default=0)
            label_metrics[true_ec]['tp_recall'] += max_sim
        
        # Calculate precision-oriented matches
        for pred_ec in sample_pred:
            max_sim = max([ec_similarity(pred_ec, t) for t in sample_true], default=0)
            label_metrics[pred_ec]['tp_precision'] += max_sim
    
    # Calculate overall metrics
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    total_true = sum(true_counts.values())
    total_pred = sum(pred_counts.values())
    
    #  Combine all unique labels from true and predicted
    all_labels = set(list(true_counts.keys()) + list(pred_counts.keys()))
    
    for label in all_labels:
        metrics = label_metrics[label]
        weight = true_counts[label] / total_true if total_true > 0 else 0
        # Calculate recall
        recall = metrics['tp_recall'] / metrics['true_total'] if metrics['true_total'] > 0 else 0
        recall_sum += recall * weight
        
        #  Calculate precision
        precision = metrics['tp_precision'] / metrics['pred_total'] if metrics['pred_total'] > 0 else 0
        precision_sum += precision * weight
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0    
        f1_sum += f1 * weight
        
    # Calculate F1 score
    weighted_precision = precision_sum
    weighted_recall = recall_sum
    weighted_f1 = f1_sum
    
    return {
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1': weighted_f1,
    }

class DynamicHierarchyLoss(nn.Module):

    """
    Dynamic Hierarchy Loss: A custom loss function designed for hierarchical classification tasks.
    This loss function combines joint embedding loss and hierarchy-based triplet loss.
    """
    
    def __init__(self):
        """
        Initialize the DynamicHierarchyLoss module.
        """
        super().__init__()
    
    def compute_joint_loss(self, y_pred, true_labels, y_true_embeddings):
        """
        Compute the joint embedding loss between predicted embeddings and true embeddings.
        
        Args:
            y_pred (Tensor): Predicted embeddings of shape [B, D].
            true_labels (list[str]): List of true EC labels in the format "EC1;EC2;...".
            y_true_embeddings (list[list[Tensor]]): List of true EC embeddings for each sample.
            
        Returns:
            Tensor: Scalar joint loss value.
        """
        device = y_pred.device
        
        # Compute the maximum number of ECs across all samples
        max_ec = max(len(ec.strip().split(';')) for ec in true_labels)
        
        # Dynamically pad embeddings and generate masks
        padded_embeds = []
        ec_masks = []
        for embeds in y_true_embeddings:
            pad_size = max_ec - len(embeds)
            
            # Pad embeddings
            padded = torch.cat([
                torch.stack(embeds, dim=0), 
                torch.zeros(pad_size, embeds[0].shape[0], device=device)
            ], dim=0)
            padded_embeds.append(padded)
            
            # Generate boolean masks
            mask = torch.cat([
                torch.ones(len(embeds), dtype=torch.bool, device=device),
                torch.zeros(pad_size, dtype=torch.bool, device=device)
            ], dim=0)
            ec_masks.append(mask)
        
        # Convert to tensors
        padded_embeds = torch.stack(padded_embeds)  # [B, max_ec, D]
        ec_masks = torch.stack(ec_masks)            # [B, max_ec]
        
        #  Compute distances in a vectorized manner
        y_pred_exp = y_pred.unsqueeze(1)            # [B, 1, D]
        Dp = torch.norm(y_pred_exp - padded_embeds, p=2, dim=-1)  # [B, max_ec]
        Dp_squared = Dp ** 2
        
        # Mask out invalid positions
        valid_loss = (Dp_squared * ec_masks.float()).sum(dim=1)  # [B]
        num_valid_ec = ec_masks.sum(dim=1).clamp(min=1e-6)       # Prevent division by zero
        loss_per_sample = valid_loss / num_valid_ec              # Average per sample
        
        return loss_per_sample.mean()  # Average over the batch   
    
    def forward(self, model_output,true_labels, y_true_embeddings):  #,labels_loss
        """
        Compute the total loss combining joint embedding loss and hierarchy-based triplet loss.
        
        Args:
            model_output (dict): Dictionary containing model outputs.
                - 'pred': Predicted embeddings of shape [B, D].
                - 'hard_negatives': Dictionary containing hard negative samples.
                    - 'Dn': Distances to hard negative samples.
                    - 'margins': Margin values for triplet loss.
                    - 'mask': Mask indicating valid hard negative samples.
                    - 'negs_exist': Boolean tensor indicating samples with hard negatives.
            true_labels (list[str]): List of true EC labels in the format "EC1;EC2;...".
            y_true_embeddings List[Union[Tensor, List[Tensor]]] (N_true_samples, emb_dim): List of true EC embeddings for each sample.
            
        Returns:
            Tensor: Total loss value.
        """
        total_loss = 0.0
        joint_loss = 0.0
        hierarchy_loss = 0.0
        y_pred = model_output['pred']
        hard_negatives = model_output['hard_negatives']
        # Compute joint embedding loss
        joint_loss = self.compute_joint_loss(y_pred, true_labels, y_true_embeddings)
        # Extract hard negative components
        Dn = hard_negatives['Dn']
        margins = hard_negatives['margins']
        mask = hard_negatives['mask']
        negs_exist = hard_negatives['negs_exist']  # [B]

        # Only process samples with hard negatives
        if negs_exist.any():
            # Extract valid samples
            y_pred_valid = y_pred[negs_exist]
            Dn_valid = Dn[negs_exist]
            margins_valid = margins[negs_exist]
            mask_valid = mask[negs_exist]
            # Compute maximum number of positive embeddings
            max_positives = max(len(pos_list) for pos_list in y_true_embeddings)
            padded_embs = []
            for emb in y_true_embeddings:
                pad_size = max_positives - len(emb)
                padded_embs.append(F.pad(torch.stack(emb, dim=0) , (0,0,0,pad_size), value=0))
            padded_y_true_embeddings = torch.stack(padded_embs)  # [B, max_pos, D] 
                       
            y_true_valid = padded_y_true_embeddings[negs_exist]

            # Compute positive distances
            Dp = torch.norm(y_pred_valid.unsqueeze(1) - y_true_valid, dim=-1)
            Dp_exp = Dp.unsqueeze(-1).expand_as(Dn_valid)

            # Compute loss matrix
            loss_matrix = torch.clamp_min(Dp_exp - Dn_valid + margins_valid, 0)
            masked_loss = (loss_matrix * mask_valid.float()).sum()
            valid_counts = mask_valid.sum() + 1e-8#.any(dim=-1)
            hierarchy_loss = masked_loss / valid_counts
        else:
            hierarchy_loss = torch.tensor(0.0)
        # Combine losses    
        total_loss = joint_loss + hierarchy_loss
        return total_loss

if __name__ == '__main__':
    Seed_everything(seed=42)
    parser = argparse.ArgumentParser(description='Train your own ACCESS model.')
    
    parser.add_argument("--root_dir", type=str, default="./final_code/gvp_protein_embedding/processed", 
                        help="Root directory containing processed data")
    parser.add_argument("--resultFolder", type=str, default="./final_code/result",
                        help="Output directory for models and results.")
    
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of samples per batch")
    
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loading workers")    

    args = parser.parse_args()

    os.makedirs(os.path.join(args.resultFolder, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.resultFolder, "results"), exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("")
    log_file_path  = os.path.join(args.resultFolder, 'results', 'logfile.log')
    handler = logging.FileHandler(log_file_path)
    handler.setFormatter(logging.Formatter('%(message)s', ""))
    logger.addHandler(handler)
    torch.set_num_threads(1)
    # # ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Define the path to the CSV file containing data lengths
    csv_file_path = args.root_dir + '/data_length.csv'
    train = ACCSEE_DataSet(args.root_dir,csv_file_path,type='train')
    valid = ACCSEE_DataSet(args.root_dir,csv_file_path,type='valid')
    test = ACCSEE_DataSet(args.root_dir,csv_file_path,type='test')
    logging.info(f"Train size: {len(train)}, Valid size: {len(valid)}, Test size: {len(test)}")
    # print(train[0])
    num_workers = args.num_workers #Set the number of workers for data loading
    #Create samplers for each dataset to ensure sampling within files
    train_sampler = FileWiseSubsetRandomSampler(train, replacement=False)
    valid_sampler = FileWiseSubsetRandomSampler(valid, replacement=False)
    test_sampler = FileWiseSubsetRandomSampler(test, replacement=False)
    # Create data loaders for each dataset
    train_loader = DataLoader(train, batch_size=args.batch_size, follow_batch=['x'], sampler=train_sampler, pin_memory=False, num_workers=num_workers) 
    valid_loader = DataLoader(valid, batch_size=args.batch_size, follow_batch=['x'], sampler=valid_sampler, pin_memory=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x'], sampler=test_sampler, pin_memory=False, num_workers=num_workers)
    # Set hyperparameters
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.001    
    dropout = 0.3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ec_file_path = "./PDB_file/benchmark_train_labels_all_0001.csv"
    model = get_model(device = device,ec_file=ec_file_path,dropout=dropout)
    
    #Calculate the number of model parameters
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"The number of model parameters.: {num_params}")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
    emb_dict = model.node_dict
    criterion = DynamicHierarchyLoss()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=0.0001)

    epoch_not_improving = 0    
    
    train_epochs_loss = [] 

    valid_epochs_loss = []  
    train_epochs_f1 = []
    valid_epochs_f1 = []
    train_epochs_auc = []  
    valid_epochs_auc = []  
    training_time = []

    # Initialize variables for the best model
    best_f1 = 0.0
    best_auc = 0.0
    epoch_not_improving = 0
    best_model = None
    # Check for existing checkpoint to continue training
    checkpoint_path = os.path.join(args.resultFolder, 'models', 'checkpoint.pt')
    if os.path.exists(checkpoint_path):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'),weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] +1 
        best_auc = checkpoint['best_auc']
        best_f1 = checkpoint['best_f1']
        epoch_not_improving = checkpoint['epoch_not_improving']     
        train_epochs_loss = checkpoint['train_epochs_loss']
        valid_epochs_loss = checkpoint['valid_epochs_loss']
        valid_epochs_f1 = checkpoint['valid_epochs_f1']
        valid_epochs_auc = checkpoint['valid_epochs_auc']
        training_time = checkpoint['total_training_time']
        logging.info(f"Continue training from epoch {start_epoch + 1}, best_auc: {best_auc:.3f}, best_f1: {best_f1:.3f}")
    else:
        start_epoch = 0
    # Training loop        
    for epoch in range(start_epoch, 20):
        epoch_start_time = time.time()
        model.to(device)
        model.train()
        train_epoch_loss = []
        all_train_preds = []
        all_train_labels = []
        # Training batch loop
        data_it = tqdm(train_loader)
        for data in data_it:
            data = data.to(device)
            optimizer.zero_grad()
            model_output= model(data)
            y_true_embedding = model.get_embeddings(data.y,device=device)
            loss= criterion(model_output,data.y, y_true_embedding)  
            loss.backward()
            optimizer.step()
            scheduler.step()
       
            train_epoch_loss.append(loss.item())
            
        # Log training loss
        train_epochs_loss.append(np.average(train_epoch_loss))
        logging.info(f"Epoch {epoch + 1}, Loss: {train_epochs_loss[-1]:.4f}")
        logging.info(f"Epoch {epoch+1} training took {time.time() - epoch_start_time:.2f} seconds")
        
        # Validation step                
        out_filename = os.path.join(args.resultFolder, 'results', 'valid')
        model.eval()
        val_epoch_loss = []  
        all_preds = []
        all_labels = []
        # Validation batch loop
        with torch.no_grad():
            for data in tqdm(valid_loader,dynamic_ncols=True):
                data = data.to(device)
                model_output = model(data)
                y_true_embedding = model.get_embeddings(data.y,device=device)
                loss = criterion(model_output,data.y, y_true_embedding)  
                val_epoch_loss.append(loss.item())
                # Additional validation logic (e.g., saving predictions and labels)
                ids = data['protein_name']
                eval_similarity = dist_map_helper(ids, model_output['pred'], list(emb_dict.keys()), model.ec_embeddings)
                result_ec, top10_dist = write_max_sep_choices(pd.DataFrame.from_dict(eval_similarity))
                save_prediction_results(ids,result_ec,out_filename + '_pred.csv',top10_dist) 
                
                # Save true labels to CSV
                true_labels_df = pd.DataFrame({
                    'protein_name': ids,
                    'true_label': data.y
                })
                os.makedirs(os.path.dirname(out_filename), exist_ok=True)
                true_labels_df.to_csv(out_filename + '_true.csv', mode='a', index=False, sep='\t',header=not os.path.exists(out_filename + '_true.csv'))                                

        #Calculate validation metrics
        pred_label = get_pred_labels(out_filename, pred_type='_pred')
        pred_probs = get_pred_probs(out_filename, pred_type='_pred')
        true_label, all_label = get_true_labels(out_filename, pred_type='_true')
        prec, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        metrics = hierarchical_metrics(true_label, pred_label)
        training_time.append(time.time() - epoch_start_time)
        valid_epochs_auc.append(roc)
        valid_epochs_f1.append(f1)
        
        #Log validation metrics
        logging.info("############ Protein EC Labels Prediction Results ############")
        logging.info('-' * 62)
        logging.info(f"Epoch {epoch+1} training and validation took {training_time[-1]:.2f} seconds")
        logging.info(f'total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'Precision: {prec:.3} | Recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} ')    
        logging.info(f"Hierarchical_Precision: {metrics['precision']:.4f}")
        logging.info(f"Hierarchical_Recall: {metrics['recall']:.4f}")
        logging.info(f"Hierarchical_F1: {metrics['f1']:.4f}")    
        logging.info('-' * 62)
        
        # Clean up intermediate files
        os.remove(out_filename + '_true.csv')  
        os.remove(out_filename + '_pred.csv')       
                           
        valid_epochs_loss.append(np.average(val_epoch_loss))
        
        # Update best model if current model is better
        if valid_epochs_f1[-1] > best_f1:
            best_auc = max(valid_epochs_auc[-1], best_auc)
            best_f1 = max(valid_epochs_f1[-1], best_f1) 
            best_model = model.cpu().state_dict()
            epoch_not_improving = 0
            torch.save({
                'model_state_dict': best_model,
                'best_auc': best_auc,
                'best_f1': best_f1
            }, os.path.join(args.resultFolder, 'models', 'best_model.pt'))
            logging.info(f"New best model saved at Epoch {epoch+1}, Best F1: {best_f1:.3f}, Best AUC: {best_auc:.3f}")
        else:
            epoch_not_improving += 1  
        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch +1,
            'best_auc': best_auc,
            'best_f1': best_f1,
            'epoch_not_improving': epoch_not_improving,
            'train_epochs_loss': train_epochs_loss,
            'valid_epochs_loss': valid_epochs_loss,
            'valid_epochs_f1': valid_epochs_f1,
            'valid_epochs_auc': valid_epochs_auc,
            'total_training_time': training_time
        }, os.path.join(args.resultFolder, 'models', 'checkpoint.pt'))

        # Early stopping
        if epoch_not_improving > 50:
            logging.info("Early stopping")
            break
        
        # Log epoch summary
        logging.info(f"Epoch {epoch+1}, Valid Loss: {valid_epochs_loss[-1]:.4f}, Valid F1: {valid_epochs_f1[-1]:.3f}, Valid AUC: {valid_epochs_auc[-1]:.3f}")
        logging.info(f"Epoch {epoch+1} training took {training_time[-1]:.2f} seconds")
        # Clear cache
        torch.cuda.empty_cache()
    # Save training and validation results
    results_df = pd.DataFrame({
        'Train Loss': train_epochs_loss,
        'Valid Loss': valid_epochs_loss,
        'Valid F1': valid_epochs_f1, 
        'Valid AUC': valid_epochs_auc,
        'Epoch Time': training_time
    })
    results_path = os.path.join(args.resultFolder, 'results', 'training_validation_results.csv')
    results_df.to_csv(results_path, index=False)
    logging.info(f"Training and validation results saved to {results_path},Total Training Time: {np.sum(training_time) / 60:.2f} minutes")
    
    # Testing step
    best_model_path = os.path.join(args.resultFolder, 'models', 'best_model.pt')
    if os.path.exists(best_model_path):
        best_model_checkpoint = torch.load(best_model_path,weights_only=False)
        model.load_state_dict(best_model_checkpoint['model_state_dict'],strict=False)
        logging.info(f"Best model loaded!")
    else:
        logging.error("Best model not found.")
        exit()  
        
    #  Set device for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() 
    
    # Test the model
    test_filename = os.path.join(args.resultFolder, 'results', 'test')
    test_loss = []  
    all_preds = []
    all_labels = []
    all_protein_names = []
    with torch.no_grad():
        for data in tqdm(test_loader,dynamic_ncols=True):
            data = data.to(device)
            model_output = model(data)
            y_true_embedding = model.get_embeddings(data.y,device=device)
            loss = criterion(model_output,data.y, y_true_embedding) 
            test_loss.append(loss.item())  
            # Additional testing logic (e.g., saving predictions and labels)
            ids = data['protein_name']
            eval_similarity = dist_map_helper(ids, model_output['pred'], list(emb_dict.keys()), model.ec_embeddings)
            
            result_ec, top10_dist = write_max_sep_choices(pd.DataFrame.from_dict(eval_similarity))
            save_prediction_results(ids,result_ec,test_filename + '_pred.csv',top10_dist) 
            
            # Save true labels to CSV
            true_labels_df = pd.DataFrame({
                'protein_name': ids,
                'true_label': data.y
            })
            os.makedirs(os.path.dirname(test_filename), exist_ok=True)
            true_labels_df.to_csv(test_filename + '_true.csv', mode='a', index=False, sep='\t',header=not os.path.exists(test_filename + '_true.csv'))              
                                       
    # Calculate test metrics
    pred_label = get_pred_labels(test_filename, pred_type='_pred')
    pred_probs = get_pred_probs(test_filename, pred_type='_pred')
    true_label, all_label = get_true_labels(test_filename, pred_type='_true')
    metrics = hierarchical_metrics(true_label, pred_label)    
    prec, rec, f1, roc, acc = get_eval_metrics(
        pred_label, pred_probs, true_label, all_label)
    # Log test results
    logging.info("############ Protein EC Labels Prediction Results ############")
    logging.info('-' * 62)
    logging.info(f'total samples: {len(true_label)} | total ec: {len(all_label)} \n'
        f'Precision: {prec:.3} | Recall: {rec:.3}'
        f'| F1: {f1:.3} | AUC: {roc:.3} ')  
    logging.info(f"Hierarchical_Precision: {metrics['precision']:.4f}")
    logging.info(f"Hierarchical_Recall: {metrics['recall']:.4f}")
    logging.info(f"Hierarchical_F1: {metrics['f1']:.4f}")          
    logging.info('-' * 62)
