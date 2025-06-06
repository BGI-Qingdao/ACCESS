import sys
import os
from torch.nn.utils.rnn import pad_sequence
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from training.train import *

def write_max_sep_choices(df, first_grad=True):
    """Return a tuple containing predicted ECs and raw distance data"""
    result_ec = []
    all_top_candidates = {}
    
    for col in df.columns:
        # Get the top 10 nearest neighbor data
        smallest_10 = df[col].nsmallest(10)
        all_top_candidates[col] = smallest_10.copy()
        
        # Core algorithm
        distances = smallest_10.tolist()
        max_sep_i = maximum_separation(distances, first_grad)
        candidate_ecs = smallest_10.index[:max_sep_i+1].tolist()
        filtered_ecs = filter_ec_numbers(candidate_ecs)
        
        result_ec.append(';'.join(filtered_ecs))
    
    return result_ec, all_top_candidates

def get_batch_indices(batch_tensor):
    """
    Get the start and end indices of each protein in the concatenated sequence
    :param batch_tensor: [total_nodes] Indicator for which protein each residue belongs to
    :return: List of tuples indicating start and end indices
    """
    unique, counts = torch.unique(batch_tensor, return_counts=True)
    split_indices = []
    current = 0
    for count in counts:
        split_indices.append( (current, current + count.item()) )
        current += count.item()
    return split_indices

class AttentionGenerator:
    """External module for generating attention weights based on gradients"""
    def __init__(self, model, feat_dim):
        self.model = model
        self.feat_dim = feat_dim
        
        # Register gradient hook
        self.gradients = None
    
    def compute_attention(self, data):
        """Feature dimension: [N, d]"""

        # Gradient direction entropy calculation
        self.gradients = torch.norm(self.model.conv_protein.last_h_V_grad, dim=1)
        # Edge connections and weights
        edge_index = data[("protein", "p2p", "protein")].edge_index
        rbf_feats = data[("protein", "p2p", "protein")]["edge_s"][:, :16]
        
        # Decay weights       
        x = torch.linspace(0, 2, 16)  
        rbf_weights = torch.exp(-((x ** 2) * torch.where(x < 1, 0.5, 5.0))) 
        
        edge_weights = torch.softmax(
            (rbf_feats * rbf_weights.to(rbf_feats.device)).sum(dim=1), dim=0
        )
        # Symmetrize spatial matrix to reflect bidirectional interactions
        spatial_corr = torch.sparse_coo_tensor(
            edge_index, edge_weights, 
            (data.num_nodes, data.num_nodes)
        ).to_dense()
        spatial_corr = (spatial_corr + spatial_corr.T) / 2  # Force symmetry
        
        # Diffusion normalization (preserves topological information)
        degree_matrix = torch.diag(spatial_corr.sum(1))
        spatial_corr = torch.mm(torch.inverse(degree_matrix), spatial_corr)
        # Multi-order attention propagation
        first_order = torch.matmul(spatial_corr, self.gradients)
        second_order = torch.matmul(spatial_corr, first_order)
        weighted_corr = first_order + 0.3 *second_order  # Decay coefficient control  second_order
        return weighted_corr


if __name__ == '__main__':
    Seed_everything(seed=42)
    parser = argparse.ArgumentParser(description='Train your own ACCESS model.')

    parser.add_argument("--model", type=str, default="./access/saved_models/best_model.pt",
                        help="Path to the model.")
    parser.add_argument("--root_dir", type=str, default="./access/gvp_protein_embedding/processed",
                        help="Root directory containing processed data")
    parser.add_argument("--resultFolder", type=str, default="./access/inference_result",
                        help="Output directory for models and results.")
    
    #If the parameter is not added, the default is False.
    parser.add_argument("--print_true_label", action='store_true',
                        help="Whether to print true labels in output")  
    parser.add_argument("--print_embedding", action='store_true',
                        help="Whether to print embeddings in output")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of samples per batch")
    
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of data loading workers")    

    args = parser.parse_args()

    # args.print_true_label = True
    # args.print_embedding = True
    
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
    test = ACCSEE_DataSet(args.root_dir,csv_file_path,type='test')
    logging.info(f"Test size: {len(test)}")
    # print(train[0])
    num_workers = args.num_workers #Set the number of workers for data loading
    test_sampler = FileWiseSubsetRandomSampler(test, replacement=False)
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

    # Testing step
    if os.path.exists(args.model):
        best_model_checkpoint = torch.load(args.model,weights_only=False)
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
    # with torch.no_grad():
    for data in tqdm(test_loader,dynamic_ncols=True):
        data = data.to(device)
        attn_gen = AttentionGenerator(model, feat_dim=64)
        with torch.enable_grad():  
            model_output = model(data)
        protein_embeddings = model_output['pred']
        y_true_embedding = model.get_embeddings(data.y,device=device)
        loss = criterion(model_output,data.y, y_true_embedding) 
        test_loss.append(loss.item())  
        ids = data['protein_name']
        eval_similarity = dist_map_helper(ids, model_output['pred'], list(emb_dict.keys()), model.ec_embeddings)
        eval_df = pd.DataFrame.from_dict(eval_similarity)
        result_ec, top10_dist = write_max_sep_choices(pd.DataFrame.from_dict(eval_similarity))
        ec_embeddings_list = model.get_embeddings(result_ec,device=device)
        # Embedding alignment processing
        ec_valid_mask = pad_sequence(
            [torch.ones(len(ecs)) for ecs in ec_embeddings_list],
            batch_first=True,
            padding_value=0
        ).bool().to(device)
        y_pre_embeddings = pad_sequence(
            [torch.stack(ecs) for ecs in ec_embeddings_list], 
            batch_first=True, 
            padding_value=0
        )
        # Calculation process
        batch_size, max_ec, dim = y_pre_embeddings.shape
        protein_expanded = protein_embeddings.unsqueeze(1).expand(-1, max_ec, -1)
        similarity = protein_expanded * y_pre_embeddings   #8,4,64
        
        ec_scores = similarity.sum(-1) * ec_valid_mask
        scores = ec_scores.sum(1, keepdim=True) / (ec_valid_mask.sum(1, keepdim=True).float() + 1e-8)
        
        # Gradient backpropagation
        model.zero_grad()
        scores.sum().backward()
        # Calculate attention importance
        attn_imp = attn_gen.compute_attention(data)  # [N]

        # Node importance analysis
        protein_ranges = get_batch_indices(data['protein'].batch)
        batch_top_nodes = []
        for start, end in protein_ranges:
            
            local_grad_importance = attn_gen.gradients[start:end]
            local_att_importance = attn_imp[start:end]
            def compute_alpha(grad_probs, attn_probs):
                # Calculate information entropy
                grad_entropy = -torch.sum(grad_probs * torch.log(grad_probs + 1e-9))
                attn_entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9))
                
                # The larger the entropy, the higher the uncertainty, reduce its weight
                total_entropy = grad_entropy + attn_entropy
                alpha = attn_entropy / total_entropy  
                
                return alpha.clamp(min=0.2, max=0.8)  # Set boundaries     
            grad_probs = torch.softmax(local_grad_importance, dim=0)
            attn_probs = torch.softmax(local_att_importance, dim=0)
            alpha = compute_alpha(grad_probs, attn_probs)
            abs_local_importance = alpha * grad_probs + (1 - alpha) * attn_probs
                         
            threshold = torch.quantile(abs_local_importance, q=0.9)
            valid_indices = torch.nonzero(abs_local_importance > threshold).squeeze(1)
            
            if valid_indices.numel() == 0:
                # Fallback selection logic
                top_values, top_indices = torch.topk(abs_local_importance, k=1)
                sorted_indices_1_based = (top_indices + 1).tolist()
            else:
                # Sort valid values
                valid_values = abs_local_importance[valid_indices]
                sorted_values, sort_order = torch.sort(valid_values, descending=True)
                sorted_indices = valid_indices[sort_order]
                sorted_indices_1_based = (sorted_indices + 1).cpu().tolist()
            
            batch_top_nodes.append(sorted_indices_1_based)             

        save_prediction_results(ids,result_ec,test_filename + '_inference_pred.csv',top10_dist,batch_top_nodes = batch_top_nodes)
                
        result_dict = {}
        if args.print_true_label or args.print_embedding:
            result_dict['protein_name'] = ids
        if args.print_true_label:
            result_dict['true_label'] = data.y
        if args.print_embedding:
            # Process embeddings
            y_pred_embedding_list = [str(emb.tolist()) for emb in model_output['pred'].detach().cpu().numpy()]
            result_dict['embedding'] = y_pred_embedding_list
            
        # Write to CSV
        if result_dict:
            true_path = test_filename + '_inference_true.csv'
            write_header = not os.path.exists(true_path) or os.path.getsize(true_path) == 0
            pd.DataFrame(result_dict).to_csv(
                true_path,
                mode='a',
                header=write_header,
                index=False,
                sep='\t'
            )

    # Calculate test metrics
    if args.print_true_label:
        pred_label = get_pred_labels(test_filename, pred_type='_inference_pred')
        pred_probs = get_pred_probs(test_filename, pred_type='_inference_pred')
        true_label, all_label = get_true_labels(test_filename, pred_type='_inference_true')#_inference
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
    