from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Set the random seeds for reproducibility pytorch
def seed_everything(seed=0):
    import random
    random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(target, output, average='macro'):
    
    
    # Calculate f1 score
    f1 = f1_score(target.cpu().numpy(), output.argmax(1).cpu().numpy(), average=average)
    # Calculate precision score
    precision = precision_score(target.cpu().numpy(), output.argmax(1).cpu().numpy(), average=average)
    # Calculate recall score
    recall = recall_score(target.cpu().numpy(), output.argmax(1).cpu().numpy(), average=average)

    if average == 'macro':
        # Obtain the confusion matrix
        cm = confusion_matrix(target, output.argmax(1))
        TN, FP, FN, TP = cm.ravel()

        # Calculate FPR and FNR
        FPR = FP / (FP + TN)
        FNR = FN / (FN + TP)

        return f1, precision, recall, FPR, FNR
    else:
        return f1, precision, recall, 0, 0

def write_results(f, nodes, parameters, papi_results, test_accuracies, test_f1, test_precision_scores, test_recall_scores, test_fpr, test_fnr, times):
    
    # Write all on the result file
    f.write(f"Nodes:{nodes}\n")
    f.write(f"Parameters:{parameters}\n")
    
    f.write(f"PAPI Results:{papi_results}\n")
   
    f.write(f"Test Accuracies\n")
    for el in test_accuracies:
      
        f.write(f"{el}\n")
    f.write(f"Test F1\n")
    for i in test_f1:    
        f.write(f"{i}\n")
    f.write(f"Test Precision\n")
    for i in test_precision_scores:
        f.write(f"{i}\n")
    f.write(f"Test Recall\n")
    for i in test_recall_scores:
        f.write(f"{i}\n")
    f.write(f"Test FPR\n")
    for i in test_fpr:
        f.write(f"{i}\n")
    f.write(f"Test FNR\n")
    for i in test_fnr:
        f.write(f"{i}\n")
    f.write(f"Avg training time per epoch:\n")
    for i in times:
        f.write(f"{i}\n")
    f.write("_____________________\n")