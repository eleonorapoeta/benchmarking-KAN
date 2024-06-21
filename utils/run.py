from data.loading_data import prepare_dataset
from utils.utils import seed_everything, count_parameters, compute_metrics, write_results
import torch
import time
from train.train import train
from pypapi import papi_low as papi
from pypapi import events
from models.basic_mlp_net import BasicNet
from kan.efficient_kan import KAN
import numpy as np



def run_single_model(f, dataset_name, model_name, shape_dataset, train_loader, test_loader,  num_epochs, num_classes, average='macro'):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nodes = []
    training_time_per_epoch = []
    parameters = []
    papi_results = []
    test_accuracies = []
    test_f1 = []
    test_precision_scores = []
    test_recall_scores = []
    test_fpr = []
    test_fnr = []

  
    # Set the number of intermediate nodes
    if model_name == 'kan':
        num_intermediate_nodes = np.arange(1, 11, 1)
    else:
        num_intermediate_nodes = np.arange(10, 110, 10)

    if dataset_name == 'poker':
        num_intermediate_nodes*=10

    for n in num_intermediate_nodes:

        if model_name == 'kan':
            model = KAN([shape_dataset, n, num_classes])
        else:
            model = BasicNet(shape_dataset, num_classes, n)
    
       
        model.to(device)

        training_time_per_epoch.append(train(num_epochs, model, device, train_loader))
        nodes.append(n)
        parameters.append(count_parameters(model))


        # Test the model
        # We need to change the device to cpu to test the model and collect information about FLOPS and instructions
        device = 'cpu'
        model.to(device)
        model.eval()
        correct = 0
        total = 0

        papi.library_init()
        evs = papi.create_eventset()
        papi.add_event(evs, events.PAPI_SP_OPS)
        papi.add_event(evs, events.PAPI_TOT_INS)

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                papi.start(evs)
                output = model(data)
                result = papi.stop(evs)
                total += target.size(0)
                correct += (output.argmax(1) == target).sum().item()

                t_f1, test_precision, test_recall, FPR, FNR = compute_metrics(target, output, average=average)
                test_f1.append(t_f1*100)
                papi_results.append(result)
                test_precision_scores.append(test_precision*100)
                test_recall_scores.append(test_recall*100)
                test_fpr.append(FPR*100)
                test_fnr.append(FNR*100)

        papi.cleanup_eventset(evs)
        papi.destroy_eventset(evs)
        test_acc = correct / total * 100
        test_accuracies.append(test_acc)

    # Write all on the file
    write_results(f, nodes, parameters, papi_results, test_accuracies, test_f1, test_precision_scores, test_recall_scores, test_fpr, test_fnr, training_time_per_epoch)



def run_experiments(dataset_name, model_name, num_epochs):
    classes = {'poker': 10, 'adult': 2, 'diabetes': 2, 'shuttle': 7, 'spam':2,  'musk':2, 'dry_bean': 7, 'gamma':2, 'breast_cancer':2}
    multivariate_datasets = ['poker', 'shuttle', 'dry_bean']
    average = ['weighted' if dataset_name in multivariate_datasets else 'macro']
    train_loader, test_loader, shape_dataset = prepare_dataset(dataset_name)
    
    seeds = [0, 1, 2, 3, 4]

    # Open results file
    f = open(f"results_{model_name}_{dataset_name}.txt", "w")
    # Write name of dataset
    f.write(f"Dataset: {dataset_name}\n")
    f.write("_____________________\n")

    c = classes[dataset_name]

    for s in seeds:
        seed_everything(s)
        f.write(f"Seed: {s}\n")
        if model_name=='all':  
            run_single_model(f=f, dataset_name=dataset_name, shape_dataset=shape_dataset, train_loader=train_loader, test_loader=test_loader,  
                             num_epochs=num_epochs, num_classes=c, average=average[0], model_name='kan')
            run_single_model(f=f, dataset_name=dataset_name, shape_dataset=shape_dataset, train_loader=train_loader, test_loader=test_loader,  
                             num_epochs=num_epochs, num_classes=c, average=average[0], model_name='mlp')
        else:
            run_single_model(f=f, dataset_name=dataset_name, model_name=model_name, shape_dataset=shape_dataset, train_loader=train_loader, test_loader=test_loader,  
                             num_epochs=num_epochs, num_classes=c, average=average[0])
            

