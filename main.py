# Description: This file contains the main code to run the experiments
from utils.run import run_experiments 
import argparse

# Arguments 
# dataset_name: the name of the dataset to run the experiments
# model_name: the name of the model to run the experiments
# num_epochs: the number of epochs to train the model

parser = argparse.ArgumentParser(description='Run experiments')
parser.add_argument('--dataset_name', type=str, default='poker', help='The name of the dataset to run the experiments')
parser.add_argument('--model_name', type=str, default='mlp', help='The name of the model to run the experiments')
parser.add_argument('--num_epochs', type=int, default=10, help='The number of epochs to train the model')


if __name__ == '__main__':
    args = parser.parse_args()
    run_experiments(dataset_name=args.dataset_name, model_name=args.model_name, num_epochs=args.num_epochs)








