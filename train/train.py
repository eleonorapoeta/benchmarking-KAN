import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import time


def train(epochs, model, device, train_loader, experiment=None):

    optimizer= AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = ExponentialLR(optimizer, gamma=0.8)
    loss = nn.CrossEntropyLoss()

    # Times of training epochs
    times = []

    for epoch in tqdm(range(epochs), desc = 'tqdm() Progress Bar'):
        start_time = time.time()
        model.train()
        closs = 0
        pred = []
        gt = []
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)            
            loss_value = loss(output, target.long())
            loss_value.backward()
            optimizer.step()
            closs += loss_value.item()
            pred.append(output.argmax(1))
            gt.append(target)
    
        pred = torch.cat(pred).float()
        gt = torch.cat(gt).float()
        accuracy_train = (pred == gt).float().mean().item()
        # Calculate f1 score
        f1 = f1_score(gt.cpu(), pred.cpu(), average='macro')

        scheduler.step()
        end_time = time.time()

        times.append(end_time-start_time)

    # Return the average times of training epochs
    t = sum(times)/len(times)
    return t
