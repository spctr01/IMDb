#training and evaluation functions

import torch 
import torch.nn as nn

def train(data_loader, model, optimizer, device):
     
    model.train()
    for data in data_loader:
        reviews = data['review']
        target = data['target']

        reviews = reviews.(to(device, dtype= torch.long)
        target = target.to(device, dtype= torch.float)

        #clear gradient 
        optimizer.zero_grad()

        predictions = model(reviews)

        #claculate loss

        loss = nn.BCEWithLogitsLoss()(predictions, target.view(-1,1))
        loss.backward()
        optimizer.step()

        
def evaluate(data_loader, model, device):

    final_predictions = []
    final_target= []

    model.eval()
    #disable gradient calculation
    with torch.no_grad():

        reviews = data['review']
        target = data['target']

        reviews = reviews.(to(device, dtype= torch.long)
        target = target.to(device, dtype= torch.float)

        predictions = model(reviews)

        predictions = prediction.cpu().numpy().tolist()
        target = data['target'].cpu().numpy().tolist()

        final_predictions.extend(predictions)
        final_target.extend(target)
