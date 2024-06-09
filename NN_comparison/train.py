import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def run_optimization(net, num_epochs, batch_size, X, Y, data_dim, optimizer, verbose=True, timing=False):
"""
Inputs:  net - neural net, 
         num_epochs - number epochs to loop over, 
         batch_size - how often to update gradients,
         X,Y,data_dim - Input/output pairs and data dimension, 
         optimizer - AdamW, SDG,
         verbose - progress printouts,
         timing - record training time
Outputs: losses - average loss per epoch, 
         net - updated neural network, 
         epoch_times, avg_batch_times - timing for network training
Desc: Runs the optimzation neural networks
"""
    # Set up data
    full_X_dat, full_XinI, full_XinB = (torch.tensor(X[:,:data_dim,], dtype=torch.float32),
                                        torch.tensor(X[:,data_dim], dtype=torch.float32).unsqueeze(1), 
                                        torch.tensor(X[:,data_dim+1], dtype=torch.float32).unsqueeze(1))
    full_Y_dat, full_YinI, full_YinB = (torch.tensor(Y[:,:data_dim,], dtype=torch.float32),
                                        torch.tensor(Y[:,data_dim], dtype=torch.float32).unsqueeze(1), 
                                        torch.tensor(Y[:,data_dim+1], dtype=torch.float32).unsqueeze(1))
    full_X_weights = torch.tensor(X[:,data_dim+2], dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(full_X_dat, full_XinI, full_XinB, full_X_weights, full_Y_dat, full_YinI, full_YinB)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)   
    num_batches = len(loader)

    criterion = nn.MSELoss()
    losses = []

    for epoch in range(num_epochs):
        batch_loss = 0

        for batch in loader:
            X_dat, XinI, XinB, X_w, Y_dat, YinI, YinB = batch
            optimizer.zero_grad() 

            # Forward pass
            q_X = net(X_dat)
            q_Y = net(Y_dat)

            # Compute the loss
            inner_sum = XinI * q_X + XinB - YinI * q_Y - YinB
            loss = criterion(X_w * inner_sum, torch.zeros_like(XinI))
            batch_loss += loss.item()
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Update loss and monitor training progress
        losses.append(batch_loss/num_batches)
        if verbose == True:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {batch_loss/num_batches}')

    # Return
    if not timing: epoch_times, avg_batch_times = None, None
    return losses, net, epoch_times, avg_batch_times


def validate(q, X, Y, data_dim, batch_size):
"""
Inputs:   q - neural network
          X,Y,data_dim - validation data and dimension
          batch_size - can default to size of validation dataset
Outputs:  validation_loss/num_batches - return average validation error
Desc: 
"""
    ### Set up data
    full_X_dat, full_XinI, full_XinB = (torch.tensor(X[:,:data_dim,], dtype=torch.float32),
                                        torch.tensor(X[:,data_dim], dtype=torch.float32).unsqueeze(1), 
                                        torch.tensor(X[:,data_dim+1], dtype=torch.float32).unsqueeze(1))
    full_Y_dat, full_YinI, full_YinB = (torch.tensor(Y[:,:data_dim,], dtype=torch.float32),
                                        torch.tensor(Y[:,data_dim], dtype=torch.float32).unsqueeze(1), 
                                        torch.tensor(Y[:,data_dim+1], dtype=torch.float32).unsqueeze(1))
    full_X_weights = torch.tensor(X[:,data_dim+2], dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(full_X_dat, full_XinI, full_XinB, full_X_weights, full_Y_dat, full_YinI, full_YinB)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)   
    num_batches = len(loader)
    criterion = nn.MSELoss()

    validation_loss = 0.0

    for batch in loader:
        X_dat, XinI, XinB, X_w, Y_dat, YinI, YinB = batch

        # Forward pass
        with torch.no_grad():
            q_X = q(X_dat)
            q_Y = q(Y_dat)

        # Compute the loss
        inner_sum = XinI * q_X + XinB - YinI * q_Y - YinB
        loss = criterion(X_w * inner_sum, torch.zeros_like(XinI))
        validation_loss += loss.item()

    return validation_loss/num_batches
