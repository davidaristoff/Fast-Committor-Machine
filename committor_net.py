import torch
import models
import train
import scipy.io
import numpy as np
import random as rd
import logging
import os 
import pandas as pd
from sklearn.model_selection import train_test_split

# Save network feedback
save_data = True
printing = False
timing = True

if save_data:
    output_folder = 'DEFAULT'
    os.makedirs(output_folder, exist_ok=True)

# Load triple well .mat file
mat_data = scipy.io.loadmat('DATA.mat')
ref_data = scipy.io.loadmat('REF_DATA.mat')

# Assign as input/output
ws = mat_data['w']
X_data = np.hstack((mat_data['X'], mat_data['XinI'], mat_data['XinB'], ws/max(ws)))
Y_data = np.hstack((mat_data['Y'], mat_data['YinI'], mat_data['YinB']))
split_size = .2

# NN parameters
data_dim = mat_data['X'].shape[1]
num_samples = mat_data['X'].shape[0]
hidden_size = 20 
num_hidden_layers = 3 
lr =  1e-4      # learning rate

# Training parameters
max_epochs = 500                        # Max number of epochs
num_epochs = 1                          # Number of epochs/loops before checking stopping criterion...
max_loops = max_epochs // num_epochs    # ... higher is faster, data saving assumes num_epochs = 1
batch_size = 500                        # batch size before updating weights
patience = 20                           # Early stopping criterions
min_delta = 0

# Sweep parameters
num_repeats = 10
num_steps = 10
start, stop = 10**3, 10**6
data_sizes = np.linspace(np.log(start), np.log(stop), num_steps,)
data_sizes = np.ceil(np.exp(data_sizes)).astype(int)
data_sizes = [x - 1 if x % 10 != 0 else x for x in data_sizes]    # Make python and MATLAB agree

for data_size in data_sizes:
    # Pre-allocate arrays with np.nan for easier filtering later
    bulk_losses = np.full((max_epochs, num_repeats), -1, dtype=float)
    bulk_validation_scores = np.full((max_epochs, num_repeats), -1, dtype=float)
    bulk_reference_scores = np.full((max_epochs, num_repeats), -1, dtype=float)
    bulk_epoch_times = np.full((max_epochs, num_repeats), -1, dtype=float)

    for repeat in range(num_repeats):
        # Sample uniformly for reduced dataset
        reduced_samples = data_size
        validation_samples = np.floor(.1*reduced_samples).astype(int)
        indices = rd.sample(range(num_samples), reduced_samples)
        X, Y = X_data[indices], Y_data[indices]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
        
        # Get reference data
        X_ref = ref_data['Xref']
        q_ref = ref_data['qref']

        # Declare network, optimizer, and early stopper
        q = models.CommittorNet(input_dim=data_dim, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
        q.initialize_weights()
        optimizer = torch.optim.AdamW(q.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
        early_stopper = models.EarlyStopper(patience=patience, min_delta=min_delta)

        epoch_number = 0
        best_mse = np.inf
        losses = []
        validation_scores = []
        test_errors = []
        epoch_times = []

        # while epoch_number < max_epochs:
        for loop_num in range(max_loops):
            # Train model
            loss, q, epoch_time, batch_times = train.run_optimization(net=q, num_epochs=num_epochs, batch_size=batch_size, X=X_train, Y=Y_train, data_dim=data_dim, optimizer=optimizer, verbose=printing, timing=timing)

            # Get validation error
            validation_error = train.validate(q, X_test, Y_test, data_dim, batch_size)
            validation_scores.extend([validation_error] * num_epochs)

            # Get the ref error
            # Get NN predictions
            q_test = q.predict(torch.tensor(X_ref, dtype=torch.float32))

            # Get L2 Error
            test_error = np.mean((q_test - q_ref)**2)
            test_errors.extend([test_error] * num_epochs)

            losses.extend(loss)
            epoch_times.extend(epoch_time)

            # Check if this is the best net
            if test_error < best_mse:
                best_mse = test_error
                best_model_params = q.state_dict() 

            # Increment passes
            epoch_number += num_epochs

            # Add results to preallocated arrays -- assumes num_epochs=1
            bulk_losses[loop_num][repeat] = loss[0]
            bulk_validation_scores[loop_num][repeat] = validation_error
            bulk_reference_scores[loop_num][repeat] = test_error
            bulk_epoch_times[loop_num][repeat] = epoch_time[0]

            # check for early stopping
            if early_stopper.early_stop(validation_error):
                break

    # Save the training results
    if save_data:
        # Save timings and loss data
        timings_file = os.path.join(output_folder, f"{data_size}_results.csv")
        timings_df = pd.DataFrame({
            "Loss": losses,
            "ValidationError": validation_scores,
            "TestError": test_errors,
            "EpochTime": epoch_times
        })
        timings_df.to_csv(timings_file, index=False)


