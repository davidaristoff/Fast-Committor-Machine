import torch
import torch.nn as nn

class CommittorNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_hidden_layers, output_dim = 1, activations = 'Tanh'):
        super(CommittorNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        if activations.lower() == 'relu':
            self.layer_activation = nn.ReLU()
        else:
            self.layer_activation = nn.Tanh()
        self.last_layer_activation = nn.Sigmoid()

        # FCNN
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_size))
        for _ in range(self.num_hidden_layers - 1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layers.append(nn.Linear(self.hidden_size, self.output_dim))

    def initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.layer_activation(layer(x))
        x = self.last_layer_activation(self.layers[-1](x))

        return x
    
    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
        return output.numpy()

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
