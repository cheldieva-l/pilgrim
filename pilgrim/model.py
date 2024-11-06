import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock_old(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.1, activation_function="mish", use_batch_norm=True):#relu
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None
        self.activation = self._get_activation_function(activation_function)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        out += residual
        out = self.activation(out)
        return out

    @staticmethod
    def _get_activation_function(name):  # +
        if name == "relu":
            return nn.ReLU()
        elif name == "mish":
            return nn.Mish()
        else:
            raise ValueError(f"Unknown activation function: {name}")


%%time
import torch.optim as optim
import torch.nn.functional as F
# from torcheval.metrics import R2Score, MeanSquaredError

input_layer_size_for_one_hot = state_size * n_unique_symbols_in_states

# layer_1_size =1165
layer_1_size =  CFG['list_layers_sizes'][0]

 
class Pilgrim(nn.Module):
    def __init__(self, state_size, hd1=5000, hd2=1000, nrd=2, output_dim=1, dropout_rate=0.1, activation_function="mish", use_batch_norm=True): #relu
          super(Pilgrim, self).__init__() 
        
        self.n_unique_symbols_in_states = n_unique_symbols_in_states # More gradual compression 
        
        self.layers = nn.Sequential( 
            nn.Linear(state_size * 6, 2048), 
            nn.BatchNorm1d(2048), 
            nn.Mish(), 
            
            nn.Linear(2048, 1024), 
            nn.BatchNorm1d(1024), 
            nn.Mish(), 
            
            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512), 
            nn.Mish(), 
            
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256), 
            nn.Mish(), 
            
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128), 
            nn.Mish(), 
            
            nn.Linear(128, 1) 
        ) 
        
        # Skip connections 
        self.skip1 = nn.Linear(2048, 1024) 
        self.skip2 = nn.Linear(1024, 512) 
        self.skip3 = nn.Linear(512, 256) 
        
    def forward(self, x): 
        x = torch.nn.functional.one_hot(x.long(), num_classes=self.n_unique_symbols_in_states).float() 
        x = x.view(x.size(0), -1) # Add skip connections 
        x1 = self.layers[0:3](x) # 2048 
        x2 = self.layers[3:6](x1) + self.skip1(x1) # 1024 
        x3 = self.layers[6:9](x2) + self.skip2(x2) # 512 
        x4 = self.layers[9:12](x3) + self.skip3(x3) # 256 
        out = self.layers[12:](x4) # Final layers return out
        return out


    @staticmethod
    def _get_activation_function(name):
        if name == "relu":
            return nn.ReLU()
        elif name == "mish":
            return nn.Mish()
        else:
            raise ValueError(f"Unknown activation function: {name}")

def count_parameters(model):
    """Count the trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def batch_process(model, data, device, batch_size):
    """
    Process data through a model in batches.

    :param data: Tensor of input data
    :param model: A PyTorch model with a forward method that accepts data
    :param device: Device to perform computations (e.g., 'cuda', 'cpu')
    :param batch_size: Number of samples per batch
    :return: Concatenated tensor of model outputs
    """
    model.eval()
    model.to(device)

    outputs = torch.empty(data.size(0), dtype=torch.float32, device=device)

    # Process each batch
    for i in range(0, data.size(0), batch_size):
        batch = data[i:i+batch_size].to(device)
        with torch.no_grad():
            batch_output = model(batch).flatten()
        outputs[i:i+batch_size] = batch_output

    return outputs
