import os
import sys
import tqdm
import pdb

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F

class DensityNetwork(nn.Module):
    def __init__(self, encoder, bound=0.2, num_layers=8, hidden_dim=256, skips=[4], out_dim=1, last_activation="sigmoid"):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.encoder = encoder
        self.in_dim = encoder.output_dim
        self.bound = bound
        
        # Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i not in skips 
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in range(1, num_layers-1, 1)])
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # Activations
        self.activations = nn.ModuleList([nn.LeakyReLU() for i in range(0, num_layers-1, 1)])
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, x):
        
        x = self.encoder(x, self.bound)
        
        input_pts = x[..., :self.in_dim]

        for i in range(len(self.layers)):

            linear = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = linear(x)
            x = activation(x)
        
        return x
    

class ComplexGaborLayer(nn.Module):
    '''
        Implicit representation with complex Gabor nonlinearity
        
        Inputs;
            in_features: Input features
            out_features; Output features
            bias: if True, enable bias for the linear operation
            is_first: Legacy SIREN parameter
            omega_0: Legacy SIREN parameter
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=40.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features,
                                out_features,
                                bias=bias,
                                dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        
        return torch.exp(1j*omega - scale.abs().square())
    
class WireNetwork(nn.Module):
    def __init__(self, encoder, bound=0.2, num_layers=8, hidden_dim=256, skips=[4], out_dim=1, last_activation='sigmoid'):
        super().__init__()
        
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer
        
        # Variables related to encoder
        self.encoder = encoder
        self.in_dim = encoder.output_dim
        self.bound = bound
        
        # Variables for Wavelet Function
        first_omega_0=30
        hidden_omega_0=30.
        scale=10.0
        
        
        # Since complex numbers are two real numbers, reduce the number of 
        # hidden parameters by 2
        hidden_dim = int(hidden_dim/np.sqrt(2))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = 'gabor'    
            
        self.net = []
        self.net.append(self.nonlin(self.in_dim,
                                    hidden_dim, 
                                    omega0=first_omega_0,
                                    sigma0=scale,
                                    is_first=True,
                                    trainable=False))

        for i in range(num_layers - 2):
            self.net.append(self.nonlin(hidden_dim,
                                        hidden_dim, 
                                        omega0=hidden_omega_0,
                                        sigma0=scale))

        final_linear = nn.Linear(hidden_dim,
                                 out_dim,
                                 dtype=dtype)            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        
        x = self.encoder(x, self.bound)
        
        input_pts = x[..., :self.in_dim]
        
        output = self.net(coords)
        
        if self.wavelet == 'gabor':
            return output.real
         
        return output