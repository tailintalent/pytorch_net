
# coding: utf-8

# In[1]:


from __future__ import print_function
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from pytorch_net.util import get_activation, init_weight, init_bias, init_module_weights, init_module_bias
AVAILABLE_REG = ["L1", "L2", "param"]
Default_Activation = "linear"


# ## Register all layer types:

# In[2]:


def get_Layer(layer_type, input_size, output_size, W_init = None, b_init = None, settings = {}, is_cuda = False):
    if layer_type == "Simple_Layer":
        layer = Simple_Layer(input_size = input_size,
                             output_size = output_size,
                             W_init = W_init,
                             b_init = b_init,
                             settings = settings,
                             is_cuda = is_cuda,
                            )
    else:
        raise Exception("layer_type '{0}' not recognized!".format(layer_type))
    return layer


def load_layer_dict(layer_dict, layer_type, is_cuda = False):
    new_layer = get_Layer(layer_type = "Symbolic_Layer",
                          input_size = layer_dict["input_size"],
                          output_size = layer_dict["output_size"],
                          W_init = layer_dict["weights"],
                          b_init = layer_dict["bias"],
                          settings = layer_dict["settings"],
                          is_cuda = is_cuda,
                         )
    return new_layer


# ## Simple Layer:

# In[3]:


class Simple_Layer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        W_init = None,     # initialization for weights
        b_init = None,     # initialization for bias
        settings = {},     # Other settings that are relevant to this specific layer
        is_cuda = False,
        ):
        # Firstly, must perform this step:
        super(Simple_Layer, self).__init__()
        # Saving the attribuites:
        if isinstance(input_size, tuple):
            self.input_size = reduce(lambda x, y: x * y, input_size)
            self.input_size_original = input_size
        else:
            self.input_size = input_size
        if isinstance(output_size, tuple):
            self.output_size = reduce(lambda x, y: x * y, output_size)
            self.output_size_original = output_size
        else:
            self.output_size = output_size
        # self.W_init, self.b_init can be a numpy array, or a string like "glorot-normal":
        self.W_init = W_init
        self.b_init = b_init
        self.is_cuda = is_cuda
        self.settings = settings
        
        # Other attributes that are specific to this layer:
        self.activation = settings["activation"] if "activation" in settings else Default_Activation
        
        # Define the learnable parameters in the module (use any name you like). 
        # use nn.Parameter() so that the parameters is registered in the module and can be gradient-updated:
        self.W_core = nn.Parameter(torch.randn(self.input_size, self.output_size))
        self.b_core = nn.Parameter(torch.zeros(self.output_size))
        # Use the given W_init (numpy array or string) to initialize the weights:
        init_weight(self.W_core, init = self.W_init)  
        init_bias(self.b_core, init = self.b_init)
        if is_cuda:
            self.cuda()


    @property
    def struct_param(self):
        output_size = self.output_size_original if hasattr(self, "output_size_original") else self.output_size
        return [output_size, "Simple_Layer", self.settings]


    @property
    def layer_dict(self):
        input_size = self.input_size_original if hasattr(self, "input_size_original") else self.input_size
        output_size = self.output_size_original if hasattr(self, "output_size_original") else self.output_size
        Layer_dict =  {
            "input_size": input_size,
            "output_size": output_size,
            "settings": self.settings,
        }
        Layer_dict["weights"], Layer_dict["bias"] = self.get_weights_bias()
        return Layer_dict
    
    
    def load_layer_dict(self, layer_dict):
        new_layer = load_layer_dict(layer_dict, "Simple_Layer", self.is_cuda)
        self.__dict__.update(new_layer.__dict__)


    def forward(self, input, p_dict = None):
        output = input
        if hasattr(self, "input_size_original"):
            output = output.view(-1, self.input_size)
        # Perform dot(X, W) + b:
        output = torch.matmul(output, self.W_core) + self.b_core
        
        # If p_dict is not None, update the first neuron's activation according to p_dict:
        if p_dict is not None:
            p_dict = p_dict.view(-1)
            if len(p_dict) == 2:
                output_0 = output[:,:1] * p_dict[1] + p_dict[0]
            elif len(p_dict) == 1:
                output_0 = output[:,:1] + p_dict[0]
            else:
                raise
            if output.size(1) > 1:
                output = torch.cat([output_0, output[:,1:]], 1)
            else:
                output = output_0

        # Perform activation function:
        output = get_activation(self.activation)(output)
        if hasattr(self, "output_size_original"):
            output = output.view(*((-1,) + self.output_size_original))
        return output


    def prune_output_neurons(self, neuron_ids):
        if not isinstance(neuron_ids, list):
            neuron_ids = [neuron_ids]
        preserved_ids = torch.LongTensor(np.array(list(set(range(self.output_size)) - set(neuron_ids))))
        if self.is_cuda:
            preserved_ids = preserved_ids.cuda()
        self.W_core = nn.Parameter(self.W_core.data[:, preserved_ids])
        self.b_core = nn.Parameter(self.b_core.data[preserved_ids])
        self.output_size = self.W_core.size(1)
    
    
    def prune_input_neurons(self, neuron_ids):
        if not isinstance(neuron_ids, list):
            neuron_ids = [neuron_ids]
        preserved_ids = torch.LongTensor(np.array(list(set(range(self.input_size)) - set(neuron_ids))))
        self.W_core = nn.Parameter(self.W_core.data[preserved_ids, :])
        self.input_size = self.W_core.size(0)
    
    
    def add_output_neurons(self, num_neurons, mode = "imitation"):
        if mode == "imitation":
            W_core_mean = self.W_core.mean().data[0]
            W_core_std = self.W_core.std().data[0]
            b_core_mean = self.b_core.mean().data[0]
            b_core_std = self.b_core.std().data[0]
            new_W_core = torch.randn(self.input_size, num_neurons) * W_core_std + W_core_mean
            new_b_core = torch.randn(num_neurons) * b_core_std + b_core_mean
        elif mode == "zeros":
            new_W_core = torch.zeros(self.input_size, num_neurons)
            new_b_core = torch.zeros(num_neurons)
        else:
            raise Exception("mode {0} not recognized!".format(mode))
        self.W_core = nn.Parameter(torch.cat([self.W_core.data, new_W_core], 1))
        self.b_core = nn.Parameter(torch.cat([self.b_core.data, new_b_core], 0))
        self.output_size += num_neurons
        
    
    def add_input_neurons(self, num_neurons, mode = "imitation"):
        if mode == "imitation":
            W_core_mean = self.W_core.mean().data[0]
            W_core_std = self.W_core.std().data[0]
            b_core_mean = self.b_core.mean().data[0]
            b_core_std = self.b_core.std().data[0]
            new_W_core = torch.randn(num_neurons, self.output_size) * W_core_std + W_core_mean
        elif mode == "zeros":
            new_W_core = torch.zeros(num_neurons, self.output_size)
        else:
            raise Exception("mode {0} not recognized!".format(mode))
        self.W_core = nn.Parameter(torch.cat([self.W_core.data, new_W_core], 0))
        self.input_size += num_neurons


    
    def get_weights_bias(self):
        W_core, b_core = self.W_core, self.b_core
        if self.is_cuda:
            W_core = W_core.cpu()
            b_core = b_core.cpu()
        return deepcopy(W_core.data.numpy()), deepcopy(b_core.data.numpy())

    
    def get_regularization(self, mode, source = ["weight", "bias"]):
        reg = Variable(torch.FloatTensor(np.array([0])), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        for source_ele in source:
            if source_ele == "weight":
                if mode == "L1":
                    reg = reg + self.W_core.abs().sum()
                elif mode == "L2":
                    reg = reg + (self.W_core ** 2).sum()
                elif mode in AVAILABLE_REG:
                    pass
                else:
                    raise Exception("mode '{0}' not recognized!".format(mode))
            elif source_ele == "bias":
                if mode == "L1":
                    reg = reg + self.b_core.abs().sum()
                elif mode == "L2":
                    reg = reg + (self.b_core ** 2).sum()
                elif mode in AVAILABLE_REG:
                    pass
                else:
                    raise Exception("mode '{0}' not recognized!".format(mode))
        return reg


    def set_cuda(self, is_cuda):
        if is_cuda:
            self.cuda()
        else:
            self.cpu()
        self.is_cuda = is_cuda


    def set_trainable(self, is_trainable):
        if is_trainable:
            self.W_core.requires_grad = True
            self.b_core.requires_grad = True
        else:
            self.W_core.requires_grad = False
            self.b_core.requires_grad = False

