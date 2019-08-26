
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
    elif layer_type == "SuperNet_Layer":
        layer = SuperNet_Layer(input_size = input_size,
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
        assert output.size(0) == input.size(0), "output_size {0} must have same length as input_size {1}. Check shape!".format(output.size(0), input.size(0))
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


# In[ ]:


class SuperNet_Layer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        W_init = None,     # initialization for weights
        b_init = None,     # initialization for bias
        settings = {},
        is_cuda = False,
        ):
        super(SuperNet_Layer, self).__init__()
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
        self.W_init = W_init
        self.b_init = b_init
        self.is_cuda = is_cuda
        
        # Obtain additional initialization settings if provided:
        self.W_available = settings["W_available"] if "W_available" in settings else ["dense", "Toeplitz"]
        self.b_available = settings["b_available"] if "b_available" in settings else ["dense", "None"]
        self.A_available = settings["A_available"] if "A_available" in settings else ["linear", "relu"]
        self.W_sig_init  = settings["W_sig_init"] if "W_sig_init" in settings else None # initialization for the significance for the weights
        self.b_sig_init  = settings["b_sig_init"] if "b_sig_init" in settings else None # initialization for the significance for the bias
        self.A_sig_init  = settings["A_sig_init"] if "A_sig_init" in settings else None # initialization for the significance for the activations
        for W_candidate in self.W_available:
            if "2D-in" in W_candidate:
                self.input_size_2D = settings["input_size_2D"]
            if "2D-out" in W_candidate:
                self.output_size_2D = settings["output_size_2D"]
        for b_candidate in self.b_available:
            if "2D" in b_candidate:
                self.output_size_2D = settings["output_size_2D"]
        
        # Initialize layer:
        self.init_layer()
        if is_cuda:
            self.cuda()
    
    
    @property
    def struct_param(self):
        return [self.output_size, "SuperNet_Layer", self.settings]

        
    def init_layer(self):
        self.W_layer_seed = nn.Parameter(torch.FloatTensor(np.random.randn(self.input_size, self.output_size)))
        self.b_layer_seed = nn.Parameter(torch.zeros(self.output_size))
        init_weight(self.W_layer_seed, init = self.W_init)
        init_bias(self.b_layer_seed, init = self.b_init)
        if "arithmetic-series-in" in self.W_available:
            self.W_interval_j = nn.Parameter(torch.randn(self.output_size) / np.sqrt(self.input_size + self.output_size))
        if "arithmetic-series-out" in self.W_available:
            self.W_interval_i = nn.Parameter(torch.randn(self.input_size) / np.sqrt(self.input_size + self.output_size))
        if "arithmetic-series-2D-in" in self.W_available:
            self.W_mean_2D_in = nn.Parameter(torch.randn(self.output_size) / np.sqrt(self.input_size_2D[0] + self.input_size_2D[1] + self.output_size))
            self.W_interval_2D_in = nn.Parameter(torch.randn(2, self.output_size) / np.sqrt(self.input_size_2D[0] + self.input_size_2D[1] + self.output_size))
        if "arithmetic-series-2D-out" in self.W_available:
            self.W_mean_2D_out = nn.Parameter(torch.randn(self.input_size) / np.sqrt(self.input_size + self.output_size_2D[0] + self.output_size_2D[1]))
            self.W_interval_2D_out = nn.Parameter(torch.randn(2, self.input_size) / np.sqrt(self.input_size + self.output_size_2D[0] + self.output_size_2D[1]))
        if "arithmetic-series" in self.b_available:
            self.b_interval = nn.Parameter(torch.randn(1) / np.sqrt(self.output_size))
        if "arithmetic-series-2D" in self.b_available:
            self.b_mean_2D = nn.Parameter(torch.randn(1) / np.sqrt(self.output_size))
            self.b_interval_2D = nn.Parameter(torch.randn(2) / np.sqrt(self.output_size_2D[0] + self.output_size_2D[1]))
        
        if self.W_sig_init is None:
            self.W_sig = nn.Parameter(torch.zeros(len(self.W_available)))
        else:
            self.W_sig = nn.Parameter(torch.FloatTensor(self.W_sig_init))
        if self.b_sig_init is None:
            self.b_sig = nn.Parameter(torch.zeros(len(self.b_available)))
        else:
            self.b_sig = nn.Parameter(torch.FloatTensor(self.b_sig_init))
        if self.A_sig_init is None:
            self.A_sig = nn.Parameter(torch.zeros(len(self.A_available)))
        else:
            self.A_sig = nn.Parameter(torch.FloatTensor(self.A_sig_init))


    def get_layers(self, source = ["weight", "bias"]):    
        # Superimpose different weights:
        if "weight" in source:
            self.W_list = []
            for weight_type in self.W_available:
                if weight_type == "dense":
                    W_layer = self.W_layer_seed
                elif weight_type == "Toeplitz":
                    W_layer_stacked = []
                    if self.output_size > 1:
                        if self.is_cuda:
                            inv_idx = torch.arange(self.output_size - 1, 0, -1).long().cuda()
                        else:
                            inv_idx = torch.arange(self.output_size - 1, 0, -1).long()
                        W_seed = torch.cat([self.W_layer_seed[0][inv_idx], self.W_layer_seed[:,0]])
                    else:
                        W_seed = self.W_layer_seed[:,0]
                    for j in range(self.output_size):
                        W_layer_stacked.append(W_seed[self.output_size - j - 1: self.output_size - j - 1 + self.input_size])
                    W_layer = torch.stack(W_layer_stacked, 1)
                elif weight_type == "arithmetic-series-in":
                    mean_j = self.W_layer_seed.mean(0)
                    if self.is_cuda:
                        idx_i = torch.FloatTensor(np.repeat(np.arange(self.input_size), self.output_size)).cuda()
                        idx_j = torch.LongTensor(range(self.output_size) * self.input_size).cuda()
                    else:
                        idx_i = torch.FloatTensor(np.repeat(np.arange(self.input_size), self.output_size))
                        idx_j = torch.LongTensor(range(self.output_size) * self.input_size)
                    offset = self.input_size / float(2) - 0.5
                    W_layer = (mean_j[idx_j] + self.W_interval_j[idx_j] * Variable(idx_i - offset, requires_grad = False)).view(self.input_size, self.output_size)
                elif weight_type == "arithmetic-series-out":
                    mean_i = self.W_layer_seed.mean(1)
                    if self.is_cuda:
                        idx_i = torch.LongTensor(np.repeat(np.arange(self.input_size), self.output_size)).cuda()
                        idx_j = torch.FloatTensor(range(self.output_size) * self.input_size).cuda()
                    else:
                        idx_i = torch.LongTensor(np.repeat(np.arange(self.input_size), self.output_size))
                        idx_j = torch.FloatTensor(range(self.output_size) * self.input_size)
                    offset = self.output_size / float(2) - 0.5
                    W_layer = (mean_i[idx_i] + self.W_interval_i[idx_i] * Variable(idx_j - offset, requires_grad = False)).view(self.input_size, self.output_size)
                elif weight_type == "arithmetic-series-2D-in":
                    idx_i, idx_j, idx_k = np.meshgrid(range(self.input_size_2D[0]), range(self.input_size_2D[1]), range(self.output_size), indexing = "ij")
                    idx_i = torch.from_numpy(idx_i).float().view(-1)
                    idx_j = torch.from_numpy(idx_j).float().view(-1)
                    idx_k = torch.from_numpy(idx_k).long().view(-1)
                    if self.is_cuda:
                        idx_i, idx_j, idx_k = idx_i.cuda(), idx_j.cuda(), idx_k.cuda()
                    offset_i = self.input_size_2D[0] / float(2) - 0.5
                    offset_j = self.input_size_2D[1] / float(2) - 0.5
                    W_layer = (self.W_mean_2D_in[idx_k] +                                self.W_interval_2D_in[:, idx_k][0] * Variable(idx_i - offset_i, requires_grad = False) +                                self.W_interval_2D_in[:, idx_k][1] * Variable(idx_j - offset_j, requires_grad = False)).view(self.input_size, self.output_size)
                elif weight_type == "arithmetic-series-2D-out":
                    idx_k, idx_i, idx_j = np.meshgrid(range(self.input_size), range(self.output_size_2D[0]), range(self.output_size_2D[1]), indexing = "ij")
                    idx_k = torch.from_numpy(idx_k).long().view(-1)
                    idx_i = torch.from_numpy(idx_i).float().view(-1)
                    idx_j = torch.from_numpy(idx_j).float().view(-1)
                    if self.is_cuda:
                        idx_i, idx_j, idx_k = idx_i.cuda(), idx_j.cuda(), idx_k.cuda()
                    offset_i = self.output_size_2D[0] / float(2) - 0.5
                    offset_j = self.output_size_2D[1] / float(2) - 0.5
                    W_layer = (self.W_mean_2D_out[idx_k] +                                self.W_interval_2D_out[:, idx_k][0] * Variable(idx_i - offset_i, requires_grad = False) +                                self.W_interval_2D_out[:, idx_k][1] * Variable(idx_j - offset_j, requires_grad = False)).view(self.input_size, self.output_size)
                else:
                    raise Exception("weight_type '{0}' not recognized!".format(weight_type))
                self.W_list.append(W_layer)

            if len(self.W_available) == 1:
                self.W_core = W_layer
            else:
                self.W_list = torch.stack(self.W_list, dim = 2)
                W_sig_softmax = nn.Softmax(dim = -1)(self.W_sig.unsqueeze(0))
                self.W_core = torch.matmul(self.W_list, W_sig_softmax.transpose(1,0)).squeeze(2)
    
        # Superimpose different biases:
        if "bias" in source:
            self.b_list = []
            for bias_type in self.b_available:
                if bias_type == "None":
                    if self.is_cuda:
                        b_layer = Variable(torch.zeros(self.output_size).cuda(), requires_grad = False)
                    else:
                        b_layer = Variable(torch.zeros(self.output_size), requires_grad = False)
                elif bias_type == "constant":
                    b_layer = self.b_layer_seed[0].repeat(self.output_size)
                elif bias_type == "arithmetic-series":
                    mean = self.b_layer_seed.mean()
                    offset = self.output_size / float(2) - 0.5
                    if self.is_cuda:
                        idx = Variable(torch.FloatTensor(range(self.output_size)).cuda(), requires_grad = False)
                    else:
                        idx = Variable(torch.FloatTensor(range(self.output_size)), requires_grad = False)
                    b_layer = mean + self.b_interval * (idx - offset)
                elif bias_type == "arithmetic-series-2D":
                    idx_i, idx_j = np.meshgrid(range(self.output_size_2D[0]), range(self.output_size_2D[1]), indexing = "ij")
                    idx_i = torch.from_numpy(idx_i).float().view(-1)
                    idx_j = torch.from_numpy(idx_j).float().view(-1)
                    if self.is_cuda:
                        idx_i, idx_j = idx_i.cuda(), idx_j.cuda()
                    offset_i = self.output_size_2D[0] / float(2) - 0.5
                    offset_j = self.output_size_2D[1] / float(2) - 0.5
                    b_layer = (self.b_mean_2D +                                self.b_interval_2D[0] * Variable(idx_i - offset_i, requires_grad = False) +                                self.b_interval_2D[1] * Variable(idx_j - offset_j, requires_grad = False)).view(-1)
                elif bias_type == "dense":
                    b_layer = self.b_layer_seed
                else:
                    raise Exception("bias_type '{0}' not recognized!".format(bias_type))
                self.b_list.append(b_layer)

            if len(self.b_available) == 1:
                self.b_core = b_layer
            else:
                self.b_list = torch.stack(self.b_list, dim = 1)
                b_sig_softmax = nn.Softmax(dim = -1)(self.b_sig.unsqueeze(0))
                self.b_core = torch.matmul(self.b_list, b_sig_softmax.transpose(1,0)).squeeze(1)


    def forward(self, X):
        output = X
        if hasattr(self, "input_size_original"):
            output = output.view(-1, self.input_size)
        # Get superposition of layers:
        self.get_layers(source = ["weight", "bias"])

        # Perform dot(X, W) + b:
        output = torch.matmul(output, self.W_core) + self.b_core
        
        # Exert superposition of activation functions:
        if len(self.A_available) == 1:
            output = get_activation(self.A_available[0])(output)
        else:
            self.A_list = []
            A_sig_softmax = nn.Softmax(dim = -1)(self.A_sig.unsqueeze(0))
            for i, activation in enumerate(self.A_available):
                A = get_activation(activation)(output)
                self.A_list.append(A)
            self.A_list = torch.stack(self.A_list, 2)
            output = torch.matmul(self.A_list, A_sig_softmax.transpose(1,0)).squeeze(2)

        if hasattr(self, "output_size_original"):
            output = output.view(*((-1,) + self.output_size_original))
        return output
    
    
    def get_param_names(self, source):
        if source == "modules":
            param_names = ["W_layer_seed", "b_layer_seed"]
            if "arithmetic-series-in" in self.W_available:
                param_names.append("W_interval_j")
            if "arithmetic-series-out" in self.W_available:
                param_names.append("W_interval_i")
            if "arithmetic-series-2D-in" in self.W_available:
                param_names = param_names + ["W_mean_2D_in", "W_interval_2D_in"]
            if "arithmetic-series-2D-out" in self.W_available:
                param_names = param_names + ["W_mean_2D_out", "W_interval_2D_out"]
            if "arithmetic-series" in self.b_available:
                param_names.append("b_interval")
        if source == "attention":
            param_names = ["W_sig", "b_sig", "A_sig"]
        return param_names
    
    
    def get_weights_bias(self):
        self.get_layers(source = ["weight", "bias"])
        W_core, b_core = self.W_core, self.b_core
        if self.is_cuda:
            W_core = W_core.cpu()
            b_core = b_core.cpu()
        return deepcopy(W_core.data.numpy()), deepcopy(b_core.data.numpy())


    def get_regularization(self, mode, source = ["weight", "bias"]):
        reg = Variable(torch.FloatTensor(np.array([0])), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        if not isinstance(source, list):
            source = [source]
        if mode == "L1":
            if "weight" in source:
                reg = reg + self.W_core.abs().sum()
            if "bias" in source:
                reg = reg + self.b_core.abs().sum()
        elif mode == "layer_L1":
            if "weight" in source:
                self.get_layers(source = ["weight"])
                reg = reg + self.W_list.abs().sum()
            if "bias" in source:
                self.get_layers(source = ["bias"])
                reg = reg + self.b_list.abs().sum()
        elif mode == "L2":
            if "weight" in source:
                reg = reg + torch.sum(self.W_core ** 2)
            if "bias" in source:
                reg = reg + torch.sum(self.b_core ** 2)
        elif mode == "S_entropy":
            if "weight" in source:
                W_sig_softmax = nn.Softmax(dim = -1)(self.W_sig.unsqueeze(0))
                reg = reg - torch.sum(W_sig_softmax * torch.log(W_sig_softmax))
            if "bias" in source:
                b_sig_softmax = nn.Softmax(dim = -1)(self.b_sig.unsqueeze(0))
                reg = reg - torch.sum(b_sig_softmax * torch.log(b_sig_softmax))
        elif mode == "S_entropy_activation":
            A_sig_softmax = nn.Softmax(dim = -1)(self.A_sig.unsqueeze(0))
            reg = reg - torch.sum(A_sig_softmax * torch.log(A_sig_softmax))
        elif mode in AVAILABLE_REG:
            pass
        else:
            raise Exception("mode '{0}' not recognized!".format(mode))
        return reg

