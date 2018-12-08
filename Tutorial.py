#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils
import matplotlib.pylab as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
from pytorch_net.net import MLP, ConvNet, load_model_dict_net, train
from pytorch_net.util import Early_Stopping


# ## Preparing dataset:

# In[2]:


# Preparing some toy dataset:
X = np.random.randn(1000,1)
y = X ** 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train = Variable(torch.FloatTensor(X_train))
y_train = Variable(torch.FloatTensor(y_train))
X_test = Variable(torch.FloatTensor(X_test))
y_test = Variable(torch.FloatTensor(y_test))


# ## Constuct the network:

# In[3]:


# Constuct the network:
input_size = 1
struct_param = [
    [2, "Simple_Layer", {}],   # (number of neurons in each layer, layer_type, layer settings)
    [400, "Simple_Layer", {"activation": "relu"}],
    [1, "Simple_Layer", {"activation": "linear"}],
]
settings = {"activation": "relu"} # Default activation if the activation is not specified in "struct_param" in each layer.
                                    # If the activation is specified, it will overwrite this default settings.

net = MLP(input_size = input_size,
          struct_param = struct_param,
          settings = settings,
         )


# In[ ]:


# Get the prediction of the Net:
net(X_train)


# In[ ]:


# Get intermediate activation of the net:
net.inspect_operation(X_train, operation_between = (0,2))


# ## Training using explicit commands:

# In[4]:


# training settings:
batch_size = 128
epochs = 500

# Prepare training set batches:
dataset_train = data_utils.TensorDataset(X_train.data, y_train.data)   #  The data_loader must use the torch Tensor, not Variable. So I use X_train.data to get the Tensor.
train_loader = data_utils.DataLoader(dataset_train, batch_size = batch_size, shuffle = True)

# Set up optimizer:
optimizer = optim.Adam(net.parameters(), lr = 1e-3)
# Get loss function. Choose from "mse" or "huber", etc.
criterion = nn.MSELoss()
# Set up early stopping. If the validation loss does not go down after "patience" number of epochs, early stop.
early_stopping = Early_Stopping(patience = 10) 


# In[ ]:


to_stop = False
for epoch in range(epochs):
    for batch_id, (X_batch, y_batch) in enumerate(train_loader):
        # Every learning step must contain the following 5 steps:
        optimizer.zero_grad()   # Zero out the gradient buffer
        pred = net(Variable(X_batch))   # Obtain network's prediction
        loss_train = criterion(pred, Variable(y_batch))  # Calculate the loss
        loss_train.backward()    # Perform backward step on the loss to calculate the gradient for each parameter
        optimizer.step()         # Use the optimizer to perform one step of parameter update
        
    # Validation at the end of each epoch:
    loss_test = criterion(net(X_test), y_test)
    to_stop = early_stopping.monitor(loss_test.data[0])
    print("epoch {0} \tbatch {1} \tloss_train: {2:.6f}\tloss_test: {3:.6f}".format(epoch, batch_id, loss_train.data[0], loss_test.data[0]))
    if to_stop:
        print("Early stopping at epoch {0}".format(epoch))
        break


# In[6]:


# Save model:
pickle.dump(net.model_dict, open("net.p", "wb"))

# Load model:
net_loaded = load_model_dict_net(pickle.load(open("net.p", "rb")))

# Check the loaded net and the original net is identical:
net_loaded(X_train) - net(X_train)


# ## Advanced example: training MNIST using given train() function:

# In[ ]:


from torchvision import datasets
import torch.utils.data as data_utils
from pytorch_net.util import train_test_split, normalize_tensor, to_Variable
is_cuda = torch.cuda.is_available()

struct_param_conv = [
    [64, "Conv2d", {"kernel_size": 3, "padding": 1}],
    [64, "BatchNorm2d", {"activation": "relu"}],
    [None, "MaxPool2d", {"kernel_size": 2}],

    [64, "Conv2d", {"kernel_size": 3, "padding": 1}],
    [64, "BatchNorm2d", {"activation": "relu"}],
    [None, "MaxPool2d", {"kernel_size": 2}],

    [64, "Conv2d", {"kernel_size": 3, "padding": 1}],
    [64, "BatchNorm2d", {"activation": "relu"}],
    [None, "MaxPool2d", {"kernel_size": 2}],

    [64, "Conv2d", {"kernel_size": 3, "padding": 1}],
    [64, "BatchNorm2d", {"activation": "relu"}],
    [None, "MaxPool2d", {"kernel_size": 2}],
    [10, "Simple_Layer", {"layer_input_size": 64}],
]

model = ConvNet(input_channels = 1,
                struct_param = struct_param_conv,
                is_cuda = is_cuda,
               )


# In[ ]:


dataset_name = "MNIST"
batch_size = 128
dataset_raw = getattr(datasets, dataset_name)('datasets/{0}'.format(dataset_name), download = True)
X, y = to_Variable(dataset_raw.train_data.unsqueeze(1).float(), dataset_raw.train_labels, is_cuda = is_cuda)
X = normalize_tensor(X, new_range = (0, 1))
(X_train, y_train), (X_test, y_test) = train_test_split(X, y, test_size = 0.2) # Split into training and testing
dataset_train = data_utils.TensorDataset(X_train, y_train)
train_loader = data_utils.DataLoader(dataset_train, batch_size = batch_size, shuffle = True) # initialize the dataLoader


# In[ ]:


lr = 1e-3
reg_dict = {"weight": 1e-6, "bias": 1e-6}
loss_original, loss_value, data_record = train(model, 
                                               train_loader = train_loader,
                                               validation_data = (X_test, y_test),
                                               criterion = nn.CrossEntropyLoss(),
                                               lr = lr,
                                               reg_dict = reg_dict,
                                               epochs = 1000,
                                               isplot = True,
                                               patience = 40,
                                               scheduler_patience = 40,
                                               inspect_items = ["accuracy"],
                                               record_keys = ["accuracy"],
                                               inspect_interval = 1,
                                               inspect_items_interval = 1,
                                               inspect_loss_precision = 4,
                                              )

