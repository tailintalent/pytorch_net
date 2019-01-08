
# coding: utf-8

# In[ ]:


from __future__ import print_function
import numpy as np
from copy import deepcopy
import pickle
from collections import OrderedDict
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pytorch_net.modules import get_Layer, load_layer_dict
from pytorch_net.util import get_activation, get_criterion, get_optimizer, get_full_struct_param, plot_matrices, Early_Stopping, record_data, to_np_array, to_Variable, make_dir


# In[ ]:


def get_accuracy(pred, target):
    """Get accuracy from prediction and target"""
    assert len(pred.shape) == len(target.shape) == 1
    assert len(pred) == len(target)
    pred, target = to_np_array(pred, target)
    accuracy = ((pred == target).sum().astype(float) / len(pred))
    return accuracy


def flatten(*tensors):
    """Flatten the tensor except the first dimension"""
    new_tensors = []
    for tensor in tensors:
        new_tensors.append(tensor.view(tensor.size(0), -1))
    if len(new_tensors) == 1:
        new_tensors = new_tensors[0]
    return new_tensors


def get_loss(model, data_loader = None, X = None, y = None, criterion = None):
    """Get loss using the whole data or data_loader"""
    if data_loader is not None:
        assert X is None and y is None
        loss_list = []
        for X_batch, y_batch in data_loader:
            loss_ele = model.get_loss(X_batch, y_batch, criterion = criterion)
            loss_list.append(loss_ele)
        loss = torch.stack(loss_list).mean()
    else:
        assert X is not None and y is not None
        loss = model.get_loss(X, y, criterion = criterion)
    return loss


def train(model, X = None, y = None, train_loader = None, validation_data = None, validation_loader = None, criterion = nn.MSELoss(), inspect_interval = 10, isplot = False, **kwargs):
    """minimal version of training. "model" can be a single model or a ordered list of models"""
    def get_regularization(model, **kwargs):
        reg_dict = kwargs["reg_dict"] if "reg_dict" in kwargs else {"weight": 0, "bias": 0}
        reg = to_Variable([0], is_cuda = is_cuda)
        for reg_type, reg_coeff in reg_dict.items():
            reg = reg + model.get_regularization(source = reg_type, mode = "L1", **kwargs) * reg_coeff
        return reg
    if X is None and y is None:
        assert train_loader is not None
        is_cuda = train_loader.dataset.tensors[0].is_cuda
    else:
        is_cuda = X.is_cuda
    epochs = kwargs["epochs"] if "epochs" in kwargs else 10000
    lr = kwargs["lr"] if "lr" in kwargs else 5e-3
    optim_type = kwargs["optim_type"] if "optim_type" in kwargs else "adam"
    optim_kwargs = kwargs["optim_kwargs"] if "optim_kwargs" in kwargs else {}
    patience = kwargs["patience"] if "patience" in kwargs else 20
    record_keys = kwargs["record_keys"] if "record_keys" in kwargs else ["loss"]
    scheduler_type = kwargs["scheduler_type"] if "scheduler_type" in kwargs else "ReduceLROnPlateau"
    inspect_items = kwargs["inspect_items"] if "inspect_items" in kwargs else None
    inspect_items_interval = kwargs["inspect_items_interval"] if "inspect_items_interval" in kwargs else 1000
    inspect_loss_precision = kwargs["inspect_loss_precision"] if "inspect_loss_precision" in kwargs else 4
    filename = kwargs["filename"] if "filename" in kwargs else None
    if filename is not None:
        make_dir(filename)
    save_interval = kwargs["save_interval"] if "save_interval" in kwargs else None
    logdir = kwargs["logdir"] if "logdir" in kwargs else None
    data_record = {key: [] for key in record_keys}
    if patience is not None:
        early_stopping_epsilon = kwargs["early_stopping_epsilon"] if "early_stopping_epsilon" in kwargs else 0
        early_stopping_monitor = kwargs["early_stopping_monitor"] if "early_stopping_monitor" in kwargs else "loss"
        early_stopping = Early_Stopping(patience = patience, epsilon = early_stopping_epsilon, mode = "max" if early_stopping_monitor in ["accuracy"] else "min")
    if logdir is not None:
        from pytorch_net.logger import Logger
        batch_idx = 0
        logger = Logger(logdir)
    
    if validation_loader is not None:
        assert validation_data is None
        X_valid, y_valid = None, None
    elif validation_data is not None:
        X_valid, y_valid = validation_data
    else:
        X_valid, y_valid = X, y
    
    # Get original loss:
    
    loss_original = get_loss(model, validation_loader, X_valid, y_valid, criterion = criterion).item()
    
    if "loss" in record_keys:
        record_data(data_record, [-1, get_loss(model, validation_loader, X_valid, y_valid, criterion = criterion).item()], ["iter", "loss"])
    if "param" in record_keys:
        record_data(data_record, [model.get_weights_bias(W_source = "core", b_source = "core")], ["param"])
    if "param_grad" in record_keys:
        record_data(data_record, [model.get_weights_bias(W_source = "core", b_source = "core", is_grad = True)], ["param_grad"])
    if inspect_items is not None:
        print("{0}:".format(-1), end = "")
        print("\tlr: {0}\t loss:{1:.{2}f}".format(lr, loss_original, inspect_loss_precision), end = "")
        if hasattr(model, "prepare_inspection"):
            model.prepare_inspection(X_valid, y_valid)
        if hasattr(model, "info_dict"):
            for item in inspect_items:
                if item in model.info_dict:
                    print(" \t{0}: {1:.{2}f}".format(item, model.info_dict[item], inspect_loss_precision), end = "")
                    if item in record_keys and item != "loss":
                        record_data(data_record, [to_np_array(model.info_dict[item])], [item])
        print()

    # Setting up optimizer:
    parameters = model.parameters()
    num_params = len(list(model.parameters()))
    if num_params == 0:
        print("No parameters to optimize!")
        loss_value = get_loss(model, validation_loader, X_valid, y_valid, criterion = criterion).item()
        if "loss" in record_keys:
            record_data(data_record, [0, get_loss(model, validation_loader, X_valid, y_valid, criterion = criterion).item()], ["iter", "loss"])
        if "param" in record_keys:
            record_data(data_record, [model.get_weights_bias(W_source = "core", b_source = "core")], ["param"])
        if "param_grad" in record_keys:
            record_data(data_record, [model.get_weights_bias(W_source = "core", b_source = "core", is_grad = True)], ["param_grad"])
        return loss_original, loss_value, data_record
    optimizer = get_optimizer(optim_type, lr, parameters, **optim_kwargs)
    
    # Set up learning rate scheduler:
    if scheduler_type is not None:
        if scheduler_type == "ReduceLROnPlateau":
            scheduler_patience = kwargs["scheduler_patience"] if "scheduler_patience" in kwargs else 40
            scheduler_factor = kwargs["scheduler_factor"] if "scheduler_factor" in kwargs else 0.1
            scheduler_verbose = kwargs["scheduler_verbose"] if "scheduler_verbose" in kwargs else False
            scheduler = ReduceLROnPlateau(optimizer, factor = scheduler_factor, patience = scheduler_patience, verbose = scheduler_verbose)
        elif scheduler_type == "LambdaLR":
            scheduler_lr_lambda = kwargs["scheduler_lr_lambda"] if "scheduler_lr_lambda" in kwargs else (lambda epoch: 1 / (1 + 0.01 * epoch))
            scheduler = LambdaLR(optimizer, lr_lambda = scheduler_lr_lambda)
        else:
            raise

    # Training:
    to_stop = False
    for i in range(epochs + 1):
        model.train()
        if X is not None and y is not None:
            if optim_type != "LBFGS":
                optimizer.zero_grad()
                reg = get_regularization(model, **kwargs)
                loss = model.get_loss(X, y, criterion = criterion, **kwargs) + reg
                loss.backward()
                optimizer.step()
            else:
                # "LBFGS" is a second-order optimization algorithm that requires a slightly different procedure:
                def closure():
                    optimizer.zero_grad()
                    reg = get_regularization(model, **kwargs)
                    loss = model.get_loss(X, y, criterion = criterion, **kwargs) + reg
                    loss.backward()
                    return loss
                optimizer.step(closure)
        else:
            for _, (X_batch, y_batch) in enumerate(train_loader):
                if optim_type != "LBFGS":
                    optimizer.zero_grad()
                    reg = get_regularization(model, **kwargs)
                    loss = model.get_loss(X_batch, y_batch, criterion = criterion, **kwargs) + reg
                    loss.backward()
                    optimizer.step()
                else:
                    def closure():
                        optimizer.zero_grad()
                        reg = get_regularization(model, **kwargs)
                        loss = model.get_loss(X_batch, y_batch, criterion = criterion, **kwargs) + reg
                        loss.backward()
                        return loss
                    optimizer.step(closure)
                
                if logdir is not None:
                    batch_idx += 1
                    if hasattr(model, "info_dict"):
                        for item in inspect_items:
                            if item in model.info_dict:
                                logger.log_scalar(item, model.info_dict[item], batch_idx)

                    # 2. Log values and gradients of the parameters (histogram summary)
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.log_histogram(tag, to_np_array(value), batch_idx)
                        logger.log_histogram(tag + '/grad', to_np_array(value.grad), batch_idx)

        if i % inspect_interval == 0:
            model.eval()
            loss_value = get_loss(model, validation_loader, X_valid, y_valid, criterion = criterion).item()
            if scheduler_type is not None:
                if scheduler_type == "ReduceLROnPlateau":
                    scheduler.step(loss_value)
                else:
                    scheduler.step()
            if patience is not None:
                if early_stopping_monitor == "loss":
                    to_stop = early_stopping.monitor(loss_value)
                else:
                    model.prepare_inspection(X_valid, y_valid)
                    to_stop = early_stopping.monitor(model.info_dict[early_stopping_monitor])
            if inspect_items is not None:
                if i % inspect_items_interval == 0:
                    print("{0}:".format(i), end = "")
                    print("\tlr: {0:.3e}\tloss: {1:.{2}f}".format(optimizer.param_groups[0]["lr"], loss_value, inspect_loss_precision), end = "")
                    if hasattr(model, "prepare_inspection"):
                        model.prepare_inspection(X_valid, y_valid)
                    if hasattr(model, "info_dict"):
                        for item in inspect_items:
                            if item in model.info_dict:
                                print(" \t{0}: {1:.{2}f}".format(item, model.info_dict[item], inspect_loss_precision), end = "")
                                if item in record_keys and item != "loss":
                                    record_data(data_record, [to_np_array(model.info_dict[item])], [item])
                    if "loss" in record_keys:
                        record_data(data_record, [i, loss_value], ["iter", "loss"])
                    if "param" in record_keys:
                        record_data(data_record, [model.get_weights_bias(W_source = "core", b_source = "core")], ["param"])
                    if "param_grad" in record_keys:
                        record_data(data_record, [model.get_weights_bias(W_source = "core", b_source = "core", is_grad = True)], ["param_grad"])
                    print()
        if filename is not None and save_interval is not None:
            if i % save_interval == 0:
                pickle.dump(model.model_dict, open(filename[:-2] + "_{0}".format(i) + ".p", "wb"))
        if to_stop:
            break

    loss_value = get_loss(model, validation_loader, X_valid, y_valid, criterion = criterion).item()
    if isplot:
        import matplotlib.pylab as plt
        for key in data_record:
            if key != "iter":
                if key in ["accuracy"]:
                    plt.figure(figsize = (8,6))
                    plt.plot(data_record["iter"], data_record[key])
                    plt.xlabel("epoch")
                    plt.ylabel(key)
                    plt.title(key)
                    plt.show()
                else:
                    plt.figure(figsize = (8,6))
                    plt.semilogy(data_record["iter"], data_record[key])
                    plt.xlabel("epoch")
                    plt.ylabel(key)
                    plt.title(key)
                    plt.show()
    return loss_original, loss_value, data_record


def load_model_dict_net(model_dict, is_cuda = False):
    net_type = model_dict["type"]
    if net_type == "MLP":
        return MLP(input_size = model_dict["input_size"],
                   struct_param = model_dict["struct_param"],
                   W_init_list = model_dict["weights"] if "weights" in model_dict else None,
                   b_init_list = model_dict["bias"] if "bias" in model_dict else None,
                   settings = model_dict["settings"] if "settings" in model_dict else {},
                   is_cuda = is_cuda,
                  )
    elif net_type == "Multi_MLP":
        return Multi_MLP(input_size = model_dict["input_size"],
                   struct_param = model_dict["struct_param"],
                   W_init_list = model_dict["weights"] if "weights" in model_dict else None,
                   b_init_list = model_dict["bias"] if "bias" in model_dict else None,
                   settings = model_dict["settings"] if "settings" in model_dict else {},
                   is_cuda = is_cuda,
                  )
    elif net_type == "ConvNet":
        return ConvNet(input_channels = model_dict["input_channels"],
                       struct_param = model_dict["struct_param"],
                       W_init_list = model_dict["weights"] if "weights" in model_dict else None,
                       b_init_list = model_dict["bias"] if "bias" in model_dict else None,
                       settings = model_dict["settings"] if "settings" in model_dict else {},
                       return_indices = model_dict["return_indices"] if "return_indices" in model_dict else False,
                       is_cuda = is_cuda,
                      )
    elif net_type == "Conv_Autoencoder":
        model = Conv_Autoencoder(input_channels_encoder = model_dict["input_channels_encoder"],
                                 input_channels_decoder = model_dict["input_channels_decoder"],
                                 struct_param_encoder = model_dict["struct_param_encoder"],
                                 struct_param_decoder = model_dict["struct_param_decoder"],
                                 settings = model_dict["settings"],
                                 is_cuda = is_cuda,
                                )
        if "encoder" in model_dict:
            model.encoder.load_model_dict(model_dict["encoder"])
        if "decoder" in model_dict:
            model.decoder.load_model_dict(model_dict["decoder"])
        return model
    else:
        raise Exception("net_type {0} not recognized!".format(net_type))
        

def load_model_dict(model_dict, is_cuda = False):
    net_type = model_dict["type"]
    if net_type not in ["Model_Ensemble", "LSTM"]:
        return load_model_dict_net(model_dict, is_cuda = is_cuda)
    elif net_type == "Model_Ensemble":
        if model_dict["model_type"] == "MLP":
            model_ensemble = Model_Ensemble(
                num_models = model_dict["num_models"],
                input_size = model_dict["input_size"],
                model_type = model_dict["model_type"],
                output_size = model_dict["output_size"],
                is_cuda = is_cuda,
                # Here we just create some placeholder network. The model will be overwritten in the next steps:
                struct_param = [[1, "Simple_Layer", {}]],
            )
        elif model_dict["model_type"] == "LSTM":
            model_ensemble = Model_Ensemble(
                num_models = model_dict["num_models"],
                input_size = model_dict["input_size"],
                model_type = model_dict["model_type"],
                output_size = model_dict["output_size"],
                is_cuda = is_cuda,
                # Here we just create some placeholder network. The model will be overwritten in the next steps:
                hidden_size = 3,
                output_struct_param = [[1, "Simple_Layer", {}]],
            )
        else:
            raise
        for k in range(model_ensemble.num_models):
            setattr(model_ensemble, "model_{0}".format(k), load_model_dict(model_dict["model_{0}".format(k)], is_cuda = is_cuda))
        return model_ensemble
    elif net_type == "Model_with_Uncertainty":
        return Model_with_Uncertainty(model_pred = load_model_dict(model_dict["model_pred"], is_cuda = is_cuda),
                                      model_logstd = load_model_dict(model_dict["model_logstd"], is_cuda = is_cuda))
    else:
        raise Exception("net_type {0} not recognized!".format(net_type))


# ## Model_Ensemble:

# In[2]:


class Model_Ensemble(nn.Module):
    """Model_Ensemble is a collection of models with the same architecture 
       but independent parameters"""
    def __init__(
        self,
        num_models,
        input_size,
        model_type,
        is_cuda = False,
        **kwargs
        ):
        super(Model_Ensemble, self).__init__()
        self.num_models = num_models
        self.input_size = input_size
        self.is_cuda = is_cuda
        for i in range(self.num_models):
            if model_type == "MLP":
                model = MLP(input_size = self.input_size, is_cuda = is_cuda, **kwargs)
            elif model_type == "LSTM":
                model = LSTM(input_size = self.input_size, is_cuda = is_cuda, **kwargs)
            else:
                raise Exception("Net_type {0} not recognized!".format(net_type))
            setattr(self, "model_{0}".format(i), model)


    def get_all_models(self):
        return [getattr(self, "model_{0}".format(i)) for i in range(self.num_models)]


    def forward(self, input):
        output_list = []
        for i in range(self.num_models):
            output = getattr(self, "model_{0}".format(i))(input)
            if output.size(-1) == 1:
                output = output.squeeze(1)
            output_list.append(output)
        return torch.stack(output_list, -1)


    def get_loss(self, input, target, criterion, **kwargs):
        y_pred = self(input, **kwargs)
        return criterion(y_pred, target)


    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        reg = Variable(torch.FloatTensor([0]), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        for k in range(self.num_models):
            reg = reg + getattr(self, "model_{0}".format(k)).get_regularization(
                source = source, mode = mode, **kwargs)
        return reg


    def remove_models(self, model_ids):
        if not isinstance(model_ids, list):
            model_ids = [model_ids]
        model_list = []
        k = 0
        for i in range(self.num_models):
            if i not in model_ids:
                if k != i:
                    setattr(self, "model_{0}".format(k), getattr(self, "model_{0}".format(i)))
                k += 1
        num_models_new = k
        for i in range(num_models_new, self.num_models):
            delattr(self, "model_{0}".format(i))
        self.num_models = num_models_new


    def add_models(self, models):
        if not isinstance(models, list):
            models = [models]
        for i, model in enumerate(models):
            setattr(self, "model_{0}".format(i + self.num_models), model)
        self.num_models += len(models)


    def get_weights_bias(self, W_source = None, b_source = None, verbose = False, isplot = False):
        W_list_dict = {}
        b_list_dict = {}
        for i in range(self.num_models):
            if verbose:
                print("\nmodel {0}:".format(i))
            W_list_dict[i], b_list_dict[i] = getattr(self, "model_{0}".format(i)).get_weights_bias(
                W_source = W_source, b_source = b_source, verbose = verbose, isplot = isplot)
        return W_list_dict, b_list_dict


    def prepare_inspection(self, X, y):
        pass
    
    
    def set_cuda(self, is_cuda):
        for k in range(self.num_models):
            getattr(self, "model_{0}".format(k)).set_cuda(is_cuda)
        self.is_cuda = is_cuda
    
    
    def set_trainable(self, is_trainable):
        for i in range(self.num_models):
            getattr(self, "model_{0}".format(i)).set_trainable(is_trainable)
    

class Model_with_uncertainty(nn.Module):
    def __init__(
        self,
        model_pred,
        model_logstd,
        ):
        super(Model_with_uncertainty, self).__init__()
        self.model_pred = model_pred
        self.model_logstd = model_logstd
        
    def forward(self, input, noise_amp = None, **kwargs):
        return self.model_pred(input, noise_amp = noise_amp, **kwargs), self.model_logstd(input, **kwargs)
    
    def get_loss(self, input, target, criterion, noise_amp = None, **kwargs):
        pred, log_std = self(input, noise_amp = noise_amp, **kwargs)
        return criterion(pred = pred, target = target, log_std = log_std)
    
    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        return self.model_pred.get_regularization(source = source, mode = mode, **kwargs) +                 self.model_logstd.get_regularization(source = source, mode = mode, **kwargs)
    
    @property
    def model_dict(self):
        model_dict = {}
        model_dict["type"] = "Model_with_Uncertainty"
        model_dict["model_pred"] = self.model_pred.model_dict
        model_dict["model_logstd"] = self.model_logstd.model_dict
        return model_dict

    def set_cuda(self, is_cuda):
        self.model_pred.set_cuda(is_cuda)
        self.model_logstd.set_cuda(is_cuda)
        
    def set_trainable(self, is_trainable):
        self.model_pred.set_trainable(is_trainable)
        self.model_logstd.set_trainable(is_trainable)


# ## Multi_MLP:

# In[ ]:


class Multi_MLP(nn.Module):
    def __init__(
        self,
        input_size,
        struct_param,
        W_init_list = None,     # initialization for weights
        b_init_list = None,     # initialization for bias
        settings = None,          # Default settings for each layer, if the settings for the layer is not provided in struct_param
        is_cuda = False,
        ):
        super(Multi_MLP, self).__init__()
        self.input_size = input_size
        self.num_layers = len(struct_param)
        self.W_init_list = W_init_list
        self.b_init_list = b_init_list
        self.settings = deepcopy(settings)
        self.num_blocks = len(struct_param)
        self.is_cuda = is_cuda
        
        for i, struct_param_ele in enumerate(struct_param):
            input_size_block = input_size if i == 0 else struct_param[i - 1][-1][0]
            setattr(self, "block_{0}".format(i), MLP(input_size = input_size_block,
                                                     struct_param = struct_param_ele,
                                                     W_init_list = W_init_list[i] if W_init_list is not None else None,
                                                     b_init_list = b_init_list[i] if b_init_list is not None else None,
                                                     settings = self.settings[i] if self.settings is not None else {},
                                                     is_cuda = self.is_cuda,
                                                    ))
    
    def forward(self, input):
        output = input
        for i in range(self.num_blocks):
            output = getattr(self, "block_{0}".format(i))(output)
        return output


    def get_loss(self, input, target, criterion, **kwargs):
        y_pred = self(input)
        return criterion(y_pred, target)


    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        reg = Variable(torch.FloatTensor([0]), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        for i in range(self.num_blocks):
            reg = reg + getattr(self, "block_{0}".format(i)).get_regularization(mode = mode, source = source)
        return reg


    @property
    def struct_param(self):
        return [getattr(self, "block_{0}".format(i)).struct_param for i in range(self.num_blocks)]


    @property
    def model_dict(self):
        model_dict = {"type": "Multi_MLP"}
        model_dict["input_size"] = self.input_size
        model_dict["struct_param"] = self.struct_param
        model_dict["weights"], model_dict["bias"] = self.get_weights_bias(W_source = "core", b_source = "core")
        model_dict["settings"] = deepcopy(self.settings)
        model_dict["net_type"] = "Multi_MLP"
        return model_dict


    def load_model_dict(self, model_dict):
        new_net = load_model_dict_Multi_MLP(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_net.__dict__)


    def get_weights_bias(self, W_source = "core", b_source = "core"):
        W_list = []
        b_list = []
        for i in range(self.num_blocks):
            W, b = getattr(self, "block_{0}".format(i)).get_weights_bias(W_source = W_source, b_source = b_source)
            W_list.append(W)
            b_list.append(b)
        return deepcopy(W_list), deepcopy(b_list)


    def prepare_inspection(self, X, y):
        pass


    def set_cuda(self, is_cuda):
        for i in range(self.num_blocks):
            getattr(self, "block_{0}".format(i)).set_cuda(is_cuda)
        self.is_cuda = is_cuda


    def set_trainable(self, is_trainable):
        for i in range(self.num_blocks):
            getattr(self, "block_{0}".format(i)).set_trainable(is_trainable)


# ## MLP:

# In[3]:


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        struct_param,
        W_init_list = None,     # initialization for weights
        b_init_list = None,     # initialization for bias
        settings = {},          # Default settings for each layer, if the settings for the layer is not provided in struct_param
        is_cuda = False,
        ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.num_layers = len(struct_param)
        self.W_init_list = W_init_list
        self.b_init_list = b_init_list
        self.settings = deepcopy(settings)
        self.is_cuda = is_cuda
        
        self.init_layers(deepcopy(struct_param))


    @property
    def struct_param(self):
        return [getattr(self, "layer_{0}".format(i)).struct_param for i in range(self.num_layers)]


    def init_layers(self, struct_param):
        res_forward = self.settings["res_forward"] if "res_forward" in self.settings else False
        for k, layer_struct_param in enumerate(struct_param):
            if res_forward:
                num_neurons_prev = struct_param[k - 1][0] + self.input_size if k > 0 else self.input_size
            else:
                num_neurons_prev = struct_param[k - 1][0] if k > 0 else self.input_size
            num_neurons = layer_struct_param[0]
            W_init = self.W_init_list[k] if self.W_init_list is not None else None
            b_init = self.b_init_list[k] if self.b_init_list is not None else None

            # Get settings for the current layer:
            layer_settings = deepcopy(self.settings) if bool(self.settings) else {}
            layer_settings.update(layer_struct_param[2])            

            # Construct layer:
            layer = get_Layer(layer_type = layer_struct_param[1],
                              input_size = num_neurons_prev,
                              output_size = num_neurons,
                              W_init = W_init,
                              b_init = b_init,
                              settings = layer_settings,
                              is_cuda = self.is_cuda,
                             )
            setattr(self, "layer_{0}".format(k), layer)


    def forward(self, input):
        output = input
        res_forward = self.settings["res_forward"] if "res_forward" in self.settings else False
        is_res_block = self.settings["is_res_block"] if "is_res_block" in self.settings else False
        for k in range(len(self.struct_param)):
            if res_forward and k > 0:
                output = getattr(self, "layer_{0}".format(k))(torch.cat([output, input], -1))
            else:
                output = getattr(self, "layer_{0}".format(k))(output)
        if is_res_block:
            output = output + input
        return output


    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        reg = Variable(torch.FloatTensor([0]), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        for k in range(len(self.struct_param)):
            layer = getattr(self, "layer_{0}".format(k))
            reg = reg + layer.get_regularization(mode = mode, source = source)
        return reg


    def reset_layer(self, layer_id, layer):
        setattr(self, "layer_{0}".format(layer_id), layer)


    def insert_layer(self, layer_id, layer):
        if layer_id < 0:
            layer_id += self.num_layers
        if layer_id < self.num_layers - 1:
            next_layer = getattr(self, "layer_{0}".format(layer_id + 1))
            if next_layer.struct_param[1] == "Simple_Layer":
                assert next_layer.input_size == layer.output_size, "The inserted layer's output_size {0} must be compatible with next layer_{1}'s input_size {2}!"                    .format(layer.output_size, layer_id + 1, next_layer.input_size)
        for i in range(self.num_layers - 1, layer_id - 1, -1):
            setattr(self, "layer_{0}".format(i + 1), getattr(self, "layer_{0}".format(i)))
        setattr(self, "layer_{0}".format(layer_id), layer)
        self.num_layers += 1
    
    
    def remove_layer(self, layer_id):
        if layer_id < 0:
            layer_id += self.num_layers
        if layer_id < self.num_layers - 1:
            num_neurons_prev = self.struct_param[layer_id - 1][0] if layer_id > 0 else self.input_size
            replaced_layer = getattr(self, "layer_{0}".format(layer_id + 1))
            if replaced_layer.struct_param[1] == "Simple_Layer":
                assert replaced_layer.input_size == num_neurons_prev,                     "After deleting layer_{0}, the replaced layer's input_size {1} must be compatible with previous layer's output neurons {2}!"                        .format(layer_id, replaced_layer.input_size, num_neurons_prev)
        for i in range(layer_id, self.num_layers - 1):
            setattr(self, "layer_{0}".format(i), getattr(self, "layer_{0}".format(i + 1)))
        self.num_layers -= 1


    def prune_neurons(self, layer_id, neuron_ids):
        if layer_id < 0:
            layer_id = self.num_layers + layer_id
        layer = getattr(self, "layer_{0}".format(layer_id))
        layer.prune_output_neurons(neuron_ids)
        self.reset_layer(layer_id, layer)
        if layer_id < self.num_layers - 1:
            next_layer = getattr(self, "layer_{0}".format(layer_id + 1))
            next_layer.prune_input_neurons(neuron_ids)
            self.reset_layer(layer_id + 1, next_layer)


    def add_neurons(self, layer_id, num_neurons, mode = ("imitation", "zeros")):
        if not isinstance(mode, list) and not isinstance(mode, tuple):
            mode = (mode, mode)
        if layer_id < 0:
            layer_id = self.num_layers + layer_id
        layer = getattr(self, "layer_{0}".format(layer_id))
        layer.add_output_neurons(num_neurons, mode = mode[0])
        self.reset_layer(layer_id, layer)
        if layer_id < self.num_layers - 1:
            next_layer = getattr(self, "layer_{0}".format(layer_id + 1))
            next_layer.add_input_neurons(num_neurons, mode = mode[1])
            self.reset_layer(layer_id + 1, next_layer)


    def inspect_operation(self, input, operation_between):
        res_forward = self.settings["res_forward"] if "res_forward" in self.settings else False
        output = input
        for k in range(*operation_between):
            output = getattr(self, "layer_{0}".format(k))(output)
            if res_forward and k > 0:
                output = getattr(self, "layer_{0}".format(k))(torch.cat([output, input], -1))
            else:
                output = getattr(self, "layer_{0}".format(k))(output)
        return output


    def get_weights_bias(self, W_source = None, b_source = None, layer_ids = None, isplot = False, raise_error = True):
        layer_ids = range(len(self.struct_param)) if layer_ids is None else layer_ids
        W_list = []
        b_list = []
        if W_source is not None:
            for k in range(len(self.struct_param)):
                if k in layer_ids:
                    if W_source == "core":
                        try:
                            W, _ = getattr(self, "layer_{0}".format(k)).get_weights_bias()
                        except Exception as e:
                            if raise_error:
                                raise
                            else:
                                print(e)
                            W = np.array([np.NaN])
                    else:
                        raise Exception("W_source '{0}' not recognized!".format(W_source))
                    W_list.append(W)
        
        if b_source is not None:
            for k in range(len(self.struct_param)):
                if k in layer_ids:
                    if b_source == "core":
                        try:
                            _, b = getattr(self, "layer_{0}".format(k)).get_weights_bias()
                        except Exception as e:
                            if raise_error:
                                raise
                            else:
                                print(e)
                            b = np.array([np.NaN])
                    else:
                        raise Exception("b_source '{0}' not recognized!".format(b_source))
                b_list.append(b)
                
        if isplot:
            if W_source is not None:
                print("weight {0}:".format(W_source))
                plot_matrices(W_list)
            if b_source is not None:
                print("bias {0}:".format(b_source))
                plot_matrices(b_list)
        return W_list, b_list
    
    
    @property
    def model_dict(self):
        model_dict = {"type": "MLP"}
        model_dict["input_size"] = self.input_size
        model_dict["struct_param"] = get_full_struct_param(self.struct_param, self.settings)
        model_dict["weights"], model_dict["bias"] = self.get_weights_bias(W_source = "core", b_source = "core")
        model_dict["settings"] = deepcopy(self.settings)
        model_dict["net_type"] = "MLP"
        return model_dict


    def load_model_dict(self, model_dict):
        new_net = load_model_dict_net(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_net.__dict__)


    def get_loss(self, input, target, criterion, **kwargs):
        y_pred = self(input)
        return criterion(y_pred, target)


    def prepare_inspection(self, X, y):
        pass


    def set_cuda(self, is_cuda):
        for k in range(self.num_layers):
            getattr(self, "layer_{0}".format(k)).set_cuda(is_cuda)
        self.is_cuda = is_cuda


    def set_trainable(self, is_trainable):
        for k in range(self.num_layers):
            getattr(self, "layer_{0}".format(k)).set_trainable(is_trainable)


# ## RNN:

# In[ ]:


class RNNCellBase(nn.Module):
    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))


class LSTM(RNNCellBase):
    """a LSTM class"""
    def __init__(
        self,
        input_size,
        hidden_size,
        output_struct_param,
        output_settings = {},
        bias = True,
        is_cuda = False,
        ):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.W_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.output_net = MLP(input_size = self.hidden_size, struct_param = output_struct_param, settings = output_settings, is_cuda = is_cuda)
        if bias:
            self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('b_ih', None)
            self.register_parameter('b_hh', None)
        self.reset_parameters()
        self.is_cuda = is_cuda
        self.device = torch.device("cuda" if self.is_cuda else "cpu")
        self.to(self.device)

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward_one_step(self, input, hx):
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return self._backend.LSTMCell(
            input, hx,
            self.W_ih, self.W_hh,
            self.b_ih, self.b_hh,
        )
    
    def forward(self, input, hx = None):
        if hx is None:
            hx = [torch.randn(input.size(0), self.hidden_size).to(self.device),
                  torch.randn(input.size(0), self.hidden_size).to(self.device),
                 ]
        hhx, ccx = hx
        for i in range(input.size(1)):
            hhx, ccx = self.forward_one_step(input[:, i], (hhx, ccx))
        output = self.output_net(hhx)
        return output

    def get_regularization(self, source, mode = "L1", **kwargs):
        if not isinstance(source, list):
            source = [source]
        reg = self.output_net.get_regularization(source = source, mode = mode)
        for source_ele in source:
            if source_ele == "weight":
                if mode == "L1":
                    reg = reg + self.W_ih.abs().sum() + self.W_hh.abs().sum()
                elif mode == "L2":
                    reg = reg + (self.W_ih ** 2).sum() + (self.W_hh ** 2).sum()
                else:
                    raise Exception("mode {0} not recognized!".format(mode))
            elif source_ele == "bias":
                if self.bias:
                    if mode == "L1":
                        reg = reg + self.b_ih.abs().sum() + self.b_hh.abs().sum()
                    elif mode == "L2":
                        reg = reg + (self.b_ih ** 2).sum() + (self.b_hh ** 2).sum()
                    else:
                        raise Exception("mode {0} not recognized!".format(mode))
            else:
                raise Exception("source {0} not recognized!".format(source_ele))
        return reg
    
    def get_weights_bias(self, W_source = None, b_source = None, verbose = False, isplot = False):
        W_dict = OrderedDict()
        b_dict = OrderedDict()
        W_o, b_o = self.output_net.get_weights_bias(W_source = W_source, b_source = b_source)
        if W_source == "core":
            W_dict["W_ih"] = self.W_ih.cpu().detach().numpy()
            W_dict["W_hh"] = self.W_hh.cpu().detach().numpy()
            W_dict["W_o"] = W_o
            if isplot:
                print("W_ih, W_hh:")
                plot_matrices([W_dict["W_ih"], W_dict["W_hh"]])
                print("W_o:")
                plot_matrices(W_o)
        if self.bias and b_source == "core":
            b_dict["b_ih"] = self.b_ih.cpu().detach().numpy()
            b_dict["b_hh"] = self.b_hh.cpu().detach().numpy()
            b_dict["b_o"] = b_o
            if isplot:
                print("b_ih, b_hh:")
                plot_matrices([b_dict["b_ih"], b_dict["b_hh"]])
                print("b_o:")
                plot_matrices(b_o)
        return W_dict, b_dict
    
    def get_loss(self, input, target, criterion, hx = None, **kwargs):
        y_pred = self(input, hx = hx)
        return criterion(y_pred, target)
    
    def prepare_inspection(self, X, y):
        pass


# ## CNN:

# In[ ]:


class ConvNet(nn.Module):
    def __init__(
        self,
        input_channels,
        struct_param,
        W_init_list = None,
        b_init_list = None,
        settings = {},
        return_indices = False,
        is_cuda = False,
        ):
        super(ConvNet, self).__init__()
        self.input_channels = input_channels
        self.struct_param = struct_param
        self.W_init_list = W_init_list
        self.b_init_list = b_init_list
        self.settings = settings
        self.num_layers = len(struct_param)
        self.info_dict = {}
        self.is_cuda = is_cuda
        self.param_available = ["Conv2d", "ConvTranspose2d", "BatchNorm2d", "Simple_Layer"]
        self.return_indices = return_indices
        for i in range(len(self.struct_param)):
            if i > 0:
                k = 1
                while self.struct_param[i - k][0] is None:
                    k += 1
                num_channels_prev = self.struct_param[i - k][0]
            else:
                num_channels_prev = input_channels
            num_channels = self.struct_param[i][0]
            layer_type = self.struct_param[i][1]
            layer_settings = self.struct_param[i][2]
            if "layer_input_size" in layer_settings and isinstance(layer_settings["layer_input_size"], tuple):
                num_channels_prev = layer_settings["layer_input_size"][0]
            if layer_type == "Conv2d":
                layer = nn.Conv2d(num_channels_prev, 
                                  num_channels,
                                  kernel_size = layer_settings["kernel_size"],
                                  stride = layer_settings["stride"] if "stride" in layer_settings else 1,
                                  padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                  dilation = layer_settings["dilation"] if "dilation" in layer_settings else 1,
                                 )
            elif layer_type == "ConvTranspose2d":
                layer = nn.ConvTranspose2d(num_channels_prev,
                                           num_channels,
                                           kernel_size = layer_settings["kernel_size"],
                                           stride = layer_settings["stride"] if "stride" in layer_settings else 1,
                                           padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                           dilation = layer_settings["dilation"] if "dilation" in layer_settings else 1,
                                          )
            elif layer_type == "Simple_Layer":
                layer = get_Layer(layer_type = layer_type,
                                  input_size = layer_settings["layer_input_size"],
                                  output_size = num_channels,
                                  W_init = W_init_list[i] if self.W_init_list is not None and self.W_init_list[i] is not None else None,
                                  b_init = b_init_list[i] if self.b_init_list is not None and self.b_init_list[i] is not None else None,
                                  settings = layer_settings,
                                  is_cuda = self.is_cuda,
                                 )
            elif layer_type == "MaxPool2d":
                layer = nn.MaxPool2d(kernel_size = layer_settings["kernel_size"],
                                     stride = layer_settings["stride"] if "stride" in layer_settings else None,
                                     padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                     return_indices = layer_settings["return_indices"] if "return_indices" in layer_settings else False,
                                    )
            elif layer_type == "MaxUnpool2d":
                layer = nn.MaxUnpool2d(kernel_size = layer_settings["kernel_size"],
                                       stride = layer_settings["stride"] if "stride" in layer_settings else None,
                                       padding = layer_settings["padding"] if "padding" in layer_settings else 0,
                                      )
            elif layer_type == "Upsample":
                layer = nn.Upsample(scale_factor = layer_settings["scale_factor"],
                                    mode = layer_settings["mode"] if "mode" in layer_settings else "nearest",
                                   )
            elif layer_type == "BatchNorm2d":
                layer = nn.BatchNorm2d(num_features = num_channels)
            elif layer_type == "Dropout2d":
                layer = nn.Dropout2d(p = 0.5)
            elif layer_type == "Flatten":
                layer = Flatten()
            else:
                raise Exception("layer_type {0} not recognized!".format(layer_type))
            
            # Initialize using provided initial values:
            if self.W_init_list is not None and self.W_init_list[i] is not None and layer_type not in ["Simple_Layer"]:
                layer.weight.data = torch.FloatTensor(self.W_init_list[i])
                layer.bias.data = torch.FloatTensor(self.b_init_list[i])
            
            setattr(self, "layer_{0}".format(i), layer)
        if self.is_cuda:
            self.cuda()


    def forward(self, input, indices_list = None, **kwargs):
        return self.inspect_operation(input, operation_between = (0, self.num_layers), indices_list = indices_list)
    
    
    def inspect_operation(self, input, operation_between, indices_list = None):
        output = input
        if indices_list is None:
            indices_list = []
        start_layer, end_layer = operation_between
        if end_layer < 0:
            end_layer += self.num_layers
        for i in range(start_layer, end_layer):
            if "layer_input_size" in self.struct_param[i][2]:
                output_size_last = output.shape[0]
                layer_input_size = self.struct_param[i][2]["layer_input_size"]
                if not isinstance(layer_input_size, tuple):
                    layer_input_size = (layer_input_size,)
                output = output.view(-1, *layer_input_size)
                assert output.shape[0] == output_size_last, "output_size reshaped to different length. Check shape!"
            if "Unpool" in self.struct_param[i][1]:
                output_tentative = getattr(self, "layer_{0}".format(i))(output, indices_list.pop(-1))
            else:
                output_tentative = getattr(self, "layer_{0}".format(i))(output)
            if isinstance(output_tentative, tuple):
                output, indices = output_tentative
                indices_list.append(indices)
            else:
                output = output_tentative
            if "activation" in self.struct_param[i][2]:
                activation = self.struct_param[i][2]["activation"]
            else:
                if "activation" in self.settings:
                    activation = self.settings["activation"]
                else:
                    activation = "linear"
                if "Pool" in self.struct_param[i][1] or "Unpool" in self.struct_param[i][1] or "Upsample" in self.struct_param[i][1]:
                    activation = "linear"
            output = get_activation(activation)(output)
        if self.return_indices:
            return output, indices_list
        else:
            return output


    def get_loss(self, input, target, criterion, **kwargs):
        y_pred = self(input, **kwargs)
        if self.return_indices:
            y_pred = y_pred[0]
        return criterion(y_pred, target)


    def get_regularization(self, source = ["weight", "bias"], mode = "L1", **kwargs):
        if not isinstance(source, list):
            source = [source]
        reg = Variable(torch.FloatTensor([0]), requires_grad = False)
        if self.is_cuda:
            reg = reg.cuda()
        for k in range(self.num_layers):
            if self.struct_param[k][1] not in self.param_available:
                continue
            layer = getattr(self, "layer_{0}".format(k))
            for source_ele in source:
                if source_ele == "weight":
                    if self.struct_param[k][1] not in ["Simple_Layer"]:
                        item = layer.weight
                    else:
                        item = layer.W_core
                elif source_ele == "bias":
                    if self.struct_param[k][1] not in ["Simple_Layer"]:
                        item = layer.bias
                    else:
                        item = layer.b_core
                if mode == "L1":
                    reg = reg + item.abs().sum()
                elif mode == "L2":
                    reg = reg + (item ** 2).sum()
                else:
                    raise Exception("mode {0} not recognized!".format(mode))
        return reg


    def get_weights_bias(self, W_source = "core", b_source = "core"):
        W_list = []
        b_list = []
        for k in range(self.num_layers):
            if self.struct_param[k][1] == "Simple_Layer":
                layer = getattr(self, "layer_{0}".format(k))
                if W_source == "core":
                    W_list.append(to_np_array(layer.W_core))
                if b_source == "core":
                    b_list.append(to_np_array(layer.b_core))
            elif self.struct_param[k][1] in self.param_available:
                layer = getattr(self, "layer_{0}".format(k))
                if W_source == "core":
                    W_list.append(to_np_array(layer.weight))
                if b_source == "core":
                    b_list.append(to_np_array(layer.bias))
            else:
                if W_source == "core":
                    W_list.append(None)
                if b_source == "core":
                    b_list.append(None)
        return W_list, b_list


    @property
    def model_dict(self):
        model_dict = {"type": "ConvNet"}
        model_dict["net_type"] = "ConvNet"
        model_dict["input_channels"] = self.input_channels
        model_dict["struct_param"] = self.struct_param
        model_dict["settings"] = self.settings
        model_dict["weights"], model_dict["bias"] = self.get_weights_bias(W_source = "core", b_source = "core")
        model_dict["return_indices"] = self.return_indices
        return model_dict
    

    def load_model_dict(self, model_dict):
        new_net = load_model_dict_net(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(new_net.__dict__)


    def prepare_inspection(self, X, y):
        pred_prob = self(X)
        if self.return_indices:
            pred_prob = pred_prob[0]
        pred = pred_prob.max(1)[1]
        self.info_dict["accuracy"] = get_accuracy(pred, y)
    
    
    def set_cuda(self, is_cuda):
        for k in range(self.num_layers):
            if self.struct_param[k][1] == "Simple_Layer":
                getattr(self, "layer_{0}".format(k)).set_cuda(is_cuda)
            elif self.struct_param[k][1] in self.param_available:
                if is_cuda is True:
                    getattr(self, "layer_{0}".format(k)).cuda()
                else:
                    getattr(self, "layer_{0}".format(k)).cpu()
        self.is_cuda = is_cuda


    def set_trainable(self, is_trainable):
        for k in range(self.num_layers):
            layer = getattr(self, "layer_{0}".format(k))
            if self.struct_param[k][1] == "Simple_Layer":
                layer.set_trainable(is_trainable)
            elif self.struct_param[k][1] in self.param_available:
                for param in layer.parameters():
                    param.requires_grad = is_trainable


class Conv_Autoencoder(nn.Module):
    def __init__(
        self,
        input_channels_encoder,
        input_channels_decoder,
        struct_param_encoder,
        struct_param_decoder,
        latent_size = (1,2),
        share_model_among_steps = False,
        settings = {},
        is_cuda = False,
        ):
        super(Conv_Autoencoder, self).__init__()
        self.input_channels_encoder = input_channels_encoder
        self.input_channels_decoder = input_channels_decoder
        self.struct_param_encoder = struct_param_encoder
        self.struct_param_decoder = struct_param_decoder
        self.share_model_among_steps = share_model_among_steps
        self.settings = settings
        self.encoder = ConvNet(input_channels = input_channels_encoder, struct_param = struct_param_encoder, settings = settings, is_cuda = is_cuda)
        self.decoder = ConvNet(input_channels = input_channels_decoder, struct_param = struct_param_decoder, settings = settings, is_cuda = is_cuda)
        self.is_cuda = is_cuda
    
    def encode(self, input):
        if self.share_model_among_steps:
            latent = []
            for i in range(input.shape[1]):
                latent_step = self.encoder(input[:, i:i+1])
                latent.append(latent_step)
            return torch.cat(latent, 1)
        else:
            return self.encoder(input)
    
    def decode(self, latent):
        if self.share_model_among_steps:
            latent_size = self.struct_param_encoder[-1][0]
            latent = latent.view(latent.size(0), -1, latent_size)
            output = []
            for i in range(latent.shape[1]):
                output_step = self.decoder(latent[:, i].contiguous())
                output.append(output_step)
            return torch.cat(output, 1)
        else:
            return self.decoder(latent)
    
    def set_trainable(self, is_trainable):
        self.encoder.set_trainable(is_trainable)
        self.decoder.set_trainable(is_trainable)
    
    def forward(self, input):
        return self.decode(self.encode(input))
    
    def get_loss(self, input, target, criterion, **kwargs):
        return criterion(self(input), target)
    
    def get_regularization(self, source = ["weight", "bias"], mode = "L1"):
        return self.encoder.get_regularization(source = source, mode = mode) +                self.decoder.get_regularization(source = source, mode = mode)
    
    @property
    def model_dict(self):
        model_dict = {"type": "Conv_Autoencoder"}
        model_dict["net_type"] = "Conv_Autoencoder"
        model_dict["input_channels_encoder"] = self.input_channels_encoder
        model_dict["input_channels_decoder"] = self.input_channels_decoder
        model_dict["struct_param_encoder"] = self.struct_param_encoder
        model_dict["struct_param_decoder"] = self.struct_param_decoder
        model_dict["share_model_among_steps"] = self.share_model_among_steps
        model_dict["settings"] = self.settings
        model_dict["encoder"] = self.encoder.model_dict
        model_dict["decoder"] = self.decoder.model_dict
        return model_dict
    
    def load_model_dict(self, model_dict):
        model = load_model_dict(model_dict, is_cuda = self.is_cuda)
        self.__dict__.update(model.__dict__)



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

