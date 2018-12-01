from __future__ import print_function
import numpy as np
from copy import deepcopy
import random
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import _Loss


def plot_matrices(
    matrix_list, 
    shape = None, 
    images_per_row = 10, 
    scale_limit = None,
    figsize = (20, 8), 
    x_axis_list = None,
    filename = None,
    title = None,
    subtitles = [],
    highlight_bad_values = True,
    plt = None,
    pdf = None,
    ):
    """Plot the images for each matrix in the matrix_list."""
    import matplotlib
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize = figsize)
    fig.set_canvas(plt.gcf().canvas)
    if title is not None:
        fig.suptitle(title, fontsize = 18, horizontalalignment = 'left', x=0.1)
    
    matrix_list = np.array([to_np_array(matrix_list[i]) for i in range(len(matrix_list))])
    num_matrixs = len(matrix_list)
    rows = np.ceil(num_matrixs / float(images_per_row))
    try:
        matrix_list_reshaped = np.reshape(np.array(matrix_list), (-1, shape[0],shape[1])) \
            if shape is not None else np.array(matrix_list)
    except:
        matrix_list_reshaped = matrix_list
    if scale_limit == "auto":
        scale_min = np.Inf
        scale_max = -np.Inf
        for matrix in matrix_list:
            scale_min = min(scale_min, np.min(matrix))
            scale_max = max(scale_max, np.max(matrix))
        scale_limit = (scale_min, scale_max)
    for i in range(len(matrix_list)):
        ax = fig.add_subplot(rows, images_per_row, i + 1)
        image = matrix_list_reshaped[i].astype(float)
        if len(image.shape) == 1:
            image = np.expand_dims(image, 1)
        if highlight_bad_values:
            cmap = matplotlib.cm.binary
            cmap.set_bad('red', alpha = 0.2)
            mask_key = []
            mask_key.append(np.isnan(image))
            mask_key.append(np.isinf(image))
            mask_key = np.any(np.array(mask_key), axis = 0)
            image = np.ma.array(image, mask = mask_key)
        else:
            cmap = matplotlib.cm.binary
        if scale_limit is None:
            ax.matshow(image, cmap = cmap)
        else:
            assert len(scale_limit) == 2, "scale_limit should be a 2-tuple!"
            ax.matshow(image, cmap = cmap, vmin = scale_limit[0], vmax = scale_limit[1])
        if len(subtitles) > 0:
            ax.set_title(subtitles[i])
        try:
            xlabel = "({0:.4f},{1:.4f})\nshape: ({2}, {3})".format(np.min(image), np.max(image), image.shape[0], image.shape[1])
            if x_axis_list is not None:
                xlabel += "\n{0}".format(x_axis_list[i])
            plt.xlabel(xlabel)
        except:
            pass
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
    if pdf is not None:
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    else:
        plt.show()

    if scale_limit is not None:
        print("scale_limit: ({0:.6f}, {1:.6f})".format(scale_limit[0], scale_limit[1]))
    print()


def record_data(data_record_dict, data_list, key_list, nolist = False):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of 
    key_list and data_list into the dictionary."""
    assert len(data_list) == len(key_list), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        if nolist:
            data_record_dict[key] = data
        else:
            if key not in data_record_dict:
                data_record_dict[key] = [data]
            else: 
                data_record_dict[key].append(data)


def to_np_array(*arrays):
    array_list = []
    for array in arrays:
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor) or \
           isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list


def to_Variable(*arrays, is_cuda = False, requires_grad = False):
    array_list = []
    for array in arrays:
        if isinstance(array, list):
            array = np.array(array)
        if isinstance(array, np.ndarray):
            array = torch.Tensor(array)
        if isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor):
            array = Variable(array, requires_grad = requires_grad)
        if is_cuda:
            array = array.cuda()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list


def make_dir(filename):
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc)
                raise



def init_module_weights(module_list, init_weights_mode = "glorot-normal"):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        if init_weights_mode == "glorot-uniform":
            glorot_uniform_limit = np.sqrt(6 / float(module.in_features + module.out_features))
            module.weight.data.uniform_(-glorot_uniform_limit, glorot_uniform_limit)
        elif init_weights_mode == "glorot-normal":
            glorot_normal_std = np.sqrt(2 / float(module.in_features + module.out_features))
            module.weight.data.normal_(mean = 0, std = glorot_normal_std)
        else:
            raise Exception("init_weights_mode '{0}' not recognized!".format(init_weights_mode))


def init_module_bias(module_list, init_bias_mode = "zeros"):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        if init_bias_mode == "zeros":
            module.bias.data.fill_(0)
        else:
            raise Exception("init_bias_mode '{0}' not recognized!".format(init_bias_mode))


def init_weight(weight_list, init):
    if not isinstance(weight_list, list):
        weight_list = [weight_list]
    for weight in weight_list:
        if len(weight.size()) == 2:
            rows = weight.size(0)
            columns = weight.size(1)
        elif len(weight.size()) == 1:
            rows = 1
            columns = weight.size(0)
        if init is None:
            init = "glorot-normal"
        if not isinstance(init, str):
            weight.data.copy_(torch.FloatTensor(init))
        else:
            if init == "glorot-normal":
                glorot_normal_std = np.sqrt(2 / float(rows + columns))
                weight.data.normal_(mean = 0, std = glorot_normal_std)
            else:
                raise Exception("init '{0}' not recognized!".format(init))

def init_bias(bias_list, init):
    if not isinstance(bias_list, list):
        bias_list = [bias_list]
    for bias in bias_list:
        if init is None:
            init = "zeros"
        if not isinstance(init, str):
            bias.data.copy_(torch.FloatTensor(init))
        else:
            if init == "zeros":
                bias.data.fill_(0)
            else:
                raise Exception("init '{0}' not recognized!".format(init))


def get_activation(activation):
    if activation == "linear":
        f = lambda x: x
    elif activation == "relu":
        f = F.relu
    elif activation == "leakyRelu":
        f = nn.LeakyReLU(negative_slope = 0.3)
    elif activation == "leakyReluFlat":
        f = nn.LeakyReLU(negative_slope = 0.01)
    elif activation == "tanh":
        f = F.tanh
    elif activation == "softplus":
        f = F.softplus
    elif activation == "sigmoid":
        f = F.sigmoid
    elif activation == "selu":
        f = F.selu
    elif activation == "elu":
        f = F.elu
    elif activation == "sign":
        f = lambda x: torch.sign(x)
    elif activation == "heaviside":
        f = lambda x: (torch.sign(x) + 1) / 2.
    else:
        raise Exception("activation {0} not recognized!".format(activation))
    return f


class MAELoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(MAELoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        assert not target.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"
        loss = (input - target).abs()
        if self.reduce:
            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss.sum()
        return loss


def get_criterion(loss_type, reduce = True, **kwargs):
    if loss_type == "huber":
        criterion = nn.SmoothL1Loss(reduce = reduce)
    elif loss_type == "mse":
        criterion = nn.MSELoss(reduce = reduce)
    elif loss_type == "mae":
        criterion = MAELoss(reduce = reduce)
    elif loss_type == "cross-entropy":
        criterion = nn.CrossEntropyLoss(reduce = reduce)
    elif loss_type == "Loss_with_uncertainty":
        criterion = Loss_with_uncertainty(core = kwargs["loss_core"] if "loss_core" in kwargs else "mse", epsilon = 1e-6)
    else:
        raise Exception("loss_type {0} not recognized!".format(loss_type))
    return criterion


def get_optimizer(optim_type, lr, parameters):
    if optim_type == "adam":
        optimizer = optim.Adam(parameters, lr = lr)
    elif optim_type == "RMSprop":
        optimizer = optim.RMSprop(parameters, lr = lr)
    elif optim_type == "LBFGS":
        optimizer = optim.LBFGS(parameters, lr = lr)
    else:
        raise Exception("optim_type {0} not recognized!".format(optim_type))
    return optimizer


def get_full_struct_param_ele(struct_param, settings):
    struct_param_new = deepcopy(struct_param)
    for i, layer_struct_param in enumerate(struct_param_new):
        if settings is not None and layer_struct_param[1] != "Symbolic_Layer":
            layer_struct_param[2] = {key: value for key, value in deepcopy(settings).items() if key in ["activation"]}
            layer_struct_param[2].update(struct_param[i][2])
        else:
            layer_struct_param[2] = deepcopy(struct_param[i][2])
    return struct_param_new


def get_full_struct_param(struct_param, settings):
    struct_param_new_list = []
    if isinstance(struct_param, tuple):
        for i, struct_param_ele in enumerate(struct_param):
            if isinstance(settings, tuple):
                settings_ele = settings[i]
            else:
                settings_ele = settings
            struct_param_new_list.append(get_full_struct_param_ele(struct_param_ele, settings_ele))
        return tuple(struct_param_new_list)
    else:
        return get_full_struct_param_ele(struct_param, settings)


class Early_Stopping(object):
    def __init__(self, patience = 100, epsilon = 0, mode = "min"):
        self.patience = patience
        self.epsilon = epsilon
        self.mode = "min"
        self.best_value = None
        self.wait = 0
        
    def monitor(self, value):
        to_stop = False
        if self.patience is not None:
            if self.best_value is None:
                self.best_value = value
                self.wait = 0
            else:
                if (self.mode == "min" and value < self.best_value - self.epsilon) or \
                   (self.mode == "max" and value > self.best_value + self.epsilon):
                    self.best_value = value
                    self.wait = 0
                else:
                    if self.wait >= self.patience:
                        to_stop = True
                    else:
                        self.wait += 1
        return to_stop

    
def flatten(tensor):
    return tensor.view(tensor.size(0), -1)


def to_one_hot(idx, num):
    if len(idx.size()) == 1:
        idx = idx.unsqueeze(-1)
    if not isinstance(idx, Variable):
        if isinstance(idx, np.ndarray):
            idx = torch.LongTensor(idx)
        idx = Variable(idx, requires_grad = False)
    onehot = Variable(torch.zeros(idx.size(0), num), requires_grad = False)
    if idx.is_cuda:
        onehot = onehot.cuda()
    onehot.scatter_(1, idx, 1)
    return onehot


def train_test_split(X, y, test_size = 0.1):
    import torch
    num_examples = len(X)
    if test_size is not None:
        num_test = int(num_examples * test_size)
        num_train = num_examples - num_test
        idx_train = np.random.choice(range(num_examples), size = num_train, replace = False)
        idx_test = set(range(num_examples)) - set(idx_train)
        device = torch.device("cuda" if X.is_cuda else "cpu")
        idx_train = torch.LongTensor(list(idx_train)).to(device)
        idx_test = torch.LongTensor(list(idx_test)).to(device)
        X_train = X[idx_train]
        y_train = y[idx_train]
        X_test = X[idx_test]
        y_test = y[idx_test]
    else:
        X_train, X_test = X, X
        y_train, y_test = y, y
    return (X_train, y_train), (X_test, y_test)