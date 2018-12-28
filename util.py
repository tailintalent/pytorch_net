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


def to_np_array(*arrays, **kwargs):
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
        if array.shape == (1,):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list


def to_Variable(*arrays, **kwargs):
    """Transform numpy arrays into torch tensors/Variables"""
    is_cuda = kwargs["is_cuda"] if "is_cuda" in kwargs else False
    requires_grad = kwargs["requires_grad"] if "requires_grad" in kwargs else False
    array_list = []
    for array in arrays:
        if isinstance(array, int):
            array = [array]
        if isinstance(array, np.ndarray) or isinstance(array, list):
            array = torch.tensor(array).float()
        if isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor):
            array = Variable(array, requires_grad = requires_grad)
        if is_cuda:
            array = array.cuda()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list


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
    """Initialize the weights"""
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
    """Initialize the bias"""
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
    """Get activation"""
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
    """Mean absolute loss"""
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
    """Get loss function"""
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
    """Get optimizer"""
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
    """Class for monitoring and suggesting early stopping"""
    def __init__(self, patience = 100, epsilon = 0, mode = "min"):
        self.patience = patience
        self.epsilon = epsilon
        self.mode = mode
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

    
def flatten(*tensors):
    """Flatten the tensor except the first dimension"""
    new_tensors = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            new_tensor = tensor.contiguous().view(tensor.shape[0], -1)
        elif isinstance(tensor, np.ndarray):
            new_tensor = tensor.reshape(tensor.shape[0], -1)
        else:
            print(new_tensor)
            raise Exception("tensors must be either torch.Tensor or np.ndarray!")
        new_tensors.append(new_tensor)
    if len(new_tensors) == 1:
        new_tensors = new_tensors[0]
    return new_tensors


def expand_indices(vector, expand_size):
    """Expand each element ele in the vector to range(ele * expand_size, (ele + 1) * expand_size)"""
    assert isinstance(vector, torch.Tensor)
    vector *= expand_size
    vector_expand = [vector + i for i in range(expand_size)]
    vector_expand = torch.stack(vector_expand, 0)
    vector_expand = vector_expand.transpose(0, 1).contiguous().view(-1)
    return vector_expand


def to_one_hot(idx, num):
    """Transform a 1D vector into a one-hot vector with num classes"""
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
    """Split the dataset into training and testing sets"""
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


def make_dir(filename):
    """Make directory using filename if the directory does not exist"""
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


        
def get_accuracy(pred, target):
    """Get accuracy from prediction and target"""
    assert len(pred.shape) == len(target.shape) == 1
    assert len(pred) == len(target)
    pred, target = to_np_array(pred, target)
    accuracy = ((pred == target).sum().astype(float) / len(pred))
    return accuracy


def normalize_tensor(X, new_range = None, mean = None, std = None):
    """Normalize the tensor's value range to new_range"""
    X = X.float()
    if new_range is not None:
        assert mean is None and std is None
        X_min, X_max = X.min().item(), X.max().item()
        X_normalized = (X - X_min) / float(X_max - X_min)
        X_normalized = X_normalized * (new_range[1] - new_range[0]) + new_range[0]
    else:
        X_mean = X.mean().item()
        X_std = X.std().item()
        X_normalized = (X - X_mean) / X_std
        X_normalized = X_normalized * std + mean
    return X_normalized


def get_args(arg, arg_id = 1, type = "str"):
    """get sys arguments from either command line or Jupyter"""
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        arg_return = arg
    except:
        import sys
        try:
            arg_return = sys.argv[arg_id]
            if type == "int":
                arg_return = int(arg_return)
            elif type == "float":
                arg_return = float(arg_return)
            elif type == "bool":
                arg_return = eval(arg_return)
            elif type == "eval":
                arg_return = eval(arg_return)
            elif type == "tuple":
                if arg_return[0] not in ["(", "["]:
                    arg_return = eval(arg_return)
                else:
                    splitted = arg_return[1:-1].split(",")
                    List = []
                    for item in splitted:
                        try:
                            item = eval(item)
                        except:
                            pass
                        List.append(item)
                    arg_return = tuple(List)
            elif type == "str":
                pass
            else:
                raise Exception("type {0} not recognized!".format(type))
        except:
#             raise
            arg_return = arg
    return arg_return


class Loss_with_uncertainty(nn.Module):
    def __init__(self, core = "mse", epsilon = 1e-6):
        super(Loss_with_uncertainty, self).__init__()
        self.name = "Loss_with_uncertainty"
        self.core = core
        self.epsilon = epsilon
    
    def forward(self, pred, target, log_std = None, std = None, sample_weights = None, is_mean = True):
        if self.core == "mse":
            loss_core = get_criterion(self.core, reduce = False)(pred, target) / 2
        elif self.core == "mae":
            loss_core = get_criterion(self.core, reduce = False)(pred, target)
        elif self.core == "huber":
            loss_core = get_criterion(self.core, reduce = False)(pred, target)
        elif self.core == "mlse":
            loss_core = torch.log((target - pred) ** 2 + 1e-10)
        elif self.core == "mse+mlse":
            loss_core = (target - pred) ** 2 / 2 + torch.log((target - pred) ** 2 + 1e-10)
        else:
            raise Exception("loss's core {0} not recognized!".format(self.core))
        if std is not None:
            assert log_std is None
            loss = loss_core / (self.epsilon + std ** 2) + torch.log(std + 1e-7)
        else:
            loss = loss_core / (self.epsilon + torch.exp(2 * log_std)) + log_std
        if sample_weights is not None:
            sample_weights = sample_weights.view(loss.size())
            loss = loss * sample_weights
        if is_mean:
            loss = loss.mean()
        return loss


def expand_tensor(tensor, dim, times):
    """Repeat the value of a tensor locally along the given dimension"""
    if isinstance(times, int) and times == 1:
        return tensor
    if dim < 0:
        dim += len(tensor.size())
    assert dim >= 0
    size = list(tensor.size())
    repeat_times = [1] * (len(size) + 1)
    repeat_times[dim + 1] = times
    size[dim] = size[dim] * times
    return tensor.unsqueeze(dim + 1).repeat(repeat_times).view(*size)


def shrink_boolean_tensor(tensor, dim, shrink_ratio, mode = "any"):
    assert tensor.dtype == "bool"
    shape = tuple(tensor.shape)
    if dim < 0:
        dim += len(tensor.shape)
    assert shape[dim] % shrink_ratio == 0
    new_dim = int(shape[dim] / shrink_ratio)
    new_shape = shape[:dim] + (new_dim, shrink_ratio) + shape[dim+1:]
    new_tensor = np.reshape(tensor, new_shape)
    if mode == "any":
        return new_tensor.any(dim + 1)
    elif mode == "all":
        return new_tensor.all(dim + 1)


def permute_dim(X, dim, idx, group_sizes, mode = "permute"):
    from copy import deepcopy
    assert dim != 0
    device = torch.device("cuda" if X.is_cuda else "cpu")
    if isinstance(idx, tuple) or isinstance(idx, list):
        k, ll = idx
        X_permute = X[:, k, ll * group_sizes: (ll + 1) * group_sizes]
        num = X_permute.size(0)
        if mode == "permute":
            new_idx = torch.randperm(num).to(device)
        elif mode == "resample":
            new_idx = torch.randint(num, size = (num,)).long().to(device)
        else:
            raise
        X_permute = X_permute.index_select(0, new_idx)
        X_new = deepcopy(X)
        X_new[:, k, ll * group_sizes: (ll + 1) * group_sizes] = X_permute
    else:
        X_permute = X.index_select(dim, torch.arange(idx * group_sizes, (idx + 1) * group_sizes).long().to(device))
        num = X_permute.size(0)
        if mode == "permute":
            new_idx = torch.randperm(num).to(device)
        elif mode == "resample":
            new_idx = torch.randint(num, size = (num,)).long().to(device)
        else:
            raise
        X_permute = X_permute.index_select(0, new_idx)
        X_new = deepcopy(X)
        if dim == 1:
            X_new[:,idx*group_sizes: (idx+1)*group_sizes] = X_permute
        elif dim == 2:
            X_new[:,:,idx*group_sizes: (idx+1)*group_sizes] = X_permute
        elif dim == 3:
            X_new[:,:,:,idx*group_sizes: (idx+1)*group_sizes] = X_permute
        else:
            raise
    return X_new