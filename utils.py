import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.stateless import functional_call
from sklearn.metrics import accuracy_score
from collections import namedtuple


# -------------------------------------------------------------------
os.system('nvidia-smi -q -d Memory |grep -A6 GPU|grep Free >tmp')
memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
gpu_number = int(np.argmax(memory_available))
torch.cuda.set_device(gpu_number)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# -------------------------------------------------------------------
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# -------------------------------------------------------------------
Task = namedtuple("Task", ["X_sp", "y_sp", "X_qr", "y_qr", "nb_augs"])


# -------------------------------------------------------------------
def accuracy(pred, y_true):
    y_pred = pred.argmax(1).reshape(-1).cpu()
    y_true = y_true.reshape(-1).cpu()
    return accuracy_score(y_pred, y_true)


# -------------------------------------------------------------------
def func_call(model, params_dict, tsk):
    if params_dict is None: params_dict = dict(model.named_parameters())
    X = torch.cat((tsk.X_sp, tsk.X_qr))
    y = functional_call(model, params_dict, X)
    return y[:len(tsk.X_sp)], y[len(tsk.X_sp):]


# -------------------------------------------------------------------
class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# -------------------------------------------------------------------
def save_object(object, name):
    if isinstance(object, dict):
        for k, v in object.items():
            obj = object[k]
            if isinstance(obj, np.ndarray):
                object[k] = v.tolist()
            if isinstance(obj, dict):
                for k2, v2 in object[k].items():
                    if isinstance(v2, np.ndarray):
                        object[k][k2] = v2.tolist()
    with open(name, 'w') as file:
        json.dump(object, file)

    return


# -------------------------------------------------------------------
def plot_evaluation_results(result_avg, result_std, ds_name, save_dir):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 12})
    ax = plt.subplot(111)
    for name, avg in result_avg.items():
        plt.plot(avg, label=name)
        if result_std is not None:
            std = result_std[name]
            ax.fill_between(range(len(avg)), avg - std, avg + std, alpha=0.5)

    ax.set_xlabel("Adaptation steps (on the new tasks)")
    ax.set_ylabel("Accuracy")
    ax.set_title(ds_name)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save_dir:
        plt.savefig(save_dir)
    plt.show()


