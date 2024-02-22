from TaskGenerator import TaskGenerator
from utils import plot_evaluation_results, save_object
from utils_evaluation import evaluate_classification_seeds, save_object

import torch, random
from pathlib import Path
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")


def save_results(dict_avg, dict_std, save_dir):
    res = {"dict_avg": dict_avg, "dict_std": dict_std}
    save_object(res, save_dir/'result.json')

    avg = {name: np.mean(arr[-3:]) for name, arr in dict_avg.items()}
    std = {name: np.mean(arr[-3:]) for name, arr in dict_std.items()}
    with open(save_dir/'result.csv', "w") as f:
        f.write("{:<15}, {:<12}, {:<12}\n".format("Method", "Avg accuracy", "Std accuracy"))
        for name in dict_avg.keys():
            f.write("{:<15}, {:<12.2f}, {:<12.2f}\n".format(name, 100*avg[name], 100*std[name]))
    return


def test(args):
    # Set default parameters
    config_params = vars(args)

    torch.random.manual_seed(config_params['seed'])
    np.random.seed(config_params['seed'])
    random.seed(config_params['seed'])

    PATH = Path(f"results/{config_params['dataset_name']}/{config_params['model']}/k{config_params['k']}")
    print(PATH)
    PATH.mkdir(parents=True, exist_ok=True)

    # Set dataset
    print("Loading the dataset...")
    tgen = TaskGenerator(n=config_params['n'], k=config_params['k'], q=config_params['q'], background=False, dataset_name=config_params['dataset_name'])

    loss_fn = torch.nn.CrossEntropyLoss()

    folders = [f"results/{config_params['dataset_name']}/{config_params['model']}/seed{seed}" for seed in range(config_params['seeds_done'])]

    dict_avg, dict_std = evaluate_classification_seeds(tgen, folders, config_params['model'],
                                                       loss_fn, config_params['lr_inner'], config_params['n'],
                                                       steps=config_params['steps'], nb_tasks=config_params['nb_test_tasks'])

    plot_evaluation_results(dict_avg, dict_std, config_params['dataset_name'], save_dir=PATH/'results.png')
    save_results(dict_avg, dict_std, save_dir=PATH)


if __name__ == '__main__':
    # Set parameters
    parser = argparse.ArgumentParser(description='Meta-Learning for Time-series data with Augmentations')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--seeds_done', type=int, default=4,
                        help='Seeds used for training')

    # Set dataset params
    parser.add_argument('--dataset_name', type=str, default='PAMAP',
                        choices=['ADL', 'DSA', 'PAMAP', 'REALDISP-ideal', 'REALDISP-mutual', 'REALDISP-self', 'VPA', 'WISDM-phone', 'WISDM-watch'],
                        help='Dataset name')
    parser.add_argument('--n', type=int, default=5,
                        help='Number of classes per task')
    parser.add_argument('--k', type=int, default=1,
                        help='Number of images per class in support set')
    parser.add_argument('--q', type=int, default=100,
                        help='Number of images per class in query set')

    # Set training params
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet'])
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of steps per task')
    parser.add_argument('--lr_inner', type=float, default=1e-2,
                        help='Learning rate inner loop')
    parser.add_argument('--nb_test_tasks', type=int, default=50,
                        help='Number of tasks to test')

    args = parser.parse_args()

    test(args)