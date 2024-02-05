import copy
import gc
from utils import DEVICE
from TaskGenerator import TaskGenerator
from models import SimpleCNNModule, ResNetBaseline
from methods import MAMLAug, MAML
from augmentations import NB_AUGS

import torch, random
import numpy as np
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def train(args):
    # Set default parameters
    config_params = vars(args)

    # For saving models
    PATH = Path(f"results/{config_params['dataset_name']}/seed{config_params['seed']}_{config_params['model']}")
    print(PATH)
    PATH.mkdir(parents=True, exist_ok=True)

    torch.random.manual_seed(config_params['seed'])
    np.random.seed(config_params['seed'])
    random.seed(config_params['seed'])

    # Set dataset
    print("Loading the dataset...")
    tgen = TaskGenerator(n=config_params['n'], k=config_params['k'], q=config_params['q'], background=True, dataset_name=config_params['dataset_name'])

    # Training
    print("Training the models...")
    loss_fn = torch.nn.CrossEntropyLoss()
    if config_params['model'] == 'cnn':
        model = SimpleCNNModule(n_classes=config_params['n']).to(DEVICE)
    elif config_params['model'] == 'resnet':
        model = ResNetBaseline(n_classes=config_params['n']).to(DEVICE)
    else:
        raise NotImplementedError("Invalid or missing input parameter")

    # SCRATCH
    print('SCRATCH')
    torch.save(model.state_dict(), PATH / "scratch")

    # MAML
    print('MAML')
    model_maml = copy.deepcopy(model).to(DEVICE)
    MAML(model_maml, loss_fn, config_params['lr_inner'],
         config_params['lr'], config_params['adapt_steps'],
         config_params['patience']).fit(tgen, steps=config_params['steps'])
    torch.save(model_maml.state_dict(), PATH / 'MAML')
    del model_maml
    gc.collect()

    # MAMLAUG at TR and TS
    print('MAML with Augmentations')
    model_mamlaug = copy.deepcopy(model).to(DEVICE)
    MAMLAug(model_mamlaug, loss_fn, config_params['lr_inner'], NB_AUGS,
            config_params['lr'], config_params['adapt_steps'], config_params['patience'],
            with_weights=False).fit(tgen, steps=config_params['steps'])
    torch.save(model_mamlaug.state_dict(), PATH / 'MAMLAugTrTs')
    del model_mamlaug
    gc.collect()

    # MAMLAUG with WEIGHTS
    print('MAML with Augmentations and Weights')
    model_mamlaug_w = copy.deepcopy(model).to(DEVICE)
    MAMLAug(model_mamlaug_w, loss_fn, config_params['lr_inner'], NB_AUGS,
            config_params['lr'], config_params['adapt_steps'], config_params['patience'],
            with_weights=True).fit(tgen, steps=config_params['steps'])
    torch.save(model_mamlaug_w.state_dict(), PATH / 'MAMLAugTrTsW')
    torch.save(model_mamlaug_w.weights, PATH / 'MAMLAugTrTsW_weights')
    del model_mamlaug_w
    gc.collect()


if __name__ == '__main__':
    # Set parameters
    parser = argparse.ArgumentParser(description='Meta-Learning for Time-series data with Augmentations')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    # Set dataset params
    parser.add_argument('--dataset_name', type=str, default='ADL',
                        choices=['DSA', 'PAMAP', 'WISDM-phone', 'WISDM-watch', 'REALDISP-ideal', 'REALDISP-mutual', 'REALDISP-self', 'VPA', 'ADL'],
                        help='Dataset name')
    parser.add_argument('--n', type=int, default=5,
                        help='Number of classes per task')
    parser.add_argument('--k', type=int, default=1,
                        help='Number of images per class in support set')  # k=1 always because we want to consider only 1s
    parser.add_argument('--q', type=int, default=100,
                        help='Number of images per class in query set')

    # Set training params
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet'])
    parser.add_argument('--steps', type=int, default=50000,
                        help='Number of steps per task')
    parser.add_argument('--adapt_steps', type=int, default=2,
                        help='Number of adaptation steps in the inner loop')
    parser.add_argument('--lr_inner', type=float, default=1e-2,
                        help='Learning rate inner loop')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--patience', type=float, default=200,
                        help='Learning rate inner loop')

    args = parser.parse_args()

    train(args)
