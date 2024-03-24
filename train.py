import copy
import gc
from utils import DEVICE
from TaskGenerator import TaskGenerator
from TransferLearningDataset import TLOneSubject, TLTrainingSubjects
from torch.utils.data import DataLoader
from models import SimpleCNNModule, ResNetBaseline
from methods import MAMLAug, ANILAug, MAML, model_training
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
    PATH = Path(f"results/{config_params['dataset_name']}/{config_params['model']}/seed{config_params['seed']}")
    print(PATH)
    PATH.mkdir(parents=True, exist_ok=True)

    print(DEVICE)
    print(torch.cuda.current_device())

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
    
    # TRANSFER LEARNING with one subject
    print('Transfer Learning with one subject')
    dataset = TLOneSubject(tgen)  # Randomly select one subject and create a dataset
    dataloader = DataLoader(dataset, batch_size=25, shuffle=True)
    model_trlearning = SimpleCNNModule(n_classes=len(dataset.classes)).to(DEVICE) if config_params['model'] == 'cnn' else ResNetBaseline(n_classes=len(dataset.classes)).to(DEVICE)
    model_trlearning = model_training(dataloader, model_trlearning, loss_fn, config_params['lr'], config_params['steps'])
    torch.save(model_trlearning.state_dict(), PATH / 'TRLearning_onesubject')
    del model_trlearning
    gc.collect()

    # TRANSFER LEARNING with all training subjects
    print('Transfer Learning with all training subjects')
    dataset = TLTrainingSubjects(tgen)
    dataloader = DataLoader(dataset, batch_size=25, shuffle=True)
    model_trlearning = SimpleCNNModule(n_classes=len(dataset.classes)).to(DEVICE) if config_params['model'] == 'cnn' else ResNetBaseline(n_classes=len(dataset.classes)).to(DEVICE)
    model_trlearning = model_training(dataloader, model_trlearning, loss_fn, config_params['lr'], config_params['steps'])
    torch.save(model_trlearning.state_dict(), PATH / 'TRLearning')
    del model_trlearning
    gc.collect()

    # MAML
    print('MAML')
    model_maml = copy.deepcopy(model).to(DEVICE)
    MAML(model_maml, loss_fn, config_params['lr_inner'],
         config_params['lr'], config_params['adapt_steps']).fit(tgen, steps=config_params['steps'])
    torch.save(model_maml.state_dict(), PATH / 'MAML')
    del model_maml
    gc.collect()

    # MAMLAUG at TR and TS
    print('MAML with Augmentations')
    model_mamlaug = copy.deepcopy(model).to(DEVICE)
    MAMLAug(model_mamlaug, loss_fn, config_params['lr_inner'], NB_AUGS,
            config_params['lr'], config_params['adapt_steps'], with_weights=False).fit(tgen, steps=config_params['steps'])
    torch.save(model_mamlaug.state_dict(), PATH / 'MAMLAugTrTs')
    del model_mamlaug
    gc.collect()

    # MAMLAUG with WEIGHTS
    print('MAML with Augmentations and Weights')
    model_mamlaug_w = copy.deepcopy(model).to(DEVICE)
    MAMLAug(model_mamlaug_w, loss_fn, config_params['lr_inner'], NB_AUGS,
            config_params['lr'], config_params['adapt_steps'], with_weights=True).fit(tgen, steps=config_params['steps'])
    torch.save(model_mamlaug_w.state_dict(), PATH / 'MAMLAugTrTsW')
    torch.save(model_mamlaug_w.weights, PATH / 'MAMLAugTrTsW_weights')
    del model_mamlaug_w
    gc.collect()

    """ De-comment if ANIL
    # ANILAUG at TR and TS (last two layers)
    print('ANIL with Augmentations')
    model_anilaug = copy.deepcopy(model).to(DEVICE)
    ANILAug(model_anilaug, loss_fn, config_params['lr_inner'], NB_AUGS,
            config_params['lr'], config_params['adapt_steps'], anil_last=False, with_weights=False).fit(tgen, steps=config_params['steps'])
    torch.save(model_anilaug.state_dict(), PATH / 'ANILAugTrTs')
    del model_anilaug
    gc.collect()

    # ANILAUG with WEIGHTS (last two layers)
    print('ANIL with Augmentations and Weights')
    model_anilaug_w = copy.deepcopy(model).to(DEVICE)
    ANILAug(model_anilaug_w, loss_fn, config_params['lr_inner'], NB_AUGS,
            config_params['lr'], config_params['adapt_steps'], anil_last=False, with_weights=True).fit(tgen, steps=config_params['steps'])
    torch.save(model_anilaug_w.state_dict(), PATH / 'ANILAugTrTsW')
    torch.save(model_anilaug_w.weights, PATH / 'ANILAugTrTsW_weights')
    del model_anilaug_w
    gc.collect()

    # ANILAUG at TR and TS (last layer)
    print('ANIL with Augmentations')
    model_anilaug = copy.deepcopy(model).to(DEVICE)
    ANILAug(model_anilaug, loss_fn, config_params['lr_inner'], NB_AUGS,
            config_params['lr'], config_params['adapt_steps'], anil_last=True, with_weights=False).fit(tgen, steps=config_params['steps'])
    torch.save(model_anilaug.state_dict(), PATH / 'ANILAugTrTs_last')
    del model_anilaug
    gc.collect()

    # ANILAUG with WEIGHTS (last layer)
    print('ANIL with Augmentations and Weights')
    model_anilaug_w = copy.deepcopy(model).to(DEVICE)
    ANILAug(model_anilaug_w, loss_fn, config_params['lr_inner'], NB_AUGS,
            config_params['lr'], config_params['adapt_steps'], anil_last=True, with_weights=True).fit(tgen, steps=config_params['steps'])
    torch.save(model_anilaug_w.state_dict(), PATH / 'ANILAugTrTsW_last')
    torch.save(model_anilaug_w.weights, PATH / 'ANILAugTrTsW_last_weights')
    del model_anilaug_w
    gc.collect()
    """

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
                        help='Number of examples per class in support set')  # k=1 always because we want to consider only 1s
    parser.add_argument('--q', type=int, default=100,
                        help='Number of examples per class in query set')

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

    args = parser.parse_args()

    train(args)
