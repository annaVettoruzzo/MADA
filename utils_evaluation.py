from methods.metatest import adapt_and_evaluate
from models import SimpleCNNModule, ResNetBaseline
from utils import DEVICE, save_object

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json


# -------------------------------------------------------------------
def evaluate_classification_model(tgen, model, loss_fn, lr_inner, steps=20, nb_tasks=100, with_aug=False):
    lst = []

    for _ in range(nb_tasks):
        tsk = tgen.batch()
        history = adapt_and_evaluate(model, loss_fn, lr_inner, tsk, steps=steps, with_aug=with_aug)
        lst.append(history["acc"])

    return np.array(lst).mean(axis=0)


# -------------------------------------------------------------------
def evaluate_classification_models(tgen, models_dict, loss_fn, lr_inner, steps=20, nb_tasks=100, folder=""):
    results = {}

    for name, model in models_dict.items():
        print(f" {name}", end="")
        with_aug = ("Aug" in name)
        res = evaluate_classification_model(tgen, model, loss_fn, lr_inner, steps=steps, nb_tasks=nb_tasks, with_aug=with_aug)
        results[name] = res

    save_object(results, f"{folder}/k{tgen.k}/test_eval.json")

    return results


# -------------------------------------------------------------------
def load_classification_models(dirpath, model_name, n_classes):
    PATH = Path(dirpath)

    if model_name == 'cnn':
        models_dict = {
            "scratch": SimpleCNNModule(n_classes).to(DEVICE),
            "scratchAug": SimpleCNNModule(n_classes).to(DEVICE),
            "MAML": SimpleCNNModule(n_classes).to(DEVICE),
            "MAMLAugTs": SimpleCNNModule(n_classes).to(DEVICE),
            "MAMLAugTrTs": SimpleCNNModule(n_classes).to(DEVICE),
            "MAMLAugTrTsW": SimpleCNNModule(n_classes).to(DEVICE),
        }
    elif model_name == 'resnet':
        models_dict = {
            "scratch": ResNetBaseline(n_classes=n_classes).to(DEVICE),
            "scratchAug": ResNetBaseline(n_classes=n_classes).to(DEVICE),
            "MAML": ResNetBaseline(n_classes=n_classes).to(DEVICE),
            "MAMLAugTs": ResNetBaseline(n_classes=n_classes).to(DEVICE),
            "MAMLAugTrTs": ResNetBaseline(n_classes=n_classes).to(DEVICE),
            "MAMLAugTrTsW": ResNetBaseline(n_classes=n_classes).to(DEVICE),
        }

    delete = []
    for name, model in models_dict.items():
        try:
            alt_name = name.split("Aug")[0] if name == "scratchAug" or name == "MAMLAugTs" else name
            params = torch.load(PATH / alt_name, map_location=DEVICE)
            model.load_state_dict(params)

            if "W" in name:
                model.weights = (torch.load(PATH / (name + "_weights"), map_location=DEVICE))

        except FileNotFoundError as err:
            print(err)
            delete.append(name)

    for k in delete: del models_dict[k]

    return models_dict


# -------------------------------------------------------------------
def evaluate_classification_seeds(tgen, folders, model_name, loss_fn, lr_inner, n_classes, steps=20, nb_tasks=100):
    final_results = defaultdict(list)

    for dirpath in folders:
        print(f"\nEvaluating {dirpath}: ", end="")
        models_dict = load_classification_models(dirpath, model_name, n_classes)
        results = evaluate_classification_models(tgen, models_dict, loss_fn, lr_inner, steps=steps, nb_tasks=nb_tasks, folder=dirpath)
        for k, v in results.items():
            final_results[k].append(v)

    result_avg = {k: np.array(v).mean(axis=0) for k, v in final_results.items()}
    result_std = {k: np.array(v).std(axis=0) for k, v in final_results.items()}
    return result_avg, result_std

